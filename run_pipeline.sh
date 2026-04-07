#!/usr/bin/env bash
# Prompt Distillation pipeline for Llama-3-8B-Instruct (SquadShifts)
# Manages vLLM server lifecycle: starts on VLLM_GPU, trains on TRAIN_GPU.
set -euo pipefail

VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VLLM_HOST="${VLLM_HOST:-localhost}"
DATASET="${DATASET:-amazon}"
RUN_NAME="${RUN_NAME:-pd_${DATASET}}"
MAX_ITEMS="${MAX_ITEMS:-20}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-1000}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-${MODEL}}"
STUDENT_BASE="${STUDENT_BASE:-llama3-8b-instruct}"
TEACHER_BASE="${TEACHER_BASE:-${STUDENT_BASE}}"

echo "=== Prompt Distillation Pipeline ==="
echo "  Teacher Model:    ${TEACHER_MODEL}"
echo "  Student Base:     ${STUDENT_BASE}"
echo "  Teacher Base:     ${TEACHER_BASE}"
echo "  Model:            ${MODEL}"
echo "  vLLM GPU:         ${VLLM_GPU}"
echo "  Train GPU:        ${TRAIN_GPU}"
echo "  Dataset:          ${DATASET}"
echo "  Run name:         ${RUN_NAME}"
echo "  Max items (eval): ${MAX_ITEMS}"
echo "  Max items (train):${MAX_TRAIN_ITEMS}"
echo ""

# --- Start vLLM server on VLLM_GPU ---
echo "Starting vLLM server on GPU ${VLLM_GPU}..."
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python3 -u -m vllm.entrypoints.openai.api_server \
    --model "${TEACHER_MODEL}" --dtype auto --api-key token-abc123 \
    --tensor-parallel-size 1 &
VLLM_PID=$!

# Wait for server to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 120); do
    if curl -s "http://${VLLM_HOST}:8000/health" > /dev/null 2>&1; then
        echo "vLLM server ready."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server process died."
        exit 1
    fi
    sleep 2
done
if ! curl -s "http://${VLLM_HOST}:8000/health" > /dev/null 2>&1; then
    echo "ERROR: vLLM server failed to start within timeout."
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
fi

cleanup() {
    echo "Stopping vLLM server (PID ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# Step 1: Generate questions
echo "[1/8] Generating questions..."
python3 evaluation/sample_questions.py \
    --vllm_hostname "${VLLM_HOST}" \
    --base "${TEACHER_BASE}" \
    --dataset "${DATASET}" \
    --max_items "${MAX_TRAIN_ITEMS}"

# Step 2: Convert generated questions into lesson format
echo "[2/8] Converting questions to lesson format..."
python3 curriculum/csv_to_lesson.py \
    --dataset "${DATASET}" \
    --model "${TEACHER_BASE}" \
    --max_items "${MAX_TRAIN_ITEMS}"

# Step 3: Generate test set questions in lesson format
echo "[3/8] Generating exam..."
python3 curriculum/questions_to_exam.py \
    --max_items "${MAX_ITEMS}" \
    --dataset "${DATASET}"

# Step 4: Generate answers to training questions
echo "[4/8] Generating teacher answers (lesson)..."
python3 curriculum/generate_teacher_answers.py \
    --generate_lesson True \
    --base "${TEACHER_BASE}" \
    --dataset "${DATASET}" \
    --question_model "${TEACHER_BASE}" \
    --max_items "${MAX_TRAIN_ITEMS}" \
    --vllm_hostname "${VLLM_HOST}"

# Step 5: Generate answers to test set questions
echo "[5/8] Generating teacher answers (exam)..."
python3 curriculum/generate_teacher_answers.py \
    --generate_exam True \
    --base "${TEACHER_BASE}" \
    --max_items "${MAX_ITEMS}" \
    --dataset "${DATASET}" \
    --vllm_hostname "${VLLM_HOST}"

# Step 6: Run training on TRAIN_GPU
echo "[6/8] Training on GPU ${TRAIN_GPU}..."
CHECKPOINT_DIR="checkpoints/huggingface/${RUN_NAME}"
if [ -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ]; then
    echo "${CHECKPOINT_DIR}/adapter_model.safetensors already exists — skipping."
else
    TRAIN_ARGS=(
        --run_name "${RUN_NAME}"
        --use_wandb False
        --base "${STUDENT_BASE}"
        --dataset "${DATASET}"
        --lesson_model "${TEACHER_BASE}"
        --exam_model "${TEACHER_BASE}"
        --question_model "${TEACHER_BASE}"
        --max_items_train "${MAX_TRAIN_ITEMS}"
        --max_items_test "${MAX_ITEMS}"
    )
    if [ "${STUDENT_BASE}" != "${TEACHER_BASE}" ]; then
        TRAIN_ARGS+=(--token_loss_weight 1.0 --logit_loss_weight 0.0 --lesson_temp 0.25)
    fi
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 training/run_train_student.py "${TRAIN_ARGS[@]}"
fi

# Step 7: Evaluate on TRAIN_GPU (model inference)
echo "[7/8] Evaluating on GPU ${TRAIN_GPU}..."
CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 evaluation/evaluate.py \
    --adapter_id "${RUN_NAME}" \
    --dataset "${DATASET}" \
    --base "${STUDENT_BASE}"

if [ -n "${OPENAI_API_KEY:-}" ]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 evaluation/evaluate.py \
        --adapter_id "${RUN_NAME}" \
        --dataset "${DATASET}" \
        --base "${STUDENT_BASE}" \
        --output_filename output_rewritten_rag.csv \
        --openai_rag True
else
    echo "Skipping RAG evaluation (OPENAI_API_KEY not set)."
fi

# Step 8: Grade (uses vLLM server on VLLM_GPU)
echo "[8/8] Grading..."
python3 evaluation/grade_answers_llm.py \
    --input_path "./outputs/squadshifts_${DATASET}/**/output*" \
    --vllm_hostname "${VLLM_HOST}"

python3 evaluation/grade_answers_match.py \
    --input_path "./outputs/squadshifts_${DATASET}/**/output*"

echo "=== Pipeline complete ==="
