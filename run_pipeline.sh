#!/usr/bin/env bash
# Prompt Distillation pipeline for financial tool-calling + NLP
# Manages vLLM server lifecycle: starts on VLLM_GPU, trains on TRAIN_GPU.
set -euo pipefail

VLLM_GPU="${VLLM_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VLLM_HOST="${VLLM_HOST:-localhost}"
STUDENT_BASE="${STUDENT_BASE:-llama3-8b-instruct}"
TEACHER_BASE="${TEACHER_BASE:-${STUDENT_BASE}}"
DATASET_FAMILY="${DATASET_FAMILY:-financial}"
DATASET="${DATASET:-tool_calling}"
MAX_ITEMS="${MAX_ITEMS:-20}"
MAX_TRAIN_ITEMS="${MAX_TRAIN_ITEMS:-200}"
TOOL_BATCHES="${TOOL_BATCHES:-10}"
NLP_BATCHES="${NLP_BATCHES:-5}"
RUN_NAME="${RUN_NAME:-pd_fin_llm}"

# Resolve HuggingFace model ID and family from base config name
resolve_model() {
    python3 -c "from core.model_configs import MODEL_CONFIGS; print(MODEL_CONFIGS['$1'].vllm_model)"
}
resolve_family() {
    python3 -c "from core.llm import get_model_family; from core.model_configs import MODEL_CONFIGS; print(get_model_family(MODEL_CONFIGS['$1'].vllm_model))"
}
TEACHER_MODEL=$(resolve_model "${TEACHER_BASE}")
STUDENT_MODEL=$(resolve_model "${STUDENT_BASE}")
TEACHER_FAMILY=$(resolve_family "${TEACHER_BASE}")
SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-context/financial_system_prompt_${TEACHER_FAMILY}.txt}"

echo "=== Financial Prompt Distillation Pipeline ==="
echo "  Teacher:          ${TEACHER_BASE} (${TEACHER_MODEL})"
echo "  Student:          ${STUDENT_BASE} (${STUDENT_MODEL})"
echo "  Dataset:          ${DATASET_FAMILY}/${DATASET}"
echo "  System prompt:    ${SYSTEM_PROMPT_PATH}"
echo "  vLLM GPU:         ${VLLM_GPU}"
echo "  Train GPU:        ${TRAIN_GPU}"
echo "  Run name:         ${RUN_NAME}"
echo "  Max items (eval): ${MAX_ITEMS}"
echo "  Max items (train):${MAX_TRAIN_ITEMS}"
echo "  Tool batches:     ${TOOL_BATCHES}"
echo "  NLP batches:      ${NLP_BATCHES}"
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
echo "[1/6] Generating questions..."
python3 evaluation/sample_tool_questions.py \
    --system_prompt_path "${SYSTEM_PROMPT_PATH}" \
    --vllm_hostname "${VLLM_HOST}" \
    --base "${TEACHER_BASE}" \
    --dataset_family "${DATASET_FAMILY}" \
    --dataset "${DATASET}" \
    --max_items "${MAX_TRAIN_ITEMS}" \
    --train_questions "${MAX_TRAIN_ITEMS}" \
    --tool_batches "${TOOL_BATCHES}" \
    --nlp_batches "${NLP_BATCHES}"

# Step 2: Convert generated questions into lesson format
echo "[2/6] Converting questions to lesson format..."
python3 curriculum/csv_to_lesson.py \
    --dataset_family "${DATASET_FAMILY}" \
    --dataset "${DATASET}" \
    --model "${TEACHER_BASE}" \
    --max_items "${MAX_TRAIN_ITEMS}" \
    --train_questions "${MAX_TRAIN_ITEMS}"

# Step 3: Generate exam from eval questions
echo "[3/6] Generating exam..."
python3 curriculum/questions_to_exam.py \
    --dataset_family "${DATASET_FAMILY}" \
    --dataset "${DATASET}" \
    --max_items "${MAX_ITEMS}" \
    --base "${TEACHER_BASE}" \
    --train_questions "${MAX_TRAIN_ITEMS}" \
    --max_train_items "${MAX_TRAIN_ITEMS}"

# Step 4: Generate teacher answers (lesson)
echo "[4/6] Generating teacher answers (lesson)..."
python3 curriculum/generate_teacher_answers.py \
    --generate_lesson True \
    --base "${TEACHER_BASE}" \
    --student_base "${STUDENT_BASE}" \
    --dataset_family "${DATASET_FAMILY}" \
    --dataset "${DATASET}" \
    --question_model "${TEACHER_BASE}" \
    --train_questions "${MAX_TRAIN_ITEMS}" \
    --max_items "${MAX_TRAIN_ITEMS}" \
    --lesson_temp 0.25 \
    --max_total_tokens 4096 \
    --vllm_hostname "${VLLM_HOST}"

# Step 5: Generate teacher answers (exam)
echo "[5/6] Generating teacher answers (exam)..."
python3 curriculum/generate_teacher_answers.py \
    --generate_exam True \
    --base "${TEACHER_BASE}" \
    --student_base "${STUDENT_BASE}" \
    --dataset_family "${DATASET_FAMILY}" \
    --dataset "${DATASET}" \
    --max_items "${MAX_ITEMS}" \
    --max_total_tokens 4096 \
    --vllm_hostname "${VLLM_HOST}"

# Step 6: Run training on TRAIN_GPU
echo "[6/6] Training on GPU ${TRAIN_GPU}..."
CHECKPOINT_DIR="output/checkpoints/${RUN_NAME}"
if [ -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ]; then
    echo "${CHECKPOINT_DIR}/adapter_model.safetensors already exists — skipping."
else
    TRAIN_ARGS=(
        --run_name "${RUN_NAME}"
        --use_wandb False
        --base "${STUDENT_BASE}"
        --dataset_family "${DATASET_FAMILY}"
        --dataset "${DATASET}"
        --lesson_model "${TEACHER_BASE}"
        --exam_model "${TEACHER_BASE}"
        --question_model "${TEACHER_BASE}"
        --train_questions "${MAX_TRAIN_ITEMS}"
        --max_items_train "${MAX_TRAIN_ITEMS}"
        --max_items_test "${MAX_ITEMS}"
        --token_loss_weight 1.0
        --logit_loss_weight 0.0
        --lesson_temp 0.25
    )
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 training/run_train_student.py "${TRAIN_ARGS[@]}"
fi

echo "=== Pipeline complete ==="
echo "Trained adapter: output/checkpoints/${RUN_NAME}"
