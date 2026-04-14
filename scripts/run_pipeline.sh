#!/usr/bin/env bash
# Prompt Distillation pipeline for financial tool-calling + NLP
# Manages vLLM server lifecycle: starts on VLLM_GPU, trains on TRAIN_GPU.
set -euo pipefail

# Ensure project root is on Python path
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}"
cd "${SCRIPT_DIR}"

# Load environment variables
if [ -f .env ]; then
    set -a; source .env; set +a
fi

CONFIG="${1:-config/pipeline.yaml}"

# Parse YAML config into shell variables
eval "$(python3 -c "
import yaml, sys
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
p = c.get('project', {})
m = c.get('models', {})
d = c.get('dataset', {})
g = c.get('gpu', {})
q = c.get('questions', {})
ta = c.get('teacher_answers', {})
t = c.get('training', {})
# Project
print(f'RUN_NAME=\"{p.get(\"run_name\", \"pd_fin_llm\")}\"')
_gn = p.get('group_name')
print(f'GROUP_NAME=\"{_gn if _gn is not None else \"\"}\"')
print(f'TRAIN_ONLY=\"{p.get(\"train_only\", False)}\"')
print(f'USE_WANDB=\"{p.get(\"use_wandb\", False)}\"')
print(f'SYSTEM_PROMPT_PATH=\"{p.get(\"system_prompt_path\", \"\")}\"')
print(f'TOOLS_SCHEMA_PATH=\"{p.get(\"tools_schema_path\", \"\")}\"')
# Models
print(f'TEACHER_BASE=\"{m.get(\"teacher\", \"llama3-8b-instruct\")}\"')
print(f'STUDENT_BASE=\"{m.get(\"student\", \"llama3-8b-instruct\")}\"')
# Dataset
print(f'DATASET_FAMILY=\"{d.get(\"family\", \"financial\")}\"')
print(f'DATASET=\"{d.get(\"name\", \"tool_calling\")}\"')
print(f'VARIANT=\"{d.get(\"variant\", \"default\")}\"')
print(f'DISTRACTOR_DATASET=\"{d.get(\"distractor_dataset\", \"\")}\"')
print(f'DATAPATH=\"{d.get(\"datapath\", \"output/teacher_answers\")}\"')
print(f'CUSTOM_TRAIN_DATA=\"{d.get(\"custom_train_data\", \"\")}\"')
print(f'CUSTOM_VAL_DATA=\"{d.get(\"custom_val_data\", \"\")}\"')
_pi = d.get('partition_idx')
print(f'PARTITION_IDX=\"{_pi if _pi is not None else \"\"}\"')
_pt = d.get('partition_type')
print(f'PARTITION_TYPE=\"{_pt if _pt is not None else \"\"}\"')
# GPU
print(f'VLLM_GPU=\"{g.get(\"vllm\", \"0\")}\"')
print(f'TRAIN_GPU=\"{g.get(\"train\", \"0\")}\"')
print(f'VLLM_HOST=\"{g.get(\"vllm_host\", \"localhost\")}\"')
# Questions
print(f'TOOL_BATCHES=\"{q.get(\"tool_batches\", 10)}\"')
print(f'NLP_BATCHES=\"{q.get(\"nlp_batches\", 5)}\"')
print(f'EVAL_RATIO=\"{q.get(\"eval_ratio\", 0.2)}\"')
print(f'MAX_TRAIN_ITEMS=\"{q.get(\"max_train_items\", 200)}\"')
print(f'MAX_ITEMS=\"{q.get(\"max_eval_items\", 20)}\"')
print(f'QUESTION_TEMPERATURE=\"{q.get(\"question_temperature\", 1.5)}\"')
print(f'LESSON_NUM_CHOICES=\"{q.get(\"lesson_num_choices\", 1)}\"')
print(f'EXAM_NUM_CHOICES=\"{q.get(\"exam_num_choices\", 1)}\"')
# Teacher answers
print(f'LESSON_TEMP=\"{ta.get(\"lesson_temp\", 0.25)}\"')
print(f'EXAM_TEMP=\"{ta.get(\"exam_temp\", 0.25)}\"')
print(f'MAX_TOTAL_TOKENS=\"{ta.get(\"max_total_tokens\", 4096)}\"')
print(f'MAX_NEW_TOKENS=\"{ta.get(\"max_new_tokens\", 500)}\"')
# Training
print(f'LEARNING_RATE=\"{t.get(\"learning_rate\", 1e-5)}\"')
print(f'BATCH_SIZE=\"{t.get(\"batch_size\", 4)}\"')
print(f'MICRO_BATCH_SIZE=\"{t.get(\"micro_batch_size\", 4)}\"')
print(f'N_EPOCHS=\"{t.get(\"n_epochs\", 2)}\"')
print(f'LORA_R=\"{t.get(\"lora_r\", 1024)}\"')
print(f'MIXED_PRECISION=\"{t.get(\"mixed_precision\", \"bf16\")}\"')
print(f'CLOSED_BOOK=\"{t.get(\"closed_book\", True)}\"')
print(f'TOKEN_LOSS_WEIGHT=\"{t.get(\"token_loss_weight\", 1.0)}\"')
print(f'LOGIT_LOSS_WEIGHT=\"{t.get(\"logit_loss_weight\", 0.0)}\"')
print(f'TRAIN_TEMPERATURE=\"{t.get(\"train_temperature\", 2.0)}\"')
print(f'REVERSE_KL=\"{t.get(\"reverse_kl\", False)}\"')
print(f'WARMUP_RATIO=\"{t.get(\"warmup_ratio\", 0.1)}\"')
_ws = t.get('warmup_steps')
print(f'WARMUP_STEPS=\"{_ws if _ws is not None else \"\"}\"')
print(f'WEIGHT_DECAY=\"{t.get(\"weight_decay\", 0.1)}\"')
print(f'MAX_GRAD_NORM=\"{t.get(\"max_grad_norm\", 1.0)}\"')
print(f'DECAY=\"{t.get(\"decay\", True)}\"')
print(f'SEED=\"{t.get(\"seed\", 0)}\"')
print(f'EVAL_INTERVAL=\"{t.get(\"eval_interval\", -1)}\"')
print(f'LOG_INTERVAL=\"{t.get(\"log_interval\", 1)}\"')
print(f'GENERATION_INTERVAL=\"{t.get(\"generation_interval\", 0)}\"')
print(f'VALIDATE=\"{t.get(\"validate\", True)}\"')
print(f'SAVE=\"{t.get(\"save\", True)}\"')
print(f'SAVE_DURING_TRAINING=\"{t.get(\"save_during_training\", False)}\"')
print(f'CHECKPOINT_INTERVAL=\"{t.get(\"checkpoint_interval\", 0)}\"')
print(f'CHECKPOINT_INTERVAL_SECONDS=\"{t.get(\"checkpoint_interval_seconds\", 0)}\"')
print(f'MAX_LENGTH=\"{t.get(\"max_length\", 0)}\"')
print(f'MAX_TOTAL_LENGTH=\"{t.get(\"max_total_length\", 0)}\"')
print(f'DEEPSPEED_PATH=\"{t.get(\"deepspeed_path\", \"\")}\"')
print(f'DEEPSPEED_PATH_TEACHER=\"{t.get(\"deepspeed_path_teacher\", \"\")}\"')
import json as _json
_topics = q.get('topics', [])
if _topics:
    print('TOPICS=' + repr(_json.dumps(_topics)))
else:
    print('TOPICS=\"\"')
")"

# Resolve HuggingFace model ID and family from base config name
resolve_model() {
    python3 -c "from models.configs import MODEL_CONFIGS; print(MODEL_CONFIGS['$1'].vllm_model)"
}
resolve_family() {
    python3 -c "from models.llm import get_model_family; from models.configs import MODEL_CONFIGS; print(get_model_family(MODEL_CONFIGS['$1'].vllm_model))"
}
TEACHER_MODEL=$(resolve_model "${TEACHER_BASE}")
STUDENT_MODEL=$(resolve_model "${STUDENT_BASE}")
TEACHER_FAMILY=$(resolve_family "${TEACHER_BASE}")
if [ -z "${SYSTEM_PROMPT_PATH}" ]; then
    SYSTEM_PROMPT_PATH="context/financial_system_prompt_${TEACHER_FAMILY}.txt"
fi

echo "=== Financial Prompt Distillation Pipeline ==="
echo "  Teacher:          ${TEACHER_BASE} (${TEACHER_MODEL})"
echo "  Student:          ${STUDENT_BASE} (${STUDENT_MODEL})"
echo "  Dataset:          ${DATASET_FAMILY}/${DATASET}"
echo "  System prompt:    ${SYSTEM_PROMPT_PATH}"
echo "  Tools schema:     ${TOOLS_SCHEMA_PATH}"
echo "  vLLM GPU:         ${VLLM_GPU}"
echo "  Train GPU:        ${TRAIN_GPU}"
echo "  Run name:         ${RUN_NAME}"
echo "  Max items (eval): ${MAX_ITEMS}"
echo "  Max items (train):${MAX_TRAIN_ITEMS}"
echo "  Tool batches:     ${TOOL_BATCHES}"
echo "  NLP batches:      ${NLP_BATCHES}"
echo ""

if [ "${TRAIN_ONLY}" = "True" ]; then
    echo "=== Skipping data generation, training only ==="
else

# --- Start vLLM server on VLLM_GPU ---
# Compute tensor parallel size from number of vLLM GPUs
VLLM_TP=$(echo "${VLLM_GPU}" | tr ',' '\n' | wc -l)

echo "Starting vLLM server on GPU ${VLLM_GPU} (TP=${VLLM_TP})..."
CUDA_VISIBLE_DEVICES="${VLLM_GPU}" python3 -u -m vllm.entrypoints.openai.api_server \
    --model "${TEACHER_MODEL}" --dtype auto --api-key token-abc123 \
    --tensor-parallel-size "${VLLM_TP}" &
VLLM_PID=$!

# Wait for server to be ready (test actual model availability, not just health endpoint)
echo "Waiting for vLLM server..."
for i in $(seq 1 180); do
    if curl -s "http://${VLLM_HOST}:8000/v1/models" -H "Authorization: Bearer token-abc123" 2>/dev/null | grep -q "${TEACHER_MODEL}"; then
        echo "vLLM server ready (model loaded)."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server process died."
        exit 1
    fi
    sleep 2
done
if ! curl -s "http://${VLLM_HOST}:8000/v1/models" -H "Authorization: Bearer token-abc123" 2>/dev/null | grep -q "${TEACHER_MODEL}"; then
    echo "ERROR: vLLM server failed to load model within timeout."
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
QUESTION_ARGS=(
    --system_prompt_path "${SYSTEM_PROMPT_PATH}"
    --vllm_hostname "${VLLM_HOST}"
    --base "${TEACHER_BASE}"
    --dataset_family "${DATASET_FAMILY}"
    --dataset "${DATASET}"
    --max_items "${MAX_TRAIN_ITEMS}"
    --train_questions "${MAX_TRAIN_ITEMS}"
    --tool_batches "${TOOL_BATCHES}"
    --nlp_batches "${NLP_BATCHES}"
)
if [ -n "${TOPICS}" ]; then
    QUESTION_ARGS+=(--topics "${TOPICS}")
fi
python3 curriculum/sample_tool_questions.py "${QUESTION_ARGS[@]}"

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
TEACHER_ANSWER_ARGS=(
    --generate_lesson True
    --base "${TEACHER_BASE}"
    --student_base "${STUDENT_BASE}"
    --dataset_family "${DATASET_FAMILY}"
    --dataset "${DATASET}"
    --question_model "${TEACHER_BASE}"
    --train_questions "${MAX_TRAIN_ITEMS}"
    --max_items "${MAX_TRAIN_ITEMS}"
    --lesson_temp 0.25
    --max_total_tokens 4096
    --vllm_hostname "${VLLM_HOST}"
)
if [ -n "${TOOLS_SCHEMA_PATH}" ]; then
    TEACHER_ANSWER_ARGS+=(--tools_schema_path "${TOOLS_SCHEMA_PATH}")
fi
python3 curriculum/generate_teacher_answers.py "${TEACHER_ANSWER_ARGS[@]}"

# Step 5: Generate teacher answers (exam)
echo "[5/6] Generating teacher answers (exam)..."
EXAM_ANSWER_ARGS=(
    --generate_exam True
    --base "${TEACHER_BASE}"
    --student_base "${STUDENT_BASE}"
    --dataset_family "${DATASET_FAMILY}"
    --dataset "${DATASET}"
    --max_items "${MAX_ITEMS}"
    --max_total_tokens 4096
    --vllm_hostname "${VLLM_HOST}"
)
if [ -n "${TOOLS_SCHEMA_PATH}" ]; then
    EXAM_ANSWER_ARGS+=(--tools_schema_path "${TOOLS_SCHEMA_PATH}")
fi
python3 curriculum/generate_teacher_answers.py "${EXAM_ANSWER_ARGS[@]}"

# Stop vLLM server to free GPU for teacher during training
echo "Stopping vLLM server to free GPU ${VLLM_GPU}..."
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true
trap - EXIT  # disable cleanup trap since we already stopped it
sleep 5

fi  # end of data generation block

# Step 6: Run training
echo "[6/6] Training on GPUs ${TRAIN_GPU}..."
CHECKPOINT_DIR="output/checkpoints/${RUN_NAME}"
if [ -f "${CHECKPOINT_DIR}/adapter_model.safetensors" ]; then
    echo "${CHECKPOINT_DIR}/adapter_model.safetensors already exists — skipping."
else
    TRAIN_ARGS=(
        --run_name "${RUN_NAME}"
        --use_wandb "${USE_WANDB}"
        --base "${STUDENT_BASE}"
        --teacher "${TEACHER_BASE}"
        --dataset_family "${DATASET_FAMILY}"
        --dataset "${DATASET}"
        --variant "${VARIANT}"
        --lesson_model "${TEACHER_BASE}"
        --exam_model "${TEACHER_BASE}"
        --question_model "${TEACHER_BASE}"
        --train_questions "${MAX_TRAIN_ITEMS}"
        --max_items_train "${MAX_TRAIN_ITEMS}"
        --max_items_test "${MAX_ITEMS}"
        --learning_rate "${LEARNING_RATE}"
        --batch_size "${BATCH_SIZE}"
        --micro_batch_size "${MICRO_BATCH_SIZE}"
        --n_epochs "${N_EPOCHS}"
        --lora_r "${LORA_R}"
        --mixed_precision "${MIXED_PRECISION}"
        --closed_book "${CLOSED_BOOK}"
        --token_loss_weight "${TOKEN_LOSS_WEIGHT}"
        --logit_loss_weight "${LOGIT_LOSS_WEIGHT}"
        --train_temperature "${TRAIN_TEMPERATURE}"
        --reverse_kl "${REVERSE_KL}"
        --warmup_ratio "${WARMUP_RATIO}"
        --weight_decay "${WEIGHT_DECAY}"
        --max_grad_norm "${MAX_GRAD_NORM}"
        --decay "${DECAY}"
        --seed "${SEED}"
        --eval_interval "${EVAL_INTERVAL}"
        --log_interval "${LOG_INTERVAL}"
        --generation_interval "${GENERATION_INTERVAL}"
        --validate "${VALIDATE}"
        --save "${SAVE}"
        --save_during_training "${SAVE_DURING_TRAINING}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --checkpoint_interval_seconds "${CHECKPOINT_INTERVAL_SECONDS}"
        --max_length "${MAX_LENGTH}"
        --max_total_length "${MAX_TOTAL_LENGTH}"
        --datapath "${DATAPATH}"
        --lesson_temp "${LESSON_TEMP}"
        --exam_temp "${EXAM_TEMP}"
        --question_temperature "${QUESTION_TEMPERATURE}"
        --lesson_num_choices "${LESSON_NUM_CHOICES}"
        --exam_num_choices "${EXAM_NUM_CHOICES}"
    )
    if [ -n "${GROUP_NAME}" ]; then
        TRAIN_ARGS+=(--group_name "${GROUP_NAME}")
    fi
    if [ -n "${CUSTOM_TRAIN_DATA}" ]; then
        TRAIN_ARGS+=(--custom_train_data "${CUSTOM_TRAIN_DATA}")
    fi
    if [ -n "${CUSTOM_VAL_DATA}" ]; then
        TRAIN_ARGS+=(--custom_val_data "${CUSTOM_VAL_DATA}")
    fi
    if [ -n "${WARMUP_STEPS}" ]; then
        TRAIN_ARGS+=(--warmup_steps "${WARMUP_STEPS}")
    fi
    if [ -n "${DISTRACTOR_DATASET}" ]; then
        TRAIN_ARGS+=(--distractor_dataset "${DISTRACTOR_DATASET}")
    fi
    if [ -n "${PARTITION_IDX}" ]; then
        TRAIN_ARGS+=(--partition_idx "${PARTITION_IDX}")
    fi
    if [ -n "${PARTITION_TYPE}" ]; then
        TRAIN_ARGS+=(--partition_type "${PARTITION_TYPE}")
    fi
    if [ -n "${DEEPSPEED_PATH}" ]; then
        TRAIN_ARGS+=(--deepspeed_path "${DEEPSPEED_PATH}")
    fi
    if [ -n "${DEEPSPEED_PATH_TEACHER}" ]; then
        TRAIN_ARGS+=(--deepspeed_path_teacher "${DEEPSPEED_PATH_TEACHER}")
    fi
    CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python3 training/train.py "${TRAIN_ARGS[@]}"
fi

echo "=== Pipeline complete ==="
echo "Trained adapter: output/checkpoints/${RUN_NAME}"
