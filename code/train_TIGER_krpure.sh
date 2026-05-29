PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL_SIZE="${MODEL_SIZE:-mini}"   # [mini,medium,large]
ROOT_PATH="${PROJECT_ROOT}"
DATA_PATH="${ROOT_PATH}/dataset/kuairand/kuairand-Pure/data"
SID_PATH="${ROOT_PATH}/code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv"
LOG_PATH="${LOG_PATH:-${DATA_PATH}/log_session_4_08_to_5_08_Pure.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"
INFER_SIZE="${INFER_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BEAM_SIZE="${BEAM_SIZE:-30}"
TOPK_LIST="${TOPK_LIST:-5 10 20}"
SAVE_PATH="${SAVE_PATH:-${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}.pth}"
TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${ROOT_PATH}/output/KuaiRand_Pure/env/log/tiger_sid_krpure.log}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  NUM_EPOCHS=1
  BATCH_SIZE=128
  INFER_SIZE=256
  BEAM_SIZE=2
  TOPK_LIST="5"
  SAVE_PATH="${ROOT_PATH}/output/KuaiRand_Pure/env/tiger_sid_krpure_${MODEL_SIZE}_smoke.pth"
  TRAIN_LOG_PATH="${ROOT_PATH}/output/KuaiRand_Pure/env/log/tiger_sid_krpure_smoke.log"
fi

python "${ROOT_PATH}/code/train_TIGER_krpure.py" \
  --log_paths "${LOG_PATH}" \
  --sid_mapping_path "${SID_PATH}" \
  --max_hist_items 50 \
  --batch_size "${BATCH_SIZE}" \
  --infer_size "${INFER_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --model_size ${MODEL_SIZE} \
  --lr 1e-4 \
  --beam_size "${BEAM_SIZE}" \
  --topk_list ${TOPK_LIST} \
  --save_path "${SAVE_PATH}" \
  --log_path  "${TRAIN_LOG_PATH}"
