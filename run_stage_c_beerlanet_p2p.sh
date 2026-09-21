#!/usr/bin/env bash
set -euo pipefail

GPU_FROZEN="${GPU_FROZEN:-6}"
GPU_JOINT="${GPU_JOINT:-7}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-5}"
ROOT="${ROOT:-experiments/beerlanet_p2p_stage_c_$(date +%Y%m%d_%H%M%S)}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/data/yh_1/p2p-yfh/p2p-src-zzh-2025/pth/0318/best_model.pth}"
DATA_ROOT="${DATA_ROOT:-/home/data/yh_1/SET_2/p2p-src-zzh-2025/datasets}"
FFPE_NAME=$'\347\237\263\350\234\241-2025'
FROZEN_NAME=$'\345\206\260\345\206\273-2025'
FFPE_DATASET="${FFPE_DATASET:-${DATA_ROOT}/${FFPE_NAME}}"
FROZEN_DATASET="${FROZEN_DATASET:-${DATA_ROOT}/${FROZEN_NAME}}"

if [[ "${GPU_FROZEN}" == "${GPU_JOINT}" ]]; then
  echo "[ERROR] GPU_FROZEN and GPU_JOINT must be different for parallel training" >&2
  exit 2
fi

mkdir -p "${ROOT}"
printf '%s\n' "${ROOT}" > experiments/latest_beerlanet_p2p_stage_c.txt
{
  echo "root=${ROOT}"
  echo "gpu_frozen=${GPU_FROZEN}"
  echo "gpu_joint=${GPU_JOINT}"
  echo "batch_size=${BATCH_SIZE}"
  echo "seed=${SEED}"
  echo "epochs=${EPOCHS}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "ffpe_dataset=${FFPE_DATASET}"
  echo "frozen_dataset=${FROZEN_DATASET}"
  echo "source_selection_split=test (dataset has no val split)"
  echo "frozen_role=diagnostic_only"
} > "${ROOT}/protocol.txt"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "${ROOT}/gpu_before.txt"

echo "[Stage C 0/4] implementation tests"
PYTHONDONTWRITEBYTECODE=1 python tests/test_beerlanet_p2p_integration.py
python -m py_compile \
  models/beerlanet_p2p.py \
  train_beerlanet_p2p.py \
  evaluate_beerlanet_p2p.py \
  tests/smoke_beerlanet_p2p_forward.py \
  summarize_stage_c.py

common=(
  --init_checkpoint "${INIT_CHECKPOINT}"
  --dataset "${FFPE_DATASET}"
  --mean_std_path "${FFPE_DATASET}/mean_std.npy"
  --num_classes 1
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --start_eval -1
  --seed "${SEED}"
  --lr 4e-5
  --min_lr 1e-6
  --warmup_epochs 0
  --weight_decay 1e-4
  --reg_loss_coef 0.002
  --cls_loss_coef 1.0
  --eos_coef 0.5
  --set_cost_point 0.1
  --set_cost_class 1.0
  --match_dis 15
  --num_workers 0
  --eval_split test
  --beerlanet_r 8
  --beerlanet_n_iter 10
)

echo "[Stage C 1/4] real-image forward smoke"
for mode in frozen joint; do
  smoke_gpu="${GPU_FROZEN}"
  extra=()
  if [[ "${mode}" == "joint" ]]; then
    smoke_gpu="${GPU_JOINT}"
    extra+=(--beerlanet_gradient_checkpointing)
  fi
  CUDA_VISIBLE_DEVICES="${smoke_gpu}" python tests/smoke_beerlanet_p2p_forward.py \
    "${common[@]}" \
    --beerlanet_mode "${mode}" \
    "${extra[@]}" \
    --output_dir "${ROOT}/smoke_${mode}" \
    --smoke_output "${ROOT}/smoke_${mode}.json" \
    > "${ROOT}/smoke_${mode}.log" 2>&1
done

run_train() {
  local mode="$1"
  local gpu="$2"
  local output_dir="${ROOT}/beerlanet_${mode}"
  local extra=()
  if [[ "${mode}" == "joint" ]]; then
    extra+=(--beerlanet_gradient_checkpointing)
  fi
  mkdir -p "${output_dir}"
  echo "[TRAIN] mode=${mode} physical_gpu=${gpu} output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" python train_beerlanet_p2p.py \
    "${common[@]}" \
    --beerlanet_mode "${mode}" \
    "${extra[@]}" \
    --output_dir "${output_dir}" \
    > "${output_dir}/console.log" 2>&1
}

echo "[Stage C 2/4] parallel 5-epoch source-only training"
run_train frozen "${GPU_FROZEN}" &
pid_frozen=$!
run_train joint "${GPU_JOINT}" &
pid_joint=$!
echo "[TRAIN] frozen_pid=${pid_frozen} joint_pid=${pid_joint}"
status=0
wait "${pid_frozen}" || status=1
wait "${pid_joint}" || status=1
if [[ "${status}" != "0" ]]; then
  echo "[ERROR] at least one Stage C training process failed" >&2
  exit 1
fi

run_eval_pair() {
  local mode="$1"
  local gpu="$2"
  local experiment="${ROOT}/beerlanet_${mode}"
  local checkpoint="${experiment}/best_model.pth"
  local extra=()
  if [[ "${mode}" == "joint" ]]; then
    extra+=(--beerlanet_gradient_checkpointing)
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python evaluate_beerlanet_p2p.py \
    "${common[@]}" \
    --beerlanet_mode "${mode}" \
    "${extra[@]}" \
    --checkpoint "${checkpoint}" \
    --case_name "ffpe_${mode}_source_selection" \
    --output_dir "${experiment}/eval_ffpe_source_selection" \
    > "${experiment}/eval_ffpe_source_selection.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" python evaluate_beerlanet_p2p.py \
    "${common[@]}" \
    --dataset "${FROZEN_DATASET}" \
    --mean_std_path "${FROZEN_DATASET}/mean_std.npy" \
    --beerlanet_mode "${mode}" \
    "${extra[@]}" \
    --checkpoint "${checkpoint}" \
    --case_name "frozen_${mode}_diagnostic_only" \
    --output_dir "${experiment}/eval_frozen_diagnostic" \
    > "${experiment}/eval_frozen_diagnostic.log" 2>&1
}

echo "[Stage C 3/4] FFPE selected-checkpoint evaluation and Frozen diagnostics"
run_eval_pair frozen "${GPU_FROZEN}" &
eval_pid_frozen=$!
run_eval_pair joint "${GPU_JOINT}" &
eval_pid_joint=$!
status=0
wait "${eval_pid_frozen}" || status=1
wait "${eval_pid_joint}" || status=1
if [[ "${status}" != "0" ]]; then
  echo "[ERROR] at least one Stage C evaluation process failed" >&2
  exit 1
fi

echo "[Stage C 4/4] summary"
python summarize_stage_c.py "${ROOT}" | tee "${ROOT}/summary.log"
echo "[SUCCESS] ${ROOT}"
