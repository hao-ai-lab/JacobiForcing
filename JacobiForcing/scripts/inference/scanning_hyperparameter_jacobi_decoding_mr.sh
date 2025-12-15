#!/usr/bin/env bash
set -euo pipefail

PROFILE_PY="jacobi_forcing_inference_MR_humaneval_config_grid_search.py"

DF_FILE="/home/lah003/data/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet"
MODEL_NAME="/raid/lah003/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6/ckpt-344092"
TOKENIZER_NAME="/home/lah003/models/Qwen2.5-Coder-7B-Instruct"

CSV_DIR="profiling_results"
EVAL_DIR="/home/lah003/data/CLLM2_eval_generations/multiblock_testing_prompt/scanning_generation"

MAX_CALLS=64
MAX_NEW_TOKENS=1024

LOG_DIR="logs/hparam_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" "$CSV_DIR"

block_sizes=(256 128 64 32 16 8)
Ks=(1 2 3 4)
pool_sizes=(1 2 4 8 12)

# r sweep: from 0.50 to 0.95, in increments of 0.5
r_values=()
while read -r val; do r_values+=("$val"); done < <(seq 0.50 0.05 0.95)

# ---------------------------
# GPU semaphore
# ---------------------------
GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}

FIFO="$(mktemp -u)"
mkfifo "$FIFO"
exec 9<>"$FIFO"
rm "$FIFO"

# seed tokens
for g in "${GPUS[@]}"; do
  echo "$g" >&9
done

pids=()
fails=0

launch_one() {
  local n="$1" K="$2" r_fmt="$3" ng="$4"

  local run_id="ntok${n}_K${K}_r${r_fmt}_ng${ng}"
  local csv_path="${CSV_DIR}/${run_id}_diffusion_profile_humaneval.csv"
  local log_path="${LOG_DIR}/${run_id}.log"

  # acquire a GPU token
  local gpu
  read -r gpu <&9

  echo "========= LAUNCH $run_id on GPU $gpu ========="

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python3 "$PROFILE_PY" \
      "$DF_FILE" \
      "$MODEL_NAME" \
      "$TOKENIZER_NAME" \
      "$csv_path" \
      "$MAX_CALLS" \
      "$MAX_NEW_TOKENS" \
      --n_token_seq_len "$n" \
      --K "$K" \
      --r "$r_fmt" \
      --n_gram_pool_size "$ng" \
      --eval_dir "$EVAL_DIR" \
      --out_prefix "$run_id" \
      > "$log_path" 2>&1
    rc=$?

    if [[ $rc -ne 0 ]]; then
      echo "FAILED $run_id on GPU $gpu (see $log_path)" >&2
    fi

    # release GPU token
    echo "$gpu" >&9
    exit $rc
  ) &

  pids+=("$!")
}

# ---------------------------
# Sweep
# ---------------------------
for n in "${block_sizes[@]}"; do
  for K in "${Ks[@]}"; do
    for r in "${r_values[@]}"; do
      r_fmt=$(printf "%.2f" "$r")
      for ng in "${pool_sizes[@]}"; do
        launch_one "$n" "$K" "$r_fmt" "$ng"
      done
    done
  done
done

# wait all
for pid in "${pids[@]}"; do
  if wait "$pid"; then
    :
  else
    fails=$((fails + 1))
  fi
done

exec 9>&-

echo "Sweep complete. Logs in $LOG_DIR, CSVs in $CSV_DIR"
if ((fails > 0)); then
  echo "WARNING: $fails runs failed. Check logs."
  exit 1
fi
echo "All runs succeeded."
