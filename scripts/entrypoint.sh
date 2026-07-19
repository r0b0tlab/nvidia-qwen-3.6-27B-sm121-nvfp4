#!/usr/bin/env bash
# Qwen3.6-27B NVFP4 launch contract for GB10 / SM121.
set -euo pipefail

/usr/local/bin/audit_runtime.py

# Preserve Docker/sparkrun command semantics. In particular, sparkrun passes
# `bash -c <serve command>` after the image entrypoint.
if (( $# > 0 )); then
    exec "$@"
fi

MODEL_PATH="${MODEL_PATH:-/models/nvidia-Qwen3.6-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/Qwen3.6-27B-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"
MTP_TOKENS="${MTP_TOKENS:-2}"

if [[ "$KV_CACHE_DTYPE" == "nvfp4" ]]; then
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
    export VLLM_KV_CACHE_LAYOUT="${VLLM_KV_CACHE_LAYOUT:-HND}"
fi

args=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --enable-prefix-caching
    --language-model-only
)

if [[ "$MTP_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    args+=(
        --speculative-config
        "{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
    )
fi

printf 'R0B0TLAB_LAUNCH='
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
