#!/usr/bin/env bash
# Qwen3.6-27B NVFP4-weight launch contract for GB10 / SM121.
set -euo pipefail

AUDIT_BIN=/usr/local/bin/audit_runtime.py

if [[ "${1:-}" == "audit" ]]; then
    exec "$AUDIT_BIN"
fi

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
NVFP4_KV_ENABLED="${R0B0TLAB_NVFP4_KV_ENABLED:-0}"
joined_args=" $* "
if [[ "$NVFP4_KV_ENABLED" != "1" ]] && {
    [[ "$KV_CACHE_DTYPE" == "nvfp4" ]] \
        || [[ "$joined_args" == *" --kv-cache-dtype nvfp4 "* ]] \
        || [[ "$joined_args" == *" --kv-cache-dtype=nvfp4 "* ]];
}; then
    echo "R0B0TLAB_LAUNCH_REJECTED: NVFP4 KV is disabled in this production image; use fp8" >&2
    exit 64
fi

# Every executable path is admitted by the same fail-closed GPU/runtime audit.
# This includes explicit Docker/Kubernetes commands and sparkrun's `bash -c`.
"$AUDIT_BIN"

# Preserve Docker/sparkrun argv semantics after admission.
if (( $# > 0 )); then
    exec "$@"
fi

MODEL_PATH="${MODEL_PATH:-/models/nvidia-Qwen3.6-27B-NVFP4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/Qwen3.6-27B-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
MTP_TOKENS="${MTP_TOKENS:-2}"

args=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --attention-backend FLASHINFER
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --language-model-only
    --reasoning-parser qwen3
    --enable-auto-tool-choice
    --tool-call-parser qwen3_xml
)

if [[ "$MTP_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    args+=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}"
    )
fi

printf 'R0B0TLAB_LAUNCH='
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
