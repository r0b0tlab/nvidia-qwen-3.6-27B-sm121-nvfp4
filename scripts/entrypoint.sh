#!/usr/bin/env bash
set -euo pipefail

# SM121 vLLM v0.24.0 entrypoint for NVFP4 serving
# Usage:
#   docker run -v /path/to/model:/models/model:ro -p 8000:8000 <image>
#   Override with env vars: SERVED_MODEL_NAME, MAX_MODEL_LEN, SPECULATIVE_CONFIG, etc.

if [[ "${1:-}" == "audit" ]]; then
  exec /usr/bin/python3 /usr/local/bin/audit_runtime.py
fi

MODEL_PATH="${MODEL_ID:-/models/model}"
if [[ ! -e "$MODEL_PATH" ]]; then
  echo "ERROR: Model path does not exist inside container: $MODEL_PATH" >&2
  echo "Mount the model with: -v /path/to/model:/models/model:ro" >&2
  echo "Or set MODEL_ID to an HF repo id and ensure HF_TOKEN is set" >&2
  exit 2
fi

# Run the SM121 native runtime audit before serving
/usr/bin/python3 /usr/local/bin/audit_runtime.py

# Build speculative config args if provided
SPEC_ARGS=()
if [[ -n "${SPECULATIVE_CONFIG:-}" ]]; then
  SPEC_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

# v0.24.0 no longer recognizes these env vars; they produce "Unknown vLLM
# environment variable" warnings at startup. Backend selection is handled
# by --linear-backend / --moe-backend CLI flags (default: auto).
unset VLLM_NVFP4_GEMM_BACKEND VLLM_USE_FLASHINFER_MOE_FP4 2>/dev/null || true
unset VLLM_TEST_FORCE_FP8_MARLIN VLLM_MOE_FORCE_MARLIN 2>/dev/null || true

# Build quantization args: if QUANTIZATION is unset, let vLLM auto-detect
# from the checkpoint config (blog guidance). If set, use explicit value.
QUANT_ARGS=()
if [[ -n "${QUANTIZATION:-}" ]]; then
  QUANT_ARGS+=(--quantization "$QUANTIZATION")
fi

# Warmup ping: fire a small request to trigger JIT codegen before the first
# real user request arrives. See vLLM DGX Spark blog guidance.
# Must run BEFORE exec replaces the process.
if [[ -n "${WARMUP_ENABLED:-1}" ]]; then
  (
    for i in $(seq 1 60); do
      if curl -sf "http://127.0.0.1:${PORT:-8000}/v1/models" >/dev/null 2>&1; then
        break
      fi
      sleep 2
    done
    curl -sf -X POST "http://127.0.0.1:${PORT:-8000}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"'"${SERVED_MODEL_NAME:-model}"'","messages":[{"role":"user","content":"ping"}],"max_tokens":3,"temperature":0}' \
      >/dev/null 2>&1
  ) &
fi

exec /usr/bin/python3 -m vllm.entrypoints.openai.api_server \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --model "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-model}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
  --attention-backend "${ATTENTION_BACKEND:-flashinfer}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.72}" \
  --max-model-len "${MAX_MODEL_LEN:-8192}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --trust-remote-code \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${QUANT_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  ${EXTRA_ARGS:-}
