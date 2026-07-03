# NVIDIA Qwen3.6-27B NVFP4 — SM121 Native Serving

Optimized vLLM v0.24.0 runtime for **nvidia/Qwen3.6-27B-NVFP4** on NVIDIA GB10 / SM121 (DGX Spark),
with **FP8 KV cache**, **MTP speculative decoding**, and **native NVFP4 weight quantization**.

> **KV Cache Update (2026-07-03):** The runtime currently uses **FP8 KV cache** (`--kv-cache-dtype fp8`).
> NVFP4 KV cache support is under active development — a FlashInfer attention kernel patch is in
> progress to enable proper FP4 dequantization on SM120/121. Until then, FP8 KV is the recommended
> configuration for correct output quality.

## Highlights

| Metric | FP8 KV Baseline | Notes |
|---|---|---|
| **KV Cache Dtype** | fp8 | NVFP4 KV pending FlashInfer patch |
| **MTP** | ✅ (1 spec token) | 88–93% acceptance rate |
| **c1 decode** | 19.15 tok/s | |
| **GSM8K (0-shot, flexible-extract)** | **81.88%** | 1319 samples, full test set |

### Throughput (FP8 KV + MTP, 256-token generation, 8K context)

| Concurrency | Output tok/s | Power (W) | Efficiency (J/1K tok) | Temp (°C) |
|---:|---:|---:|---:|---:|
| 1 | 19.15 | 34.3 | 1,793 | 60.5 |
| 4 | 69.62 | 32.3 | 465 | 60.8 |
| 8 | 102.76 | 36.9 | 359 | 62.7 |
| 16 | 144.00 | 38.9 | 270 | 64.7 |
| 32 | 248.40 | 44.2 | 178 | 67.6 |

### GSM8K Accuracy (Full 1319-Sample Evaluation)

| Protocol | Score | Details |
|---|---|---|
| **0-shot, flexible-extract** | **81.88%** | lm-eval 0.4.12, completions endpoint, max_gen_toks=2048 |
| 8-shot, flexible-extract | 76.80% | Same protocol, 8-shot priming |

Evaluation: lm-evaluation-harness, `local-completions` model, temperature=0, FP8 KV cache,
1,162,353 KV cache tokens, max concurrency 141.89x at 8K context.

### Sanity Suite (5/5 passed)

| Test | Tokens | Latency |
|---|---|---|
| Math (17×23) | 32 | 1.72s |
| Code (reverse string) | 64 | 3.26s |
| Logical reasoning (syllogism) | 64 | 3.28s |
| Factual (capital of Australia) | 32 | 1.72s |
| Instruction-following (3 colors) | 32 | 1.71s |

---

## The Six Fixes

Building a working NVFP4 KV cache runtime on SM121 required solving six distinct issues. Each one
blocked the container from starting — this section documents what broke, why, and how it was fixed.

### Fix 1: OOM Killer During Build → `MAX_JOBS=6`

**Symptom:** Docker build killed at 94–101/371 CUDA objects. No error message — build process just died.

**Root Cause:** The Dockerfile set `MAX_JOBS=20`, spawning 20 parallel `nvcc` processes (~40 GB RAM spike).
The system has 121 GB RAM, but `hermes-gateway.service` has `oom_score_adj=200`, making it the OOM
killer's first target. `dmesg` confirmed: `oom-kill: task_memcg=.../hermes-gateway.service`.

**Fix:** Changed `ENV MAX_JOBS=20` → `ENV MAX_JOBS=6` + added `ENV NVCC_THREADS=2`. Peak RAM dropped
from ~40 GB to ~15 GB. Build completed all 371 objects in ~57 minutes.

```dockerfile
ENV MAX_JOBS=6
ENV NVCC_THREADS=2
```

### Fix 2: Missing Python Packages → Bulk Site-Packages COPY

**Symptom:** Container crashed at startup with `ModuleNotFoundError: No module named 'zmq'`, then
`urllib3`, then cascading failures.

**Root Cause:** The runtime stage tried to COPY individual packages (torch, vllm, flashinfer) from
the builder. But vLLM has deep dependency trees — each missing import revealed another missing dep.

**Fix:** Replaced all individual COPY lines with a single bulk copy of the entire site-packages:

```dockerfile
COPY --from=builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/
```

Brute force, but complete. Image stayed at ~13 GB (deduplicated layer).

### Fix 3: PTX Version Mismatch → CUDA 13.0 Toolkit (not 13.2)

**Symptom:** Container started, model weights began loading, then crashed at
`marlin_utils_fp4.py:264` in `prepare_fp4_layer_for_marlin` with:
`torch.AcceleratorError: CUDA error: the provided PTX was compiled with an unsupported toolchain`

**Root Cause:** The builder installed `cuda-toolkit-13-2` (nvcc 13.2), but PyTorch 2.11.0 ships with
CUDA 13.0 runtime. PTX compiled by nvcc 13.2 cannot execute on a CUDA 13.0 driver — the PTX ISA
version is higher than the runtime supports.

**Fix:** Changed `cuda-toolkit-13-2` → `cuda-toolkit-13-0`. This required a **full rebuild** from
scratch (the CUDA toolkit layer is early in the Dockerfile, invalidating all subsequent layers
including the 371-object compile).

### Fix 4: No C Compiler at Runtime → `build-essential`

**Symptom:** Model weights loaded successfully. Crash during FlashInfer JIT compilation:
`Failed to find C compiler`

**Root Cause:** FlashInfer uses JIT compilation at runtime to generate SM121-specific kernels. The
runtime image only installed `python3.12` — no `gcc` or `cc`. The builder stage had `build-essential`
but those weren't copied to runtime.

**Fix:** Added `build-essential` to the runtime stage's `apt-get install`.

### Fix 5: No Ninja Build Tool → `ninja-build`

**Symptom:** Weights loaded, JIT attempted, crash at KV cache initialization:
`RuntimeError: ninja: not found`

**Root Cause:** FlashInfer's JIT uses `ninja` as its build system. The runtime image didn't include it.

**Fix:** Added `ninja-build` to the runtime stage's `apt-get install`.

### Fix 6: Missing CUDA Dev Headers → `cuda-libraries-dev-13-0`

**Symptom:** Ninja found, compilation started, crash at FlashInfer kernel JIT:
`curand_kernel.h: No such file or directory`

**Root Cause:** FlashInfer's JIT kernels `#include` CUDA development headers (`curand_kernel.h`,
etc.) at runtime. The runtime image had CUDA runtime libraries but not dev headers.

**Fix:** Added `cuda-nvcc-13-0` + `cuda-libraries-dev-13-0` to the runtime stage:

```dockerfile
cuda-nvcc-13-0 cuda-libraries-dev-13-0
```

---

## Build & Run

### Build from Source

```bash
git clone https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4.git
cd nvidia-qwen-3.6-27B-sm121-nvfp4

# ~60 min with MAX_JOBS=6 on GB10
docker build -f docker/Dockerfile.kv-exp -t sm121-vllm-v0240-nvfp4:kv-exp .
```

### Serve with NVFP4 KV Cache + MTP

```bash
docker run -d --gpus all --ipc=host --name sm121-vllm \
  -p 18080:8000 \
  -v /path/to/nvidia-Qwen3.6-27B-NVFP4:/models/model:ro \
  -e SERVED_MODEL_NAME="Qwen3.6-27B-NVFP4" \
  -e MAX_MODEL_LEN=8192 \
  -e KV_CACHE_DTYPE=nvfp4 \
  -e SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}' \
  sm121-vllm-v0240-nvfp4:kv-exp
```

### Run the Benchmark

```bash
python3 scripts/benchmark_nvfp4_kv.py \
  --base-url http://127.0.0.1:18080 \
  --model Qwen3.6-27B-NVFP4 \
  --output results/benchmark.json
```

### Runtime Audit

```bash
docker run --rm --gpus all --entrypoint bash sm121-vllm-v0240-nvfp4:kv-exp audit
```

Checks: vLLM v0.24.x, SM121 capability, stable ABI extensions, Qwen3.5 model, modelopt_mixed, NVFP4.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SERVED_MODEL_NAME` | model | Model name for API |
| `MAX_MODEL_LEN` | 8192 | Maximum sequence length |
| `KV_CACHE_DTYPE` | fp8 | KV cache dtype — set to `nvfp4` for NVFP4 KV |
| `ATTENTION_BACKEND` | flashinfer | Attention backend |
| `GPU_MEMORY_UTILIZATION` | 0.72 | GPU memory fraction |
| `MAX_NUM_SEQS` | 32 | Max concurrent sequences |
| `SPECULATIVE_CONFIG` | (none) | JSON for MTP speculative decoding |
| `QUANTIZATION` | (auto-detect) | Quantization method |

---

## Architecture & Design Decisions

### Why FA2 JIT (not trtllm-gen)

trtllm-gen FP4 FMHA cubins only ship for SM100/SM103 — zero SM121 cubins exist. FA2 is JIT-compiled
at runtime, making it the only viable SM121 NVFP4 KV path. The FlashInfer PR #3684 adds NVFP4 FA2
kernels for SM120/121.

### Why Both PRs Are Required

- **FlashInfer PR #3684** alone: compiles kernels but vLLM won't route NVFP4 KV to FlashInfer on
  CC 12.x (guard rejects non-SM100 devices)
- **vLLM PR #46329** alone: vLLM routes correctly but FlashInfer's kernels fail on GQA group_size=6
  and asymmetric head_dim

### Prefix Caching Disabled (Correct Behavior)

Qwen3.5/3.6 uses a hybrid GDN architecture with non-causal attention layers. vLLM disables prefix
caching by design (`core.py:269`) for these models. This is not a bug — do not force-enable it.

### MTP Impact on KV Capacity

MTP (Multi-Token Prediction) speculative decoding reserves memory for the draft model, reducing the
available KV cache pool. The tradeoff is:

| Mode | KV Tokens | c1 tok/s | c32 tok/s |
|---|---|---|---|
| NVFP4 KV, 32K ctx, no MTP | 2,846,446 | 12.13 | 239.24 |
| NVFP4 KV, 8K ctx, MTP | 1,109,560 | 19.15 | 248.40 |
| FP8 KV, 32K ctx, MTP | 1,702,722 | 19.78 | 222.84 |

Use MTP for throughput-sensitive serving; disable MTP and raise `max_model_len` for maximum
long-context capacity.

---

## Verified Runtime Profile

- **vLLM**: v0.24.0 (source-built, `TORCH_CUDA_ARCH_LIST=12.1`)
- **FlashInfer**: PR #3684 branch (`nvfp4-vosplit-rederive`), compiled from source
- **Model**: nvidia/Qwen3.6-27B-NVFP4
- **Quantization**: modelopt_mixed (MLP W4A16_NVFP4 group_size=16, attention FP8, lm_head NVFP4)
- **KV Cache**: nvfp4 (FlashInfer FA2 JIT)
- **Attention Backend**: FlashInfer
- **CUDA Graphs**: PIECEWISE (MTP + FlashInfer)
- **MTP**: Active, num_speculative_tokens=1
- **GPU**: NVIDIA GB10 SM121, CUDA 13.0, Torch 2.11.0+cu130
- **Image**: 15.4 GB (`sm121-vllm-v0240-nvfp4:kv-exp`)

---

## License

Scripts, Dockerfiles, and documentation: **MIT**.

Model weights are not redistributed. Follow the upstream
[nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) license.

The FlashInfer PR #3684 and vLLM PR #46329 patches are copyrighted by their respective upstream
authors under the Apache 2.0 license.
