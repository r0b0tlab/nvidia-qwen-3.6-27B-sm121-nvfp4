# NVIDIA Qwen3.6-27B NVFP4 — SM121 Native + NVFP4 KV Cache

Optimized vLLM v0.24.0 runtime for **nvidia/Qwen3.6-27B-NVFP4** on NVIDIA GB10 / SM121 (DGX Spark),
with **native NVFP4 KV cache** via FlashInfer FA2 JIT — achieving **67% more KV capacity** over FP8.

## Highlights

| Metric | FP8 KV Baseline | NVFP4 KV Cache | Δ |
|---|---|---|---|
| **KV Cache Tokens** | 1,702,722 | **2,846,446** | **+67%** |
| **Max Concurrency (32K ctx)** | ~52× | **86.87×** | **+67%** |
| **KV Cache Memory** | 61.48 GiB | **62.29 GiB** | ~same budget |
| **KV Cache Dtype** | fp8 | **nvfp4** | 4-bit |

### Throughput (256-token generation, 8K context)

| Concurrency | Output tok/s | Power (W) | Efficiency (J/1K tok) | Temp (°C) |
|---:|---:|---:|---:|---:|
| 1 | 12.13 | 37.1 | 3,056 | 68.8 |
| 4 | 44.91 | 36.8 | 820 | 69.1 |
| 8 | 85.32 | 37.0 | 434 | 69.0 |
| 16 | 150.03 | 38.5 | 256 | 69.0 |
| 32 | 239.24 | 39.8 | 167 | 69.5 |

All 32/32 requests succeeded at c32. Power stays under 40W — well within GB10's 75W envelope.

### Sanity Suite (5/5 passed)

| Test | Tokens | Latency |
|---|---|---|
| Math (17×23) | 32 | 2.65s |
| Code (reverse string) | 64 | 5.27s |
| Logical reasoning (syllogism) | 64 | 5.28s |
| Factual (capital of Australia) | 32 | 2.65s |
| Instruction-following (3 colors) | 32 | 2.64s |

---

## What This Repo Contains

This is a **complete reproducibility pack** for serving NVFP4-quantized models on SM121 with native
NVFP4 KV cache. The runtime is built from source with two upstream patches that are not yet merged:

- **FlashInfer PR #3684** — NVFP4 vector casts, GQA group_size=6 support, asymmetric head_dim (128 QK vs 64 VO)
- **vLLM PR #46329** — Lift SM100-only NVFP4 KV guard, route to FA2 on CC 12.x

### Files

```
docker/
  Dockerfile.kv-exp         Multi-stage build: vLLM v0.24.0 + FlashInfer PR#3684 + vLLM PR#46329
  vllm-pr46329.diff         73KB patch lifting NVFP4 KV SM100-only guard
scripts/
  entrypoint.sh             Container entrypoint: audit → serve with NVFP4 KV + MTP defaults
  audit_runtime.py          SM121 native runtime gate verification (6 checks)
  benchmark_nvfp4_kv.py     Sanity suite + concurrency ramp + power telemetry
  public_safety_scan.py     Pre-publish secret/PII scanner
results/
  benchmark-nvfp4-kv/       NVFP4 KV cache benchmark (c1–c32, sanity, telemetry)
  baseline-fp8-32k/         FP8 KV cache baseline at 32K context
  smoke_and_ramp.json       Earlier smoke+ramp data
```

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
git clone https://github.com/r0b0tlab/nvidia-qwen-3.6-sm121-nvfp4.git
cd nvidia-qwen-3.6-sm121-nvfp4

# ~60 min with MAX_JOBS=6 on GB10
docker build -f docker/Dockerfile.kv-exp -t sm121-vllm-v0240-nvfp4:kv-exp .
```

### Serve with NVFP4 KV Cache

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
| `SPECULATIVE_CONFIG` | (none) | JSON for speculative decoding |
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

### NVFP4 KV Value: Long-Context Serving

The 67% KV capacity gain matters most for **long-context serving** (32K–131K tokens). At 8K context,
KV usage is <5% of budget. The real win is enabling concurrent long-context requests that would
OOM with FP8 KV.

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
