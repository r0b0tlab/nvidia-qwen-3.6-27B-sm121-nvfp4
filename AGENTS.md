# AGENTS.md

## Purpose

This repository provides a reproducible, source-built vLLM v0.24.0 runtime for serving
**nvidia/Qwen3.6-27B-NVFP4** on NVIDIA GB10 / SM121 (DGX Spark) with **native NVFP4 KV cache**.

It combines two unmerged upstream PRs (FlashInfer #3684 + vLLM #46329) into a single Docker image
that achieves 67% more KV cache capacity than FP8.

## Key Facts for Agents

- **Platform**: NVIDIA GB10 / SM121 (DGX Spark), aarch64
- **CUDA**: 13.0 exactly — NOT 13.2 (PTX incompatibility, see Fix #3 in README)
- **PyTorch**: 2.11.0+cu130 (ships CUDA 13.0 runtime)
- **vLLM**: v0.24.0 source-built with `TORCH_CUDA_ARCH_LIST=12.1`
- **FlashInfer**: PR #3684 branch compiled from source
- **Model**: nvidia/Qwen3.6-27B-NVFP4 with modelopt_mixed quantization
- **KV Cache**: NVFP4 via FlashInfer FA2 JIT (not trtllm-gen — no SM121 cubins exist)
- **Build parallelism**: `MAX_JOBS=6` + `NVCC_THREADS=2` mandatory — higher values trigger OOM killer

## Build Constraints

1. **MAX_JOBS=6 is non-negotiable.** The GB10 system has 121 GB RAM but system services
   (hermes-gateway, oom_score_adj=200) are OOM-killer targets. MAX_JOBS=20 spawns ~20 parallel nvcc
   processes (~40 GB spike) and kills the build at ~100/371 objects.
2. **CUDA 13.0 only.** PyTorch 2.11.0 ships CUDA 13.0 runtime. Building with CUDA 13.2 toolkit
   produces PTX that crashes at `prepare_fp4_layer_for_marlin` with "unsupported toolchain."
3. **Full rebuild takes ~60 minutes.** The 371-object CUDA compile is the bottleneck (~57 min).
   Flash attention kernels (objects ~130–220) are slowest at ~2 obj/min.
4. **Runtime image needs dev tools.** FlashInfer JIT requires gcc, ninja, nvcc, and CUDA dev headers
   at runtime. Omitting any of these causes a crash during KV cache initialization.

## Critical Runtime Requirements

The Docker runtime stage MUST include:
- `build-essential` (gcc/cc for FlashInfer JIT)
- `ninja-build` (build system for FlashInfer JIT)
- `cuda-nvcc-13-0` (nvcc for FlashInfer JIT CUDA kernels)
- `cuda-libraries-dev-13-0` (CUDA dev headers: curand_kernel.h, etc.)
- `python3.12-dev` (Python headers for JIT)

## Upstream Dependencies

| Component | Version/Branch | Source |
|---|---|---|
| vLLM | v0.24.0 (tag) | github.com/vllm-project/vllm |
| FlashInfer | PR #3684 (`nvfp4-vosplit-rederive`) | github.com/jethac/flashinfer |
| vLLM PR #46329 | diff applied via `docker/vllm-pr46329.diff` | github.com/vllm-project/vllm |
| PyTorch | 2.11.0+cu130 | download.pytorch.org/whl/cu130 |

## Verification Procedure

Before publishing or committing:

1. **Safety scan**: `python3 scripts/public_safety_scan.py`
2. **Bash syntax**: `bash -n scripts/entrypoint.sh`
3. **Python syntax**: `python3 -m py_compile scripts/*.py`
4. **Docker audit**: `docker run --rm --gpus all --entrypoint bash <image> audit`
5. **Benchmark**: `python3 scripts/benchmark_nvfp4_kv.py --model Qwen3.6-27B-NVFP4`

## File Map

```
docker/Dockerfile.kv-exp       Source build: vLLM + FlashInfer + PR patches
docker/vllm-pr46329.diff       73KB patch: lift SM100 NVFP4 KV guard → SM121
scripts/entrypoint.sh          Entrypoint: audit → serve with NVFP4 KV defaults
scripts/audit_runtime.py       6-gate runtime verification
scripts/benchmark_nvfp4_kv.py  Sanity + concurrency ramp + power telemetry
scripts/public_safety_scan.py  Pre-publish secret scanner
results/                        Benchmark artifacts (JSON)
```

## Operating Rules

- Do NOT redistribute model weights — they are licensed separately by NVIDIA.
- Do NOT change CUDA toolkit version from 13.0 — this is root-caused (see Fix #3 in README).
- Do NOT increase MAX_JOBS above 6 — this is root-caused (see Fix #1 in README).
- Do NOT enable prefix caching — Qwen3.5/3.6 hybrid GDN architecture has non-causal attention;
  vLLM disables it by design.
- Do NOT use the trtllm-gen attention backend for NVFP4 KV on SM121 — no SM121 cubins exist.
  Use FlashInfer FA2 JIT (the path this repo provides).
- Always run the safety scan before pushing.
- This is a r0b0tlab-only repo. No upstream contributions unless explicitly requested.
