# AGENTS.md

## Purpose

This repository provides a reproducible, source-built **vLLM 0.25.1** runtime for serving
`nvidia/Qwen3.6-27B-NVFP4` on NVIDIA GB10 / SM121 (DGX Spark).

The production profile uses **native NVFP4 weights, FP8 KV cache, FlashInfer, and Qwen3 MTP K=2**.
The repository also carries the reviewed SM120/121 NVFP4-KV backport for explicit experiments, but
NVFP4 KV is not a production default because matched quality evidence has not cleared that profile.

## Release Identity

- Platform: NVIDIA GB10 / SM121, aarch64
- CUDA toolkit/runtime contract: 13.0
- PyTorch: 2.11.0+cu130
- vLLM: 0.25.1, commit `752a3a504485790a2e8491cacbb35c137339ad34`
- FlashInfer: commit `741b63720bb345d9036d38b33a7b5a043d4c2674`
- Model: `nvidia/Qwen3.6-27B-NVFP4`
- Model revision: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`
- Quantization: calibrated native `modelopt_mixed` W4A4/W4A16 routing; no Marlin or emulation fallback
- Production KV cache: FP8
- Production speculative decoding: `mtp`, two speculative tokens
- Validated production serving envelope: 8,192-token configured context, 32 sequences, 8,192 batched tokens, 0.75 GPU-memory utilization
- OpenAI chat parsers: `qwen3` reasoning and `qwen3_xml` tool calls
- Prefix caching: disabled

`docker/runtime-manifest.json` and `docker/backport-equivalence-v0.25.1.json` are the machine-readable
release contracts.

## Build Constraints

1. Keep CUDA at 13.0. PyTorch 2.11.0+cu130 and the source-built extensions must share that toolkit.
2. Keep `MAX_JOBS=6` and `NVCC_THREADS=2`. Higher compile fan-out has caused host OOM failures.
3. A clean GB10 build takes about an hour; preserve it in tmux and record explicit PASS/FAIL stamps.
4. The runtime image needs gcc, ninja, nvcc, Python headers, and CUDA development headers for
   FlashInfer JIT.
5. Build only for the native SM121 path. Do not add Marlin, emulation, fallback packages, or metadata
   shortcuts.
6. Build from a clean, exact source commit and set `IMAGE_REVISION` to that full commit SHA.

## Critical Runtime Requirements

The runtime stage must retain:

- `build-essential`
- `ninja-build`
- `cuda-nvcc-13-0`
- `cuda-libraries-dev-13-0`
- `python3.12-dev`
- the complete `/opt/vllm` environment
- `R0B0TLAB_QWEN27_NATIVE_W4A4=1`

The entrypoint must run the runtime audit first, preserve nonzero argv boundaries with `exec "$@"`,
and return the exact child exit code. `audit` is a dedicated subcommand.

## Upstream and Backport Contract

| Component | Identity | Source |
|---|---|---|
| vLLM | 0.25.1 / `752a3a5…` | github.com/vllm-project/vllm |
| FlashInfer | `741b637…` | github.com/jethac/flashinfer |
| vLLM NVFP4-KV series | 7 reviewed commits through `dfed053…` | upstream PR #46329 |
| PyTorch | 2.11.0+cu130 | download.pytorch.org/whl/cu130 |

The cumulative v0.25.1 ports are tracked in:

- `docker/vllm-pr46329-v0.25.1.diff`
- `docker/native-w4a4-qwen27-v0.25.1.diff`
- `docker/backport-equivalence-v0.25.1.json`

Do not claim a clean cherry-pick when a release-aware port was required. Preserve upstream authorship
and links without adding tracking parameters.

## Verification Procedure

Before publishing or committing:

1. `python3 scripts/public_safety_scan.py .`
2. `bash -n scripts/entrypoint.sh`
3. `python3 -m py_compile scripts/*.py tests/*.py`
4. `python3 tests/test_release_contract.py`
5. `python3 tests/test_launch_contract.py`
6. `python3 tests/test_benchmark_harness_scaffold.py`
7. `python3 scripts/verify_backport.py .`
8. `docker run --rm --gpus all <image> audit`
9. Verify entrypoint argv preservation and package/stable-ABI imports.
10. Launch the exact image/model and run semantic, long-generation, MTP-metric, log, and matched
    performance gates.
11. Run the benchmark harness with an explicit run-owned `--container-name`.

Synthetic harness tests are scaffold evidence only; they are not runtime or benchmark proof.

## Benchmark and Evaluation Policy

- Default accuracy protocol: GSM8K 0-shot, flexible-extract.
- Do not repeat a full evaluation when image, model, tokenizer/template, command, and serving behavior
  are equivalent. Reuse hash-verified evidence and run narrow packaging-equivalence canaries.
- Separate client output throughput from server prompt/decode throughput.
- Report TTFT, stream-event interarrival/TPOT definitions, p50/p90/p99, MTP drafted/accepted tokens,
  power, temperature, utilization, available host memory, swap, and all request errors.
- Wait for the scheduler to become idle between profiles and inspect live logs during decode.
- Treat profile/config changes as per-model evidence; do not merge unmatched runs.

## File Map

```text
docker/Dockerfile.production                 vLLM 0.25.1 production FP8-KV source build
docker/Dockerfile.kv-exp                     separate NVFP4-KV experimental source build
docker/vllm-pr46329-v0.25.1.diff             reviewed NVFP4-KV release port (experimental image only)
docker/native-w4a4-qwen27-v0.25.1.diff       calibrated native Qwen weight-routing port
docker/backport-equivalence-v0.25.1.json     commit/path equivalence map
docker/runtime-manifest.production.json      production FP8-KV release identity
docker/runtime-manifest.json                 experimental NVFP4-KV release identity
scripts/entrypoint.sh                         audit + command-driven launch contract
scripts/audit_runtime.py                      fail-closed runtime audit
scripts/benchmark_nvfp4_kv.py                 streaming benchmark and telemetry harness
scripts/public_safety_scan.py                 pre-publish safety scanner
sparkrun/                                     pinned single-node FP8/MTP recipe
results/                                      historical evidence; preserve provenance
```

## Operating Rules

- Do not redistribute model weights.
- Do not change CUDA from 13.0 without a complete compatibility requalification.
- Do not increase build parallelism above `MAX_JOBS=6`, `NVCC_THREADS=2` on GB10.
- Do not enable prefix caching for Qwen3.5/3.6 hybrid GDN models.
- Do not use trtllm-gen NVFP4 attention on SM121; no compatible SM121 cubins are available.
- Do not present NVFP4 KV as production-safe until matched semantic/quality gates pass.
- Do not hide fallback, corruption, empty output, repeated punctuation, or template leakage.
- Always run the public-safety scan before pushing.
- Keep private paths, LAN addresses, credentials, raw diagnostics, and unpublished evidence out of
  tracked public files.
- This is an r0b0tlab repository. Do not submit upstream changes unless explicitly requested.
