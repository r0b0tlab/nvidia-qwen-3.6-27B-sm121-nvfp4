# NVIDIA Qwen3.6-27B NVFP4 on GB10 / SM121

A reproducible, source-built vLLM 0.25.1 runtime for `nvidia/Qwen3.6-27B-NVFP4` on one NVIDIA GB10 / DGX Spark.

Production profile:

- native calibrated NVFP4 W4A4 weight kernels on SM121
- FP8 KV cache and FlashInfer attention
- MTP K=2
- no active Marlin, emulation, or weight fallback
- 8,192-token configured context, 32 sequences, and 8,192 batched tokens
- Qwen reasoning and `qwen3_xml` tool-call parsers

NVFP4 KV was **not promoted**. It remains isolated in the experimental Dockerfile; production uses FP8 KV.

## Release identity

| Component | Pinned identity |
|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` |
| Model revision | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` |
| vLLM | `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` |
| vLLM package | `0.25.1+r0b0tlab.w4a4.1` |
| FlashInfer | `0.6.13` |
| PyTorch | `2.11.0+cu130` |
| CUDA | `13.0` |
| Target | Linux aarch64, NVIDIA GB10, compute capability 12.1 |
| Source commit used for image | `4401cbc4720b06e5db3136eaaa18dd431bf6f52e` |
| Local qualified image ID | `sha256:29c75d7b04295d7875fe5b2a629283fdc69fdf476e658e5e892ac951e82b96f8` |
| Public immutable image | `ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0` |

## Prebuilt image

Versioned tag:

```bash
docker pull ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-production
```

Immutable pull:

```bash
docker pull ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0
```

Run the fail-closed runtime audit:

```bash
docker run --rm --gpus all ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0 audit
```

Serve a local model checkout using the image's qualified defaults:

```bash
docker run --rm --gpus all --ipc=host   --name qwen36-27b   -p 8000:8000   -v /path/to/nvidia-Qwen3.6-27B-NVFP4:/models/nvidia-Qwen3.6-27B-NVFP4:ro   ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0
```

The default launch resolves to FP8 KV, FlashInfer, MTP K=2, 8K context, reasoning parser `qwen3`, and tool parser `qwen3_xml`.

Explicit commands are preserved after the same audit via `exec "$@"`. For example:

```bash
docker run --rm --gpus all ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0 vllm --version
```

The production entrypoint rejects `--kv-cache-dtype nvfp4` before model load.

## sparkrun

The in-repo v2 recipe pins the exact model revision and immutable image:

```bash
sparkrun registry add https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4
sparkrun recipe validate @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab
sparkrun run @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab --solo
```

Recipe: `sparkrun/recipes/qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml`.

## v0.25.1 production results

Measured on one GB10 with the exact immutable image above, model revision `0893e1606ff3d5f97a441f405d5fc541a6bdf404`, FP8 KV, and MTP K=2. Each level used one warmup and three measured repeats with 256 completion tokens per request. All requests succeeded.

| Concurrency | Aggregate output tok/s, median | Three-repeat range | TTFT p50 | ITL p50 | MTP acceptance | Mean power |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.57 | 26.53–26.58 | 218 ms | 100 ms | 85.6% | 40.3 W |
| 2 | 49.10 | 47.91–49.10 | 285 ms | 102 ms | 82.9% | 40.8 W |
| 4 | 92.95 | 92.92–92.97 | 362 ms | 108 ms | 81.1% | 41.9 W |
| 8 | 163.58 | 163.55–163.87 | 432 ms | 123 ms | 85.4% | 44.0 W |
| 16 | 228.28 | 228.21–228.35 | 600 ms | 180 ms | 84.3% | 50.7 W |
| 32 | 400.89 | 383.56–403.77 | 994 ms | 203 ms | 82.9% | 57.9 W |

Aggregate MTP acceptance across the final sweep: **83.46%** (30,231 accepted / 36,220 drafted).

FP8 KV capacity was **700,142 tokens** with **66.56 GiB** available KV memory.

### MTP-depth selection

The same runtime was screened at K=1/2/3/4 with three measured repeats at c1, c8, and c32:

| MTP depth | c1 tok/s | c8 tok/s | c32 tok/s | Acceptance | KV capacity |
|---:|---:|---:|---:|---:|---:|
| K=1 | 20.56 | 134.78 | 332.90 | 87.7% | 877,909 |
| K=2 | 27.07 | 164.55 | 397.40 | 82.7% | 702,327 |
| K=3 | 30.16 | 174.13 | 396.83 | 74.2% | 583,452 |
| K=4 | 32.20 | 165.01 | 338.64 | 64.9% | 490,739 |

K=2 is the production default because it delivered the best mixed-concurrency balance. K=3 improved c1/c8 but did not improve c32 over K=2; K=4 reduced acceptance and high-concurrency efficiency sharply.

## Quality and API gates

- GSM8K 0-shot flexible extract, chat completions, thinking disabled: **87.49%** over **1,319** samples
- request errors: 0
- deterministic semantic canaries: PASS
- tool call parser: PASS
- 7,973-token retrieval: PASS
- 768-token long generation: PASS
- native W4A4 and FlashInfer markers: PASS
- active Marlin, emulation, and fallback markers: zero

The public aggregate is under `results/v0251-node2/`; raw samples and host telemetry remain private.

The earlier v0.24.0 GSM8K result remains historical and is not relabeled as v0.25.1 evidence.

## Reproduce the build

```bash
git clone https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4.git
cd nvidia-qwen-3.6-27B-sm121-nvfp4

docker build --progress=plain   --build-arg IMAGE_REVISION="$(git rev-parse HEAD)"   -f docker/Dockerfile.production   -t qwen27-vllm:0.25.1-production .
```

Important constraints:

- `MAX_JOBS=6`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`
- CUDA toolkit 13.0 matching PyTorch cu130
- exact vLLM tag and commit verification
- runtime compiler and CUDA development headers for FlashInfer JIT
- separate production and experimental Dockerfiles

Run the repository gates:

```bash
python3 scripts/public_safety_scan.py .
bash -n scripts/entrypoint.sh
python3 -m py_compile scripts/*.py tests/*.py
python3 tests/test_release_contract.py
python3 tests/test_launch_contract.py
python3 tests/test_benchmark_harness_scaffold.py
python3 tests/test_dependency_check.py
python3 tests/test_verify_release.py
python3 scripts/verify_backport.py .
python3 scripts/verify_release.py
```

## Production versus NVFP4-KV experiment

`docker/Dockerfile.production` uses vLLM 0.25.1 and FlashInfer 0.6.13 plus the checkpoint-scoped native W4A4 reroute. It hard-disables NVFP4 KV.

`docker/Dockerfile.kv-exp` preserves the reviewed SM120/121 NVFP4-KV work for investigation. It is not the production image, recipe default, or a quality claim.

## Upstream credit

- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Apache-2.0
- NVFP4-KV SM120/121 work by [@jethac](https://github.com/jethac), tracked through vLLM PR #46329 and related FlashInfer work
- [NVIDIA Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
- [sparkrun](https://github.com/spark-arena/sparkrun) and issue reporter [@mrpmorris](https://github.com/mrpmorris) for the packaging request

## License

Repository scripts and documentation are MIT licensed. Upstream patches retain their original Apache-2.0 licensing and authorship. Model weights are not redistributed; follow the model repository's license and terms.
