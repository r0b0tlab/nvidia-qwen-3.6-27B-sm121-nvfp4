# NVIDIA Qwen3.6-27B NVFP4 on GB10 / SM121

A reproducible, source-built vLLM 0.25.1 runtime for `nvidia/Qwen3.6-27B-NVFP4` on one NVIDIA GB10 / DGX Spark.

The production profile is intentionally narrow:

- native calibrated NVFP4 W4A4 weight kernels on SM121
- FP8 KV cache
- FlashInfer attention
- MTP with one speculative token
- no Marlin, emulation, or weight-only fallback
- an 8,192-token validated serving profile
- Qwen reasoning and XML tool-call parsers

NVFP4 KV remains a separate experiment. It is not enabled in the production image and is not claimed as quality-safe here.

## Release identity

| Component | Pinned identity |
|---|---|
| Model | `nvidia/Qwen3.6-27B-NVFP4` |
| Model revision | `0893e1606ff3d5f97a441f405d5fc541a6bdf404` |
| vLLM | `v0.25.1`, commit `752a3a504485790a2e8491cacbb35c137339ad34` |
| vLLM package | `0.25.1+r0b0tlab.w4a4.1` |
| FlashInfer | `0.6.13` |
| PyTorch | `2.11.0+cu130` |
| CUDA | 13.0 |
| Target | Linux aarch64, NVIDIA GB10, compute capability 12.1 |
| Production KV | FP8 |
| Production MTP | `{"method":"mtp","num_speculative_tokens":1}` |

The checkpoint-scoped W4A4 reroute is recorded in `docker/native-w4a4-qwen27-v0.25.1.diff`. It uses the checkpoint's finite calibrated `input_scale` tensors with vLLM's native NVFP4 W4A4 kernel. Routed and ordinary linear paths were admitted only after logs showed native FlashInfer/CUTLASS selection and zero `MarlinNvFp4`, `Using 'MARLIN'`, or emulation markers.

## Prebuilt image

```bash
docker pull ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-production
```

The image is source-built for CUDA 13.0 / SM121 and includes the compiler, Ninja, CUDA development headers, and bounded FlashInfer JIT settings required at runtime.

Run the fail-closed audit:

```bash
docker run --rm --gpus all \
  ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-production audit
```

Serve a local model checkout:

```bash
docker run --rm --gpus all --ipc=host \
  --name qwen36-27b \
  -p 18080:8000 \
  -v /path/to/nvidia-Qwen3.6-27B-NVFP4:/models/model:ro \
  ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-production \
  vllm serve /models/model \
    --served-model-name Qwen3.6-27B-NVFP4 \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
    --dtype bfloat16 --attention-backend FLASHINFER \
    --kv-cache-dtype fp8 --max-model-len 8192 \
    --max-num-seqs 32 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.75 --language-model-only \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

The entrypoint audits the runtime before either its zero-argument default launch or an explicit command, preserves argument boundaries with `exec "$@"`, and returns the child process's exact exit code. The production image rejects `--kv-cache-dtype nvfp4` before model load.

## sparkrun

The repository includes a v2 recipe with the model revision, image tag, and tested production defaults pinned:

```bash
sparkrun registry add https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4
sparkrun recipe validate @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab
sparkrun run @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab --solo
```

Recipe source: `sparkrun/recipes/qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml`.

## v0.25.1 production qualification

Measured on one GB10 with the exact production image, model revision, FP8 KV, MTP K=1, 8K configured context, and Qwen reasoning/tool parsers. Each concurrency has one warmup plus three measured repeats with 256 completion tokens per request. All 189 measured requests completed successfully.

Client completion throughput includes both parsed reasoning and final-content tokens reported by the OpenAI-compatible usage object.

| Concurrency | Client completion tok/s, median | Three-repeat range | Mean power | Median p50 TTFT | Median p50 ITL |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.25 | 17.79–18.25 | 40.37 W | 343 ms | 102 ms |
| 2 | 35.25 | 35.13–36.38 | 41.07 W | 317 ms | 104 ms |
| 4 | 66.69 | 64.80–69.44 | 41.52 W | 343 ms | 109 ms |
| 8 | 118.29 | 118.02–121.85 | 43.84 W | 392 ms | 121 ms |
| 16 | 168.91 | 166.03–221.99 | 47.45 W | 498 ms | 178 ms |
| 32 | 322.93 | 195.55–328.41 | 53.93 W | 1,097 ms | 171 ms |

Additional runtime evidence:

- MTP accepted 32,130 of 34,724 drafted tokens: 92.53%
- FP8 KV capacity: 884,736 tokens
- available KV memory: 66.57 GiB
- reported maximum concurrency: 108.00× at 8,192 tokens/request
- streaming transport sanity: 5/5
- reasoning parser and `qwen3_xml` tool-call parser: passed live API canaries
- benchmark errors: zero; warmup errors: zero

The full raw qualification is intentionally kept as machine-local evidence rather than committed telemetry. The public harness and synthetic parser tests are included so the procedure can be reproduced without publishing host-specific paths or logs.

## Accuracy evidence and claim boundary

The unchanged model has a retained full-set GSM8K 0-shot flexible-extract result of **81.88%** from the earlier v0.24.0 FP8-KV/MTP campaign. Its public aggregate, samples, log, and SHA-256 manifest remain under `results/gsm8k-full-0shot-node2-20260703/`.

That historical score is model evidence, not a newly measured v0.25.1 headline. This update changed the serving runtime and native linear-kernel routing, so the current release relies on deterministic semantics, parser/tool canaries, context retrieval, long generation, and the measured concurrency ramp. It does not imply that the historical GSM8K score was rerun on the new image.

## Build from source

```bash
git clone https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4.git
cd nvidia-qwen-3.6-27B-sm121-nvfp4

docker build --progress=plain \
  --build-arg IMAGE_REVISION="$(git rev-parse HEAD)" \
  -f docker/Dockerfile.production \
  -t qwen27-vllm:0.25.1-production .
```

Important build/runtime constraints:

- `MAX_JOBS=6`, `NVCC_THREADS=2`, and `FLASHINFER_NVCC_THREADS=2`
- CUDA toolkit 13.0, matching PyTorch cu130
- exact vLLM tag and commit verification before compilation
- full runtime compiler and CUDA development toolchain for FlashInfer JIT
- separate production and experimental Dockerfiles

Run repository gates:

```bash
python3 scripts/public_safety_scan.py .
bash -n scripts/entrypoint.sh
python3 -m py_compile scripts/*.py tests/*.py
python3 tests/test_release_contract.py
python3 tests/test_launch_contract.py
python3 tests/test_benchmark_harness_scaffold.py
python3 tests/test_dependency_check.py
python3 scripts/verify_backport.py .
```

## Production versus NVFP4-KV experiment

`docker/Dockerfile.production` uses the official vLLM 0.25.1 and FlashInfer 0.6.13 release stack plus only the checkpoint-scoped native W4A4 reroute. It hard-disables NVFP4 KV.

`docker/Dockerfile.kv-exp` separately retains the reviewed SM120/121 NVFP4-KV release-aware port for investigation. The backport is preserved for upstream provenance and reproducibility, but it is not the production image, the sparkrun default, or a quality claim. Do not promote it without matched semantic and quality evidence.

## Upstream credit

- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer), Apache-2.0
- NVFP4-KV SM120/121 work by upstream contributor [@jethac](https://github.com/jethac), tracked through vLLM PR #46329 and the related FlashInfer work
- [NVIDIA Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)

## License

Repository scripts and documentation are MIT licensed. Upstream patches retain their original Apache-2.0 licensing and authorship. Model weights are not redistributed; follow the model repository's license and terms.
