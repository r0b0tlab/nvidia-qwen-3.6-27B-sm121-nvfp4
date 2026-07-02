#!/usr/bin/env python3
"""SM121 native NVFP4 runtime audit for vLLM v0.24.0.

v0.24.0 uses the stable ABI migration:
  - _C_stable_libtorch replaces _C
  - _moe_C_stable_libtorch replaces _moe_C
  - FA extensions are built from source (not deleted)

Verifies:
  1. vLLM reports v0.24.0
  2. Correct GPU compute capability (SM121 / GB10)
  3. Core compiled extensions (stable ABI naming)
  4. Qwen3.5 model support registered
  5. modelopt_mixed quantization available
  6. NVFP4 quantization available

Exits 0 if all gates pass, 1 otherwise.
"""
import sys

def main():
    checks = []

    import torch
    import vllm

    # vLLM version
    ver = getattr(vllm, "__version__", "unknown")
    checks.append(("vllm_version", "0.24" in ver, f"got {ver}"))

    # Device
    cap = torch.cuda.get_device_capability()
    checks.append(("cuda_capability", cap == (12, 1), f"got {cap}"))

    # Core extensions — v0.24.0 uses stable ABI
    for mod in ["vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"]:
        try:
            __import__(mod)
            checks.append((f"import_{mod}", True, ""))
        except Exception as e:
            checks.append((f"import_{mod}", False, str(e)[:100]))

    # Qwen3.5 model support
    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        checks.append(("qwen3_5_model", True, ""))
    except Exception as e:
        checks.append(("qwen3_5_model", False, str(e)[:120]))

    # modelopt_mixed quantization
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
        checks.append(("modelopt_mixed", True, ""))
    except Exception as e:
        checks.append(("modelopt_mixed", False, str(e)[:120]))

    # NVFP4 quantization
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
        checks.append(("nvfp4_quant", True, ""))
    except Exception as e:
        checks.append(("nvfp4_quant", False, str(e)[:120]))

    # Report
    failed = [n for n, ok, _ in checks if not ok]
    if failed:
        print("AUDIT FAIL: SM121 native NVFP4 runtime gates not satisfied")
        for n, ok, d in checks:
            print(f"  {'✗' if not ok else '✓'} {n}" + (f" ({d})" if d else ""))
        sys.exit(1)

    print("AUDIT PASS: SM121 native NVFP4 runtime gates satisfied")
    for n, ok, d in checks:
        print(f"  ✓ {n}" + (f" ({d})" if d else ""))

if __name__ == "__main__":
    main()
