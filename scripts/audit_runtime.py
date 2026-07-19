#!/usr/bin/env python3
"""Fail-closed pre-load audit for Qwen3.6-27B SM121 runtime images."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

EXPECTED = {
    "vllm": "0.25.1",
    "vllm_package": "0.25.1+r0b0tlab.w4a4.1",
    "vllm_tag": "v0.25.1",
    "vllm_commit": "752a3a504485790a2e8491cacbb35c137339ad34",
    "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
}


def main() -> int:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: Any = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    manifest_path = Path("/opt/r0b0tlab/runtime-manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        manifest = {}
        add("runtime_manifest", False, repr(exc))

    profile = manifest.get("profile")
    nvfp4_enabled = manifest.get("nvfp4_kv_enabled") is True
    expected_flashinfer = manifest.get("flashinfer_version")
    add("manifest_profile", profile == "production-fp8", profile)
    add("manifest_vllm_version", manifest.get("vllm_version") == EXPECTED["vllm"], manifest.get("vllm_version"))
    add("manifest_vllm_package", manifest.get("vllm_package_version") == EXPECTED["vllm_package"], manifest.get("vllm_package_version"))
    add("manifest_vllm_tag", manifest.get("vllm_tag") == EXPECTED["vllm_tag"], manifest.get("vllm_tag"))
    add("manifest_vllm_commit", manifest.get("vllm_commit") == EXPECTED["vllm_commit"], manifest.get("vllm_commit"))
    add("manifest_model_revision", manifest.get("model_revision") == EXPECTED["model_revision"], manifest.get("model_revision"))
    add("manifest_flashinfer_version", isinstance(expected_flashinfer, str) and bool(expected_flashinfer), expected_flashinfer)
    add("manifest_default_kv", manifest.get("default_kv_cache_dtype") == "fp8", manifest.get("default_kv_cache_dtype"))
    if profile == "production-fp8":
        add("production_nvfp4_disabled", not nvfp4_enabled, nvfp4_enabled)
        add("production_flashinfer_release", expected_flashinfer == "0.6.13", expected_flashinfer)

    try:
        import torch
        import vllm

        version = metadata.version("vllm")
        flashinfer_version = metadata.version("flashinfer-python")
        add("vllm_version", version == EXPECTED["vllm_package"], version)
        add("flashinfer_version", flashinfer_version == expected_flashinfer, flashinfer_version)
        add("torch_version", torch.__version__.startswith("2.11.0+cu130"), torch.__version__)
        add("torch_cuda", torch.version.cuda == "13.0", torch.version.cuda)
        capability = torch.cuda.get_device_capability()
        add("cuda_capability", capability == (12, 1), capability)
        add("vllm_module_version", getattr(vllm, "__version__", None) == EXPECTED["vllm_package"], getattr(vllm, "__version__", None))
    except Exception as exc:
        add("core_imports", False, repr(exc))

    for module in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        try:
            __import__(module)
            add(f"import_{module}", True)
        except Exception as exc:
            add(f"import_{module}", False, repr(exc))

    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: F401
        add("qwen3_5_model", True)
    except Exception as exc:
        add("qwen3_5_model", False, repr(exc))

    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMixedPrecisionConfig,  # noqa: F401
            ModelOptNvFp4Config,  # noqa: F401
        )
        add("modelopt_mixed_nvfp4", True)
    except Exception as exc:
        add("modelopt_mixed_nvfp4", False, repr(exc))

    vllm_cli = shutil.which("vllm")
    add("vllm_on_path", bool(vllm_cli), vllm_cli)
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            nvcc_text = subprocess.check_output([nvcc, "--version"], text=True, timeout=20)
            add("nvcc_13_0", "release 13.0" in nvcc_text, nvcc_text.splitlines()[-1])
        except Exception as exc:
            add("nvcc_13_0", False, repr(exc))
    else:
        add("nvcc_13_0", False, "not on PATH")

    try:
        pip_check = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        pip_detail = (pip_check.stdout or pip_check.stderr).strip()
        add("pip_check", pip_check.returncode == 0, pip_detail)
    except Exception as exc:
        add("pip_check", False, repr(exc))

    try:
        import vllm.model_executor.layers.quantization.modelopt as modelopt
        import vllm.v1.attention.backends.flashinfer as fi_backend

        modelopt_source = Path(modelopt.__file__).read_text()
        fi_source = Path(fi_backend.__file__).read_text()
        add("native_w4a4_patch", "R0B0TLAB_NATIVE_W4A4_FROM_W4A16" in modelopt_source)
        sm12x_marker = "consumer Blackwell (sm120/sm121)" in fi_source
        hnd_marker = "NVFP4 KV cache requires the HND KV cache layout" in fi_source
        add("nvfp4_kv_patch_profile_match", sm12x_marker == nvfp4_enabled, sm12x_marker)
        add("hnd_patch_profile_match", hnd_marker == nvfp4_enabled, hnd_marker)
    except Exception as exc:
        add("source_markers", False, repr(exc))

    add("native_w4a4_enabled", os.getenv("R0B0TLAB_QWEN27_NATIVE_W4A4") == "1", os.getenv("R0B0TLAB_QWEN27_NATIVE_W4A4"))
    env_nvfp4_enabled = os.getenv("R0B0TLAB_NVFP4_KV_ENABLED") == "1"
    add("nvfp4_env_profile_match", env_nvfp4_enabled == nvfp4_enabled, env_nvfp4_enabled)
    add("max_jobs", os.getenv("MAX_JOBS") == "6", os.getenv("MAX_JOBS"))
    add("flashinfer_nvcc_threads", os.getenv("FLASHINFER_NVCC_THREADS") == "2", os.getenv("FLASHINFER_NVCC_THREADS"))

    failed = [item for item in checks if not item["ok"]]
    report = {
        "schema_version": 2,
        "status": "PASS" if not failed else "FAIL",
        "expected": EXPECTED,
        "manifest": manifest,
        "checks": checks,
    }
    print("R0B0TLAB_RUNTIME_AUDIT=" + json.dumps(report, sort_keys=True))
    for item in checks:
        print(f"{'PASS' if item['ok'] else 'FAIL'} {item['name']}: {item['detail']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
