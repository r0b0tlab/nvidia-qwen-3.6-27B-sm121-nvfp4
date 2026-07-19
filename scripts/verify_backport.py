#!/usr/bin/env python3
"""Validate immutable release/backport inputs before a Docker build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

RELEASE = "752a3a504485790a2e8491cacbb35c137339ad34"
PR_HEAD = "dfed053c9e5cddc2ea35939e5dcf439f69290a57"
FLASHINFER = "741b63720bb345d9036d38b33a7b5a043d4c2674"
MODEL = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
EXPECTED_PATHS = {
    "csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu",
    "tests/v1/attention/test_gemma4_nvfp4_flashinfer_routing.py",
    "tests/v1/attention/test_nvfp4_flashinfer_vosplit_mm.py",
    "vllm/envs.py",
    "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py",
    "vllm/model_executor/models/config.py",
    "vllm/v1/attention/backends/flashinfer.py",
}


def diff_paths(text: str) -> set[str]:
    return {
        match.group(1)
        for match in re.finditer(r"^diff --git a/\S+ b/(\S+)$", text, re.MULTILINE)
    }


def validate(root: Path) -> list[str]:
    failures: list[str] = []
    docker = root / "docker"
    eq = json.loads((docker / "backport-equivalence-v0.25.1.json").read_text())
    scale = json.loads((docker / "qwen27-w4a16-scale-audit.json").read_text())
    runtime = json.loads((docker / "runtime-manifest.json").read_text())
    pr_patch = (docker / "vllm-pr46329-v0.25.1.diff").read_text()
    native_patch = (docker / "native-w4a4-qwen27-v0.25.1.diff").read_text()
    dockerfile = (docker / "Dockerfile.kv-exp").read_text()

    if eq.get("release_base") != RELEASE:
        failures.append("equivalence release_base mismatch")
    if eq.get("upstream_pr_head") != PR_HEAD:
        failures.append("equivalence PR head mismatch")
    if len(eq.get("commits", [])) != 7:
        failures.append("equivalence map must account for 7 commits")
    if len(eq.get("files", [])) != 7:
        failures.append("equivalence map must account for 7 files")
    for item in eq.get("commits", []) + eq.get("files", []):
        if item.get("status") not in {"integrated", "superseded", "ported"}:
            failures.append(f"invalid equivalence status: {item!r}")
    if not eq.get("release_fixes_preserved"):
        failures.append("release_fixes_preserved is not true")
    if diff_paths(pr_patch) != EXPECTED_PATHS:
        failures.append(f"PR patch path mismatch: {sorted(diff_paths(pr_patch))}")
    if diff_paths(native_patch) != {
        "vllm/model_executor/layers/quantization/modelopt.py"
    }:
        failures.append("native reroute patch touches paths outside modelopt.py")
    if "R0B0TLAB_NATIVE_W4A4_FROM_W4A16" not in native_patch:
        failures.append("native reroute marker missing")
    if scale.get("status") != "PASS" or scale.get("targets") != 193:
        failures.append("checkpoint scale audit is not PASS for 193 targets")
    if scale.get("missing") or scale.get("nonfinite") or scale.get("nonpositive"):
        failures.append("checkpoint scale audit contains invalid scales")

    required_docker = [
        RELEASE,
        FLASHINFER,
        "cuda-toolkit-13-0",
        "/usr/bin/python3 -m venv /opt/vllm",
        "COPY --from=builder /opt/vllm/ /opt/vllm/",
        "MAX_JOBS=6",
        "NVCC_THREADS=2",
        "FLASHINFER_NVCC_THREADS=2",
        "ARG VLLM_TAG=v0.25.1",
        'refs/tags/${VLLM_TAG}:refs/tags/${VLLM_TAG}',
        "describe --tags --exact-match HEAD",
        "--no-build-isolation --no-deps .",
        "git apply --check /tmp/vllm-pr46329-v0.25.1.diff",
        "git apply --check /tmp/native-w4a4-qwen27-v0.25.1.diff",
        'org.opencontainers.image.source="https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4"',
    ]
    for marker in required_docker:
        if marker not in dockerfile:
            failures.append(f"Dockerfile marker missing: {marker}")
    if "FlashInfer build failed" in dockerfile or "pip install flashinfer-python" in dockerfile:
        failures.append("Dockerfile contains a silent FlashInfer fallback")
    if runtime.get("vllm_commit") != RELEASE:
        failures.append("runtime manifest vLLM commit mismatch")
    if runtime.get("vllm_tag") != "v0.25.1":
        failures.append("runtime manifest vLLM tag mismatch")
    if runtime.get("flashinfer_commit") != FLASHINFER:
        failures.append("runtime manifest FlashInfer commit mismatch")
    if runtime.get("model_revision") != MODEL:
        failures.append("runtime manifest model revision mismatch")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path(__file__).resolve().parents[1], type=Path)
    args = parser.parse_args()
    failures = validate(args.root)
    if failures:
        print("BACKPORT_VERIFY_FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("BACKPORT_VERIFY_PASS: 7 commits, 7 paths, exact release and artifact pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
