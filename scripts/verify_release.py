#!/usr/bin/env python3
"""Fail-closed verifier for the public v0.25.1 release evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

EXPECTED = {
    "model_id": "nvidia/Qwen3.6-27B-NVFP4",
    "model_revision": "0893e1606ff3d5f97a441f405d5fc541a6bdf404",
    "vllm": "0.25.1",
    "vllm_package": "0.25.1+r0b0tlab.w4a4.1",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "flashinfer": "0.6.13",
}
EXPECTED_CONCURRENCY = [1, 2, 4, 8, 16, 32]
FORBIDDEN_PUBLIC_TEXT = (
    "/home/",
    "192.168.",
    "r0b0tdgx",
    "r0b0t-dgx",
    "GITHUB_TOKEN",
    "HF_" + "TOKEN=hf_",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path, manifest: Path, errors: list[str]) -> None:
    if not manifest.is_file():
        errors.append(f"missing manifest: {manifest}")
        return
    seen: set[str] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw)
        if not match:
            errors.append(f"manifest line {line_number} is malformed")
            continue
        expected_hash, relative = match.groups()
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            errors.append(f"manifest path escapes result root: {relative}")
            continue
        if relative in seen:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        seen.add(relative)
        if not candidate.is_file():
            errors.append(f"manifest file is missing: {relative}")
        elif _sha256(candidate) != expected_hash:
            errors.append(f"manifest hash mismatch: {relative}")
    if not seen:
        errors.append("manifest is empty")


def verify(
    summary_path: Path,
    recipe_path: Path,
    readme_path: Path,
    html_path: Path,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    try:
        summary = _load_json(summary_path)
    except Exception as exc:
        return [f"cannot read summary: {exc}"]

    if summary.get("schema_version") != 1:
        errors.append("summary schema_version must be 1")
    if summary.get("status") != "PASS":
        errors.append("summary status must be PASS")

    release = summary.get("release") or {}
    for key, expected in EXPECTED.items():
        if release.get(key) != expected:
            errors.append(f"release.{key} must equal {expected!r}")
    local_image_id = release.get("local_image_id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(local_image_id)):
        errors.append("release.local_image_id must be an immutable sha256 ID")
    ghcr_digest = release.get("ghcr_digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(ghcr_digest)):
        errors.append("release.ghcr_digest must be an immutable sha256 digest")
    ghcr_image = release.get("ghcr_image", "")
    if ghcr_image != "ghcr.io/r0b0tlab/sm121-vllm-nvfp4":
        errors.append("release.ghcr_image is unexpected")

    profile = summary.get("profile") or {}
    if profile.get("kv_cache_dtype") != "fp8":
        errors.append("production KV cache must be fp8")
    if profile.get("mtp") != {"method": "mtp", "num_speculative_tokens": 2}:
        errors.append("production MTP profile must be mtp K=2")
    if profile.get("configured_context_tokens") != 8192:
        errors.append("qualified context must be exactly 8192 tokens")
    if profile.get("max_num_seqs") != 32:
        errors.append("qualified max_num_seqs must be 32")

    native = summary.get("native_gate") or {}
    if native.get("runtime_audit") != "PASS":
        errors.append("runtime audit did not pass")
    for marker in ("marlin_markers", "emulation_markers", "fallback_markers"):
        if native.get(marker) != 0:
            errors.append(f"native_gate.{marker} must be zero")
    if not native.get("native_w4a4_marker"):
        errors.append("native W4A4 positive marker is absent")
    if not native.get("flashinfer_attention_marker"):
        errors.append("FlashInfer attention positive marker is absent")

    canaries = summary.get("canaries") or {}
    for name in ("models", "semantic", "tool_call", "long_generation"):
        if canaries.get(name) is not True:
            errors.append(f"canary {name} did not pass")

    performance = summary.get("performance") or {}
    if performance.get("errors") not in ([], None):
        errors.append("performance evidence contains errors")
    rows = performance.get("rows") or []
    levels = [row.get("concurrency") for row in rows if isinstance(row, dict)]
    if levels != EXPECTED_CONCURRENCY:
        errors.append(f"performance concurrency levels must be {EXPECTED_CONCURRENCY}")
    required_metrics = (
        "output_tokens_per_second",
        "prompt_tokens_per_second",
        "ttft_p50_seconds",
        "ttft_p90_seconds",
        "ttft_p99_seconds",
        "itl_p50_seconds",
        "itl_p90_seconds",
        "itl_p99_seconds",
        "power_mean_watts",
        "mtp_acceptance",
    )
    for row in rows:
        if not isinstance(row, dict):
            errors.append("performance row is not an object")
            continue
        if row.get("requests_ok") != row.get("requests_total") or not row.get("requests_total"):
            errors.append(f"concurrency {row.get('concurrency')} has failed or missing requests")
        for metric in required_metrics:
            value = row.get(metric)
            if not isinstance(value, (int, float)):
                errors.append(f"concurrency {row.get('concurrency')} lacks numeric {metric}")
        accepted = row.get("mtp_accepted")
        drafted = row.get("mtp_drafted")
        if not isinstance(accepted, (int, float)) or not isinstance(drafted, (int, float)):
            errors.append(f"concurrency {row.get('concurrency')} lacks exact MTP counters")
        elif not (0 < accepted <= drafted):
            errors.append(f"concurrency {row.get('concurrency')} has invalid MTP counters")

    quality = summary.get("quality") or {}
    expected_quality = {
        "task": "gsm8k_qwen_chat",
        "dataset": "openai/gsm8k",
        "num_fewshot": 0,
        "sample_count": 1319,
        "endpoint": "chat-completions",
        "enable_thinking": False,
        "request_errors": 0,
    }
    for key, expected in expected_quality.items():
        if quality.get(key) != expected:
            errors.append(f"quality.{key} must equal {expected!r}")
    score = quality.get("flexible_extract_exact_match")
    if not isinstance(score, (int, float)) or not 0 <= score <= 1:
        errors.append("quality score is missing or outside [0, 1]")

    nvfp4 = summary.get("nvfp4_kv") or {}
    if nvfp4.get("status") != "NOT_PROMOTED":
        errors.append("NVFP4 KV must remain NOT_PROMOTED")

    public_paths = (summary_path, recipe_path, readme_path, html_path)
    for path in public_paths:
        if not path.is_file():
            errors.append(f"missing public artifact: {path}")
            continue
        text = path.read_text(errors="replace")
        for forbidden in FORBIDDEN_PUBLIC_TEXT:
            if forbidden.lower() in text.lower():
                errors.append(f"forbidden public text {forbidden!r} in {path}")

    immutable_ref = f"{ghcr_image}@{ghcr_digest}"
    recipe_text = recipe_path.read_text(errors="replace") if recipe_path.is_file() else ""
    readme_text = readme_path.read_text(errors="replace") if readme_path.is_file() else ""
    html_text = html_path.read_text(errors="replace") if html_path.is_file() else ""
    if immutable_ref not in recipe_text:
        errors.append("recipe does not use the immutable GHCR digest")
    for label, text in (("README", readme_text), ("HTML", html_text)):
        if immutable_ref not in text:
            errors.append(f"{label} omits the immutable GHCR reference")
        if EXPECTED["model_revision"] not in text:
            errors.append(f"{label} omits the model revision")
        if "0.25.1+r0b0tlab.w4a4.1" not in text:
            errors.append(f"{label} omits the derivative vLLM package version")
        if "NVFP4 KV" not in text or "not promoted" not in text.lower():
            errors.append(f"{label} omits the NVFP4-KV claim boundary")

    _verify_manifest(summary_path.parent, manifest_path, errors)
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/v0251-node2/summary.json"))
    parser.add_argument("--recipe", type=Path, default=Path("sparkrun/recipes/qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--html", type=Path, default=Path("results/v0251-node2/benchmark.html"))
    parser.add_argument("--manifest", type=Path, default=Path("results/v0251-node2/MANIFEST.sha256"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = verify(args.summary, args.recipe, args.readme, args.html, args.manifest)
    if errors:
        for error in errors:
            print(f"RELEASE_VERIFY_FAIL: {error}")
        return 1
    print("RELEASE_VERIFY_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
