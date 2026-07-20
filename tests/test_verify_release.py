#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_release", ROOT / "scripts" / "verify_release.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

DIGEST = "sha256:" + "a" * 64
IMAGE_ID = "sha256:" + "b" * 64
IMAGE = "ghcr.io/r0b0tlab/sm121-vllm-nvfp4"
MODEL_REV = "0893e1606ff3d5f97a441f405d5fc541a6bdf404"
IMMUTABLE = f"{IMAGE}@{DIGEST}"


def valid_summary() -> dict:
    rows = []
    for concurrency in (1, 2, 4, 8, 16, 32):
        rows.append(
            {
                "concurrency": concurrency,
                "requests_ok": concurrency * 3,
                "requests_total": concurrency * 3,
                "output_tokens_per_second": 10.0 * concurrency,
                "prompt_tokens_per_second": 4.0 * concurrency,
                "ttft_p50_seconds": 0.1,
                "ttft_p90_seconds": 0.2,
                "ttft_p99_seconds": 0.3,
                "itl_p50_seconds": 0.08,
                "itl_p90_seconds": 0.09,
                "itl_p99_seconds": 0.1,
                "power_mean_watts": 40.0,
                "mtp_drafted": 100.0,
                "mtp_accepted": 90.0,
                "mtp_acceptance": 0.9,
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS",
        "release": {
            "model_id": "nvidia/Qwen3.6-27B-NVFP4",
            "model_revision": MODEL_REV,
            "vllm": "0.25.1",
            "vllm_package": "0.25.1+r0b0tlab.w4a4.1",
            "torch": "2.11.0+cu130",
            "cuda": "13.0",
            "flashinfer": "0.6.13",
            "local_image_id": IMAGE_ID,
            "ghcr_image": IMAGE,
            "ghcr_digest": DIGEST,
        },
        "profile": {
            "kv_cache_dtype": "fp8",
            "mtp": {"method": "mtp", "num_speculative_tokens": 2},
            "configured_context_tokens": 8192,
            "max_num_seqs": 32,
        },
        "native_gate": {
            "runtime_audit": "PASS",
            "marlin_markers": 0,
            "emulation_markers": 0,
            "fallback_markers": 0,
            "native_w4a4_marker": True,
            "flashinfer_attention_marker": True,
        },
        "canaries": {"models": True, "semantic": True, "tool_call": True, "long_generation": True},
        "performance": {"errors": [], "rows": rows},
        "quality": {
            "task": "gsm8k_qwen_chat",
            "dataset": "openai/gsm8k",
            "num_fewshot": 0,
            "sample_count": 1319,
            "endpoint": "chat-completions",
            "enable_thinking": False,
            "request_errors": 0,
            "flexible_extract_exact_match": 0.85,
        },
        "nvfp4_kv": {"status": "NOT_PROMOTED"},
    }


class ReleaseVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.summary = self.root / "summary.json"
        self.recipe = self.root / "recipe.yaml"
        self.readme = self.root / "README.md"
        self.html = self.root / "benchmark.html"
        self.manifest = self.root / "MANIFEST.sha256"
        self.data = valid_summary()
        self._write_fixture()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_fixture(self) -> None:
        self.summary.write_text(json.dumps(self.data, indent=2) + "\n")
        public_text = (
            f"{IMMUTABLE}\n{MODEL_REV}\n0.25.1+r0b0tlab.w4a4.1\n"
            "NVFP4 KV was not promoted.\n"
        )
        self.recipe.write_text(f"container: {IMMUTABLE}\n")
        self.readme.write_text(public_text)
        self.html.write_text("<html><body>" + public_text + "</body></html>")
        entries = []
        for path in (self.summary, self.recipe, self.readme, self.html):
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        self.manifest.write_text("\n".join(entries) + "\n")

    def _errors(self) -> list[str]:
        return MODULE.verify(self.summary, self.recipe, self.readme, self.html, self.manifest)

    def test_valid_bundle_passes(self) -> None:
        self.assertEqual(self._errors(), [])

    def test_active_marlin_is_rejected(self) -> None:
        self.data["native_gate"]["marlin_markers"] = 1
        self._write_fixture()
        self.assertTrue(any("marlin_markers" in error for error in self._errors()))

    def test_failed_request_is_rejected(self) -> None:
        self.data["performance"]["rows"][0]["requests_ok"] = 2
        self._write_fixture()
        self.assertTrue(any("failed or missing requests" in error for error in self._errors()))

    def test_mutable_recipe_is_rejected(self) -> None:
        self.recipe.write_text(f"container: {IMAGE}:v0.25.1-production\n")
        entries = []
        for path in (self.summary, self.recipe, self.readme, self.html):
            entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
        self.manifest.write_text("\n".join(entries) + "\n")
        self.assertTrue(any("immutable GHCR digest" in error for error in self._errors()))

    def test_missing_latency_metric_is_rejected(self) -> None:
        del self.data["performance"]["rows"][2]["ttft_p90_seconds"]
        self._write_fixture()
        self.assertTrue(any("ttft_p90_seconds" in error for error in self._errors()))

    def test_incomplete_quality_set_is_rejected(self) -> None:
        self.data["quality"]["sample_count"] = 1318
        self._write_fixture()
        self.assertTrue(any("quality.sample_count" in error for error in self._errors()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
