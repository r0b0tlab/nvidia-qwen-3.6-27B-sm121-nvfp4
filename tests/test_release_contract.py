#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_backport import validate  # noqa: E402


class ReleaseContractTests(unittest.TestCase):
    def test_repository_contract_passes(self) -> None:
        self.assertEqual(validate(ROOT), [])

    def test_experimental_release_identity_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns(".git"))
            p = clone / "docker" / "runtime-manifest.json"
            data = json.loads(p.read_text())
            data["vllm_commit"] = "0" * 40
            p.write_text(json.dumps(data))
            failures = validate(clone)
            self.assertTrue(any("experimental runtime manifest vLLM commit mismatch" in x for x in failures))

    def test_production_profile_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns(".git"))
            p = clone / "docker" / "runtime-manifest.production.json"
            data = json.loads(p.read_text())
            data["nvfp4_kv_enabled"] = True
            p.write_text(json.dumps(data))
            failures = validate(clone)
            self.assertTrue(any("must disable NVFP4 KV" in x for x in failures))

    def test_patch_scope_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns(".git"))
            p = clone / "docker" / "native-w4a4-qwen27-v0.25.1.diff"
            p.write_text(
                p.read_text()
                + "\ndiff --git a/setup.py b/setup.py\n"
                + "--- a/setup.py\n+++ b/setup.py\n@@ -1 +1 @@\n-a\n+b\n"
            )
            failures = validate(clone)
            self.assertTrue(any("outside modelopt.py" in x for x in failures))

    def test_dockerfiles_fetch_and_verify_exact_release_tag(self) -> None:
        for name in ("Dockerfile.production", "Dockerfile.kv-exp"):
            with self.subTest(name=name):
                text = (ROOT / "docker" / name).read_text()
                self.assertIn("ARG VLLM_TAG=v0.25.1", text)
                self.assertIn('refs/tags/${VLLM_TAG}:refs/tags/${VLLM_TAG}', text)
                self.assertIn("describe --tags --exact-match HEAD", text)
                self.assertIn('io.r0b0tlab.upstream.vllm.tag="${VLLM_TAG}"', text)
                install = "RUN python3 -m pip install --no-build-isolation --no-deps ."
                audit = "RUN python3 - <<'PY'"
                self.assertLess(text.index(install), text.index(audit))
                self.assertNotIn(install + " \\\n    && python3", text)

    def test_production_excludes_experimental_stack(self) -> None:
        text = (ROOT / "docker" / "Dockerfile.production").read_text()
        self.assertNotIn("741b63720bb345d9036d38b33a7b5a043d4c2674", text)
        self.assertNotIn("vllm-pr46329-v0.25.1.diff", text)
        self.assertIn('md.version("flashinfer-python") == "0.6.13"', text)
        runtime = json.loads((ROOT / "docker" / "runtime-manifest.production.json").read_text())
        self.assertEqual(runtime["profile"], "production-fp8")
        self.assertEqual(runtime["vllm_package_version"], "0.25.1+r0b0tlab.w4a4.1")
        self.assertEqual(runtime["vllm_derivative_tag"], "v0.25.1+r0b0tlab.w4a4.1")
        self.assertFalse(runtime["nvfp4_kv_enabled"])
        self.assertEqual(runtime["default_kv_cache_dtype"], "fp8")


if __name__ == "__main__":
    unittest.main(verbosity=2)
