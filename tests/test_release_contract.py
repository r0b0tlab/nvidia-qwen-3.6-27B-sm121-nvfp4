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

    def test_release_identity_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns(".git"))
            p = clone / "docker" / "runtime-manifest.json"
            data = json.loads(p.read_text())
            data["vllm_commit"] = "0" * 40
            p.write_text(json.dumps(data))
            failures = validate(clone)
            self.assertTrue(any("vLLM commit mismatch" in x for x in failures))

    def test_exact_release_version_override_is_pinned(self) -> None:
        dockerfile = (ROOT / "docker" / "Dockerfile.kv-exp").read_text()
        self.assertIn("VLLM_VERSION_OVERRIDE=0.25.1", dockerfile)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
