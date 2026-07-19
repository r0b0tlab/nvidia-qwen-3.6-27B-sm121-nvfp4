#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "entrypoint.sh"
RECIPE = ROOT / "sparkrun" / "recipes" / "qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml"


class LaunchContractTests(unittest.TestCase):
    def test_entrypoint_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)

    def test_entrypoint_preserves_argv(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('exec "$@"', text)
        self.assertNotIn("eval ", text)

        expected = ["alpha beta", "gamma", "delta epsilon"]
        result = subprocess.run(
            [
                str(ENTRYPOINT),
                sys.executable,
                "-c",
                "import json,sys; print(json.dumps(sys.argv[1:]))",
                *expected,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout), expected)

    def test_entrypoint_returns_exact_child_exit_code(self) -> None:
        result = subprocess.run(
            [str(ENTRYPOINT), "/bin/bash", "-c", "exit 37"],
            check=False,
        )
        self.assertEqual(result.returncode, 37)

    def test_audit_is_a_dedicated_subcommand(self) -> None:
        text = ENTRYPOINT.read_text()
        audit_branch = 'if [[ "${1:-}" == "audit" ]]; then'
        self.assertIn(audit_branch, text)
        self.assertIn('exec /usr/local/bin/audit_runtime.py', text)
        self.assertLess(text.index(audit_branch), text.index('if (( $# > 0 )); then'))

    def test_zero_arg_defaults_are_fail_closed(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"', text)
        self.assertIn('MTP_TOKENS="${MTP_TOKENS:-1}"', text)
        self.assertIn('VLLM_ATTENTION_BACKEND:-FLASHINFER', text)
        self.assertNotIn("--enable-prefix-caching", text)
        self.assertIn("--language-model-only", text)

    def test_recipe_is_immutable_and_command_driven(self) -> None:
        text = RECIPE.read_text()
        self.assertIn("recipe_version: \"2\"", text)
        self.assertIn("model_revision: 0893e1606ff3d5f97a441f405d5fc541a6bdf404", text)
        self.assertIn("container: ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-kv-exp", text)
        self.assertIn("vllm serve {model}", text)
        self.assertIn("kv_dtype: fp8", text)
        self.assertIn("--kv-cache-dtype fp8", text)
        self.assertIn('"num_speculative_tokens":1', text)
        self.assertIn("--speculative-config '{speculative_config}'", text)
        self.assertNotIn("--enable-prefix-caching", text)
        self.assertNotIn("VLLM_KV_CACHE_LAYOUT", text)

    def test_registry_manifest_points_to_recipe_tree(self) -> None:
        text = (ROOT / ".sparkrun" / "registry.yaml").read_text()
        self.assertIn("name: r0b0tlab", text)
        self.assertIn("recipes: sparkrun/recipes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
