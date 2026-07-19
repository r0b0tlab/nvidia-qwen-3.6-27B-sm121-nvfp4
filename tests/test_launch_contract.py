#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "entrypoint.sh"
RECIPE = ROOT / "sparkrun" / "recipes" / "qwen3.6-27b-nvfp4-kv-vllm-r0b0tlab.yaml"


class LaunchContractTests(unittest.TestCase):
    def test_entrypoint_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)

    def test_entrypoint_preserves_argv(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('exec "$@"', text)
        self.assertNotIn("eval ", text)
        self.assertIn("/usr/local/bin/audit_runtime.py", text)

    def test_zero_arg_defaults_are_fail_closed(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-nvfp4}"', text)
        self.assertIn('VLLM_ATTENTION_BACKEND:-FLASHINFER', text)
        self.assertIn('VLLM_KV_CACHE_LAYOUT:-HND', text)
        self.assertIn("--language-model-only", text)

    def test_recipe_is_immutable_and_command_driven(self) -> None:
        text = RECIPE.read_text()
        self.assertIn("recipe_version: \"2\"", text)
        self.assertIn("model_revision: 0893e1606ff3d5f97a441f405d5fc541a6bdf404", text)
        self.assertIn("container: ghcr.io/r0b0tlab/sm121-vllm-nvfp4:v0.25.1-kv-exp", text)
        self.assertIn("vllm serve {model}", text)
        self.assertIn("--kv-cache-dtype nvfp4", text)
        self.assertIn("--speculative-config '{speculative_config}'", text)

    def test_registry_manifest_points_to_recipe_tree(self) -> None:
        text = (ROOT / ".sparkrun" / "registry.yaml").read_text()
        self.assertIn("name: r0b0tlab", text)
        self.assertIn("recipes: sparkrun/recipes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
