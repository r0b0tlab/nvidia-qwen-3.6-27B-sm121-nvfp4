#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "entrypoint.sh"
DOCKERFILE = ROOT / "docker" / "Dockerfile.production"
RECIPE = ROOT / "sparkrun" / "recipes" / "qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml"


def staged_entrypoint(tmp: Path, audit_exit: int = 0) -> tuple[Path, Path]:
    audit_marker = tmp / "audit-ran"
    fake_audit = tmp / "audit_runtime.py"
    fake_audit.write_text(
        "#!/usr/bin/env bash\n"
        f"printf ran > {audit_marker!s}\n"
        f"exit {audit_exit}\n"
    )
    fake_audit.chmod(fake_audit.stat().st_mode | stat.S_IEXEC)
    staged = tmp / "entrypoint.sh"
    staged.write_text(
        ENTRYPOINT.read_text().replace(
            "AUDIT_BIN=/usr/local/bin/audit_runtime.py",
            f"AUDIT_BIN={fake_audit}",
        )
    )
    staged.chmod(staged.stat().st_mode | stat.S_IEXEC)
    return staged, audit_marker


class LaunchContractTests(unittest.TestCase):
    def test_runtime_manifest_is_readable_by_sparkrun_uid(self) -> None:
        text = DOCKERFILE.read_text()
        self.assertIn("chmod 0644 /opt/r0b0tlab/runtime-manifest.json", text)

    def test_entrypoint_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(ENTRYPOINT)], check=True)

    def test_entrypoint_preserves_argv_after_audit(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('exec "$@"', text)
        self.assertNotIn("eval ", text)
        expected = ["alpha beta", "gamma", "delta epsilon"]
        with tempfile.TemporaryDirectory() as raw:
            staged, audit_marker = staged_entrypoint(Path(raw))
            result = subprocess.run(
                [
                    staged,
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
            self.assertTrue(audit_marker.exists())

    def test_entrypoint_returns_exact_child_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            staged, _ = staged_entrypoint(Path(raw))
            result = subprocess.run([staged, "/bin/bash", "-c", "exit 37"], check=False)
            self.assertEqual(result.returncode, 37)

    def test_failed_audit_blocks_explicit_child(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            staged, audit_marker = staged_entrypoint(tmp, audit_exit=23)
            child_marker = tmp / "child-ran"
            result = subprocess.run(
                [staged, "/bin/bash", "-c", f"printf ran > {child_marker}"],
                check=False,
            )
            self.assertEqual(result.returncode, 23)
            self.assertTrue(audit_marker.exists())
            self.assertFalse(child_marker.exists())

    def test_audit_is_a_dedicated_subcommand_and_launch_admission(self) -> None:
        text = ENTRYPOINT.read_text()
        audit_branch = 'if [[ "${1:-}" == "audit" ]]; then'
        explicit_branch = 'if (( $# > 0 )); then'
        self.assertIn(audit_branch, text)
        self.assertIn('exec "$AUDIT_BIN"', text)
        self.assertIn('"$AUDIT_BIN"', text)
        self.assertLess(text.index(audit_branch), text.index(explicit_branch))
        self.assertLess(text.index('"$AUDIT_BIN"', text.index(audit_branch) + len(audit_branch)), text.index(explicit_branch))

    def test_production_rejects_nvfp4_kv_before_audit(self) -> None:
        for argv, env_update in (
            (["/bin/true", "--kv-cache-dtype", "nvfp4"], {}),
            (["/bin/true", "--kv-cache-dtype=nvfp4"], {}),
            (["/bin/true"], {"KV_CACHE_DTYPE": "nvfp4"}),
            (["bash", "-c", "vllm serve /m --kv-cache-dtype nvfp4"], {}),
        ):
            with self.subTest(argv=argv, env_update=env_update), tempfile.TemporaryDirectory() as raw:
                staged, audit_marker = staged_entrypoint(Path(raw))
                env = os.environ.copy()
                env["R0B0TLAB_NVFP4_KV_ENABLED"] = "0"
                env.update(env_update)
                result = subprocess.run([staged, *argv], env=env, text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 64)
                self.assertIn("NVFP4 KV is disabled", result.stderr)
                self.assertFalse(audit_marker.exists())

    def test_zero_arg_defaults_are_fail_closed(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertIn('KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"', text)
        self.assertIn('MTP_TOKENS="${MTP_TOKENS:-2}"', text)
        self.assertIn('MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"', text)
        self.assertIn('MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"', text)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"', text)
        self.assertNotIn("VLLM_ATTENTION_BACKEND", text)
        self.assertNotIn("--enable-prefix-caching", text)
        self.assertIn("--attention-backend", text)
        self.assertIn("--language-model-only", text)
        self.assertIn("--max-num-batched-tokens", text)
        self.assertIn("--reasoning-parser", text)
        self.assertIn("--tool-call-parser", text)

    def test_recipe_is_immutable_and_command_driven(self) -> None:
        text = RECIPE.read_text()
        self.assertIn("recipe_version: \"2\"", text)
        self.assertIn("model_revision: 0893e1606ff3d5f97a441f405d5fc541a6bdf404", text)
        self.assertIn("container: ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:a5ff6d4bcca5b89ac10ee4525d9cba5ce0c9a17a7007313f10bd2e75c76af6e0", text)
        self.assertIn("vllm serve {model} --revision {model_revision}", text)
        self.assertIn("kv_dtype: fp8", text)
        self.assertIn("R0B0TLAB_NVFP4_KV_ENABLED: \"0\"", text)
        self.assertIn("--kv-cache-dtype fp8", text)
        self.assertIn("--attention-backend FLASHINFER", text)
        self.assertIn("max_model_len: 8192", text)
        self.assertIn("max_num_batched_tokens: 8192", text)
        self.assertIn("gpu_memory_utilization: 0.75", text)
        self.assertIn('"method":"mtp"', text)
        self.assertIn('"num_speculative_tokens":2', text)
        self.assertIn("--reasoning-parser qwen3", text)
        self.assertIn("--tool-call-parser qwen3_xml", text)
        self.assertIn("--speculative-config '{speculative_config}'", text)
        self.assertNotIn("--enable-prefix-caching", text)
        self.assertNotIn("VLLM_KV_CACHE_LAYOUT", text)

    def test_registry_manifest_points_to_recipe_tree(self) -> None:
        text = (ROOT / ".sparkrun" / "registry.yaml").read_text()
        self.assertIn("name: r0b0tlab", text)
        self.assertIn("recipes: sparkrun/recipes", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
