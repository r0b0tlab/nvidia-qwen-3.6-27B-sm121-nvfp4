#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_dependencies import KNOWN_SBSA_WARNING, evaluate  # noqa: E402


class DependencyCheckTests(unittest.TestCase):
    def test_clean_pip_check_passes(self) -> None:
        result = evaluate(pip_returncode=0, pip_output="", machine="aarch64")
        self.assertTrue(result.ok)

    def test_exact_verified_sbsa_warning_passes(self) -> None:
        result = evaluate(
            pip_returncode=1,
            pip_output=KNOWN_SBSA_WARNING + "\n",
            machine="aarch64",
            version="0.8.0",
            wheel_text="Tag: py3-none-manylinux2014_sbsa\n",
            elf_header="Machine: AArch64\n",
        )
        self.assertTrue(result.ok)

    def test_additional_dependency_error_fails(self) -> None:
        result = evaluate(
            pip_returncode=1,
            pip_output=KNOWN_SBSA_WARNING + "\nother-package has requirement conflict\n",
            machine="aarch64",
            version="0.8.0",
            wheel_text="Tag: py3-none-manylinux2014_sbsa\n",
            elf_header="Machine: AArch64\n",
        )
        self.assertFalse(result.ok)

    def test_wrong_architecture_fails(self) -> None:
        result = evaluate(
            pip_returncode=1,
            pip_output=KNOWN_SBSA_WARNING,
            machine="x86_64",
            version="0.8.0",
            wheel_text="Tag: py3-none-manylinux2014_sbsa\n",
            elf_header="Machine: AArch64\n",
        )
        self.assertFalse(result.ok)

    def test_wrong_elf_fails(self) -> None:
        result = evaluate(
            pip_returncode=1,
            pip_output=KNOWN_SBSA_WARNING,
            machine="aarch64",
            version="0.8.0",
            wheel_text="Tag: py3-none-manylinux2014_sbsa\n",
            elf_header="Machine: Advanced Micro Devices X86-64\n",
        )
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
