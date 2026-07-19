#!/usr/bin/env python3
"""Strict dependency check with one verified NVIDIA SBSA wheel-tag exception."""

from __future__ import annotations

import importlib.metadata as metadata
from pathlib import Path
import platform
import subprocess
import sys
from typing import NamedTuple


KNOWN_SBSA_WARNING = "nvidia-cusparselt-cu13 0.8.0 is not supported on this platform"
KNOWN_SBSA_VERSION = "0.8.0"
KNOWN_SBSA_TAG = "Tag: py3-none-manylinux2014_sbsa"


class Evaluation(NamedTuple):
    ok: bool
    reason: str


def evaluate(
    *,
    pip_returncode: int,
    pip_output: str,
    machine: str,
    version: str | None = None,
    wheel_text: str = "",
    elf_header: str = "",
) -> Evaluation:
    lines = [line.strip() for line in pip_output.splitlines() if line.strip()]
    if pip_returncode == 0:
        return Evaluation(True, "pip check passed without exceptions")
    if lines != [KNOWN_SBSA_WARNING]:
        return Evaluation(False, f"unexpected pip check output: {lines!r}")
    if machine != "aarch64":
        return Evaluation(False, f"SBSA exception requires aarch64, got {machine!r}")
    if version != KNOWN_SBSA_VERSION:
        return Evaluation(False, f"SBSA exception version mismatch: {version!r}")
    if KNOWN_SBSA_TAG not in wheel_text:
        return Evaluation(False, "SBSA wheel tag is absent")
    if "Machine:" not in elf_header or "AArch64" not in elf_header:
        return Evaluation(False, "cuSPARSELt library is not an AArch64 ELF")
    return Evaluation(
        True,
        "accepted exact NVIDIA cuSPARSELt 0.8.0 SBSA tag warning after AArch64 ELF verification",
    )


def main() -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())

    if result.returncode == 0:
        evaluation = evaluate(
            pip_returncode=0,
            pip_output=output,
            machine=platform.machine(),
        )
    else:
        try:
            dist = metadata.distribution("nvidia-cusparselt-cu13")
            wheel_path = Path(dist._path) / "WHEEL"  # type: ignore[attr-defined]
            library = Path(dist.locate_file("nvidia/cusparselt/lib/libcusparseLt.so.0"))
            wheel_text = wheel_path.read_text()
            elf_header = subprocess.check_output(
                ["readelf", "-h", str(library)], text=True, timeout=30
            )
            evaluation = evaluate(
                pip_returncode=result.returncode,
                pip_output=output,
                machine=platform.machine(),
                version=dist.version,
                wheel_text=wheel_text,
                elf_header=elf_header,
            )
        except Exception as exc:
            evaluation = Evaluation(False, f"SBSA verification failed: {exc!r}")

    if output:
        print("PIP_CHECK_OUTPUT=" + output.replace("\n", " | "))
    print(
        ("DEPENDENCY_CHECK_PASS: " if evaluation.ok else "DEPENDENCY_CHECK_FAIL: ")
        + evaluation.reason
    )
    return 0 if evaluation.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
