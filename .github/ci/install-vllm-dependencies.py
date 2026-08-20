#!/usr/bin/env python3
"""Install vLLM runtime dependencies that are required by LightOp CI."""

from __future__ import annotations

import argparse
from email.parser import Parser
import importlib.metadata
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


SKIPPED_DEPENDENCIES = {
    "llguidance",  # Only needed by vLLM structured-output support.
    "opencv-python-headless",  # Not used by CI; recent wheels require NumPy 2.
    "torch",  # The DTK build is selected explicitly by the matrix lane.
    "triton",  # The DTK build is selected explicitly by the matrix lane.
    "torchvision",  # Must match the selected DTK Torch build.
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-wheel", type=Path, required=True)
    parser.add_argument("--extra-index-url", required=True)
    parser.add_argument("--trusted-host", required=True)
    args = parser.parse_args()

    environment = default_environment()
    environment["extra"] = ""
    requirements: list[str] = []

    with zipfile.ZipFile(args.vllm_wheel) as wheel:
        metadata_files = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise SystemExit(
                f"Expected one METADATA file in {args.vllm_wheel}, "
                f"found {len(metadata_files)}."
            )
        metadata = Parser().parsestr(
            wheel.read(metadata_files[0]).decode("utf-8", errors="replace")
        )

    for raw_requirement in metadata.get_all("Requires-Dist", []):
        requirement = Requirement(raw_requirement)
        if canonicalize_name(requirement.name) in SKIPPED_DEPENDENCIES:
            continue
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        requirements.append(str(requirement))

    if not requirements:
        raise SystemExit("vLLM did not declare any installable runtime dependencies.")

    # Prevent transitive dependencies from replacing DTK packages that were
    # selected explicitly for this compatibility lane.
    constraints = "\n".join(
        f"{package}==={importlib.metadata.version(package)}"
        for package in ("numpy", "torch", "triton", "torchvision")
    )
    with tempfile.TemporaryDirectory(prefix="lightop-vllm-") as temp_dir:
        constraints_path = Path(temp_dir, "constraints.txt")
        constraints_path.write_text(f"{constraints}\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--constraint",
                str(constraints_path),
                "--extra-index-url",
                args.extra_index_url,
                "--trusted-host",
                args.trusted_host,
                *requirements,
            ],
            check=True,
        )

    print(
        "Skipped vLLM dependencies: llguidance, opencv-python-headless, "
        "torch, triton, torchvision"
    )


if __name__ == "__main__":
    main()
