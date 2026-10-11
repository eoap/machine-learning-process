"""Require real-data candidate evidence before stable publication."""

import argparse
import json
import re
import subprocess
from pathlib import Path


def _check_execution_evidence(evidence: dict[str, object]) -> None:
    """Require the real-data execution flags and all nine successful notebooks."""
    for field in [
        "real_data",
        "editor_s3",
        "mlflow_artifacts",
        "lifecycle",
        "model_equivalence",
        "scientific_outputs",
    ]:
        if evidence.get(field) is not True:
            raise ValueError(f"Missing acceptance: {field}")
    if (
        evidence.get("runner") != "calrissian"
        or not evidence.get("cluster")
        or evidence.get("architecture") != "amd64"
    ):
        raise ValueError("Require identified AMD64 Kubernetes/Calrissian acceptance")
    notebooks = evidence.get("notebooks", [])
    expected = {
        f"practice-labs/{scenario}/{lesson}.ipynb"
        for scenario in [
            "Alternative1-Application_Steps",
            "Alternative2-Containers",
            "Alternative3-CWL-Workflows",
        ]
        for lesson in ["Step1A-training", "Step1B-ExtractModel", "Step2-inference"]
    }
    if (
        len(notebooks) != 9
        or {result.get("path") for result in notebooks} != expected
        or any(result.get("status") != "passed" for result in evidence["notebooks"])
    ):
        raise ValueError("All nine notebooks must pass")


def check(path: Path) -> None:
    """Require a complete acceptance record for the exact release candidate.

    Args:
        path: JSON acceptance record.

    Raises:
        ValueError: Evidence is incomplete or differs from the release source.
    """
    evidence = json.loads(path.read_text())
    if evidence.get("dirty_source") is not False:
        raise ValueError("Acceptance requires a clean source checkout")
    if not re.fullmatch(r"\d+\.\d+\.\d+-rc\.[0-9]+\.[0-9]+", evidence["candidate"]):
        raise ValueError("Expected a published candidate version")
    source = evidence["source_commit"]
    if not re.fullmatch("[a-f0-9]{40}", source):
        raise ValueError("Expected full accepted source commit")
    version = json.loads((Path(__file__).resolve().parents[1] / "codemeta.json").read_text())[
        "version"
    ]
    if not evidence["candidate"].startswith(version + "-rc."):
        raise ValueError("Candidate does not match the release version")
    tagged_source = subprocess.check_output(
        ["git", "rev-parse", evidence["candidate"] + "^{commit}"], text=True
    ).strip()
    if tagged_source != source:
        raise ValueError("Accepted source must match the published candidate tag")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", source, "HEAD"], text=True
    ).splitlines()
    if any(not file.startswith("docs/acceptance/") for file in changed):
        raise ValueError("Stable implementation differs from accepted candidate source")
    _check_execution_evidence(evidence)
    for kind in ["training", "inference"]:
        if not re.fullmatch(r"ghcr\.io/[^\s]+@sha256:[a-f0-9]{64}", evidence["images"][kind]):
            raise ValueError("Expected immutable accepted image references")
    if not evidence.get("evidence_location"):
        raise ValueError("Retained evidence location required")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    check(parser.parse_args().record)
