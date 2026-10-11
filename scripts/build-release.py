"""Serialize standalone release CWL with mandatory immutable images."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def build(version, digests, output):
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", version):
        raise ValueError("Invalid release version")
    output.mkdir(parents=True, exist_ok=True)
    inventory = []
    for kind in ["training", "inference"]:
        image = (digests / f"{kind}.txt").read_text().strip()
        if not re.fullmatch(r"ghcr\.io/[^\s]+@sha256:[a-f0-9]{64}", image):
            raise ValueError(f"Invalid immutable image for {kind}")
        document = yaml.safe_load((ROOT / f"{kind}/app-package/tile-sat-{kind}.cwl").read_text())
        document["s:softwareVersion"] = version
        for process in document["$graph"]:
            if process["class"] == "CommandLineTool":
                process["requirements"]["DockerRequirement"] = {"dockerPull": image}
        filename = f"tile-sat-{kind}.{version}.cwl"
        target = output / filename
        target.write_text(yaml.safe_dump(document, sort_keys=False))
        subprocess.run(["cwltool", "--validate", str(target) + "#main"], check=True)
        inventory.append(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {filename}")
        annotations = {
            "$manifest": {
                "org.opencontainers.image.title": filename,
                "org.opencontainers.image.version": version,
                "org.opencontainers.image.source": "https://github.com/eoap/machine-learning-process",
                "org.opencontainers.image.revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "org.opencontainers.image.licenses": "Apache-2.0",
                "org.cwl.entrypoint": filename + "#main",
                "org.cwl.spec": "v1.2",
                "org.cwl.type": "Workflow",
                "org.opencontainers.image.description": document["$graph"][0]["doc"],
            }
        }
        (output / f"{kind}.annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    (output / "SHA256SUMS").write_text("\n".join(inventory) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--digests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.version, args.digests, args.output)
