"""Download a selected release without modifying repository CWL."""

import argparse
import hashlib
import tempfile
from pathlib import Path

import requests
import yaml


def validate(path, version):
    document = yaml.safe_load(path.read_text())
    if document.get("cwlVersion") != "v1.2" or document.get("s:softwareVersion") != version:
        raise ValueError("Release CWL version/metadata mismatch")
    if not any(p.get("id") == "main" for p in document.get("$graph", [])):
        raise ValueError("Release CWL has no #main entrypoint")
    for process in document["$graph"]:
        if process["class"] == "CommandLineTool":
            image = (
                process.get("requirements", {}).get("DockerRequirement", {}).get("dockerPull", "")
            )
            if "@sha256:" not in image or len(image.rsplit("@sha256:", 1)[-1]) != 64:
                raise ValueError("Release requires immutable container references")


def prepare(version, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    base = f"https://github.com/eoap/machine-learning-process/releases/download/{version}"
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        response = requests.get(base + "/SHA256SUMS", timeout=60)
        response.raise_for_status()
        checksums = {line.split()[1]: line.split()[0] for line in response.text.splitlines()}
        for kind in ["training", "inference"]:
            name = f"tile-sat-{kind}.{version}.cwl"
            response = requests.get(base + "/" + name, timeout=60)
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != checksums.get(name):
                raise ValueError(f"Checksum mismatch for {name}")
            path = scratch / name
            path.write_bytes(response.content)
            validate(path, version)
        for path in scratch.iterdir():
            path.replace(output / path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.version, args.output)
