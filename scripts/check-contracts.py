"""Validate isolated CWL, embedded types, metadata and generated reference pages."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    ROOT / "training/app-package/tile-sat-training.cwl",
    ROOT / "inference/app-package/tile-sat-inference.cwl",
]


def _inspect_dependencies(value: object) -> None:
    """Reject external dependencies that prevent standalone CWL execution."""
    if isinstance(value, dict):
        assert "$include" not in value, "External includes are not standalone"
        if "$import" in value:
            assert value["$import"] in {
                "https://raw.githubusercontent.com/eoap/schemas/main/ogc.yaml",
                "https://raw.githubusercontent.com/eoap/schemas/main/string_format.yaml",
            }
        for child in value.values():
            _inspect_dependencies(child)
    elif isinstance(value, list):
        for child in value:
            _inspect_dependencies(child)


def check(path: Path, output: Path) -> None:
    """Validate a standalone CWL package and write its port documentation.

    Args:
        path: Source CWL package.
        output: Directory receiving the generated reference page.
    """
    document = yaml.safe_load(path.read_text())
    assert document["cwlVersion"] == "v1.2"
    for key in ["s:softwareVersion", "s:codeRepository", "s:license", "s:author"]:
        assert document.get(key), f"{path}: missing {key}"
    assert document["s:codeRepository"] == "https://github.com/eoap/machine-learning-process"
    assert (
        document["s:softwareVersion"] == json.loads((ROOT / "codemeta.json").read_text())["version"]
    )
    assert any(p["id"] == "main" for p in document["$graph"])

    _inspect_dependencies(document)
    lines = [
        f"# {path.stem}",
        "",
        f"Version: {document['s:softwareVersion']}",
        "",
        f"License: {document['s:license']}",
        "",
    ]
    for process in document["$graph"]:
        assert process.get("doc") and process.get("label", process.get("id"))
        lines += [f"## {process['id']}", "", process["doc"], ""]
        for group in ["inputs", "outputs"]:
            lines += [f"### {group}", "", "| Port | Type | Description |", "| --- | --- | --- |"]
            for name, port in process[group].items():
                assert port.get("label") and port.get("doc"), f"{name}: undocumented port"
                description = port["doc"].replace("\n", " ").replace("|", "\\|")
                lines += [f"| {name} | `{port['type']}` | {description} |"]
            lines += [""]
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / path.name
        target.write_bytes(path.read_bytes())
        subprocess.run(["cwltool", "--validate", str(target) + "#main"], check=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / (path.stem + ".md")).write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/reference")
    args = parser.parse_args()
    for package in PACKAGES:
        check(package, args.output)
