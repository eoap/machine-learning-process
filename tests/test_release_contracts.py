import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_rejects_mutable_images(tmp_path):
    release = load_script("build-release")
    (tmp_path / "training.txt").write_text("ghcr.io/eoap/machine-learning-process/training:latest")
    with pytest.raises(ValueError, match="immutable"):
        release.build("0.1.0", tmp_path, tmp_path / "output")


def test_release_is_standalone_and_keeps_sources(tmp_path):
    release = load_script("build-release")
    preparation = load_script("prepare-release")
    original = (ROOT / "training/app-package/tile-sat-training.cwl").read_bytes()
    for kind in ["training", "inference"]:
        (tmp_path / (kind + ".txt")).write_text(
            f"ghcr.io/eoap/machine-learning-process/{kind}@sha256:" + "a" * 64
        )
    release.build("0.1.0-rc.1.1", tmp_path, tmp_path / "output")
    for path in (tmp_path / "output").glob("*.cwl"):
        preparation.validate(path, "0.1.0-rc.1.1")
        document = yaml.safe_load(path.read_text())
        assert document["$graph"][0]["requirements"]["SchemaDefRequirement"]["types"]
    assert (ROOT / "training/app-package/tile-sat-training.cwl").read_bytes() == original
    assert (tmp_path / "output/SHA256SUMS").exists()


def test_download_rejects_wrong_release_metadata(tmp_path):
    path = tmp_path / "package.cwl"
    path.write_bytes((ROOT / "inference/app-package/tile-sat-inference.cwl").read_bytes())
    with pytest.raises(ValueError, match="mismatch"):
        load_script("prepare-release").validate(path, "9.9.9")


def test_stable_acceptance_rejects_dirty_or_duplicate_notebooks(tmp_path, monkeypatch):
    gate = load_script("check-acceptance")
    monkeypatch.setattr(
        gate.subprocess,
        "check_output",
        lambda args, **kwargs: "a" * 40 if "rev-parse" in args else "",
    )
    evidence = {
        "candidate": "0.1.0-rc.1.1",
        "source_commit": "a" * 40,
        "dirty_source": True,
        "runner": "calrissian",
        "cluster": "acceptance-cluster",
        "architecture": "amd64",
        "notebooks": [{"path": "duplicate.ipynb", "status": "passed"}] * 9,
        "images": {
            kind: f"ghcr.io/eoap/{kind}@sha256:" + "a" * 64 for kind in ["training", "inference"]
        },
        "evidence_location": "retained-evidence",
        **dict.fromkeys(
            [
                "real_data",
                "editor_s3",
                "mlflow_artifacts",
                "lifecycle",
                "model_equivalence",
                "scientific_outputs",
            ],
            True,
        ),
    }
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="clean source"):
        gate.check(path)
    evidence["dirty_source"] = False
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="nine notebooks"):
        gate.check(path)
    evidence["notebooks"] = [
        {"path": str(p.relative_to(ROOT)), "status": "passed"}
        for p in sorted((ROOT / "practice-labs").rglob("*.ipynb"))
    ]
    path.write_text(json.dumps(evidence))
    gate.check(path)
    evidence["source_commit"] = "b" * 40
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="candidate tag"):
        gate.check(path)
