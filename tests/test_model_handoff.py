import hashlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import numpy as np
import pytest
import tensorflow as tf
from click.testing import CliRunner
from tile_based_training.export import export_model, main


def test_training_export_handoff(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.Sequential(
        [
            tf.keras.Input((64, 64, 12)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    data = np.random.default_rng(42).random((2, 64, 64, 12), dtype=np.float32)
    model.train_on_batch(data, np.array([0, 1]))
    source, output = tmp_path / "trained.keras", tmp_path / "model.onnx"
    model.save(source)
    metadata = export_model(source, output)
    assert metadata["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert metadata["input_shape"] == [None, 64, 64, 12]
    assert len(metadata["bands"]) == 12
    assert len(metadata["classes"]) == 10
    assert output.with_suffix(".json").exists()
    from mlm.describe import describe

    item = describe(output, tmp_path / "model-stac.json")
    assert item.properties["mlm:input"][0]["bands"] == metadata["bands"]
    assert item.assets["model"].extra_fields["mlm:artifact_type"] == "onnx"
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        describe(output, tmp_path / "invalid.json")


def test_export_requires_explicit_selection(tmp_path):
    result = CliRunner().invoke(main, ["--output", str(tmp_path / "model.onnx")])
    assert result.exit_code != 0
    assert "exactly one" in result.output


def test_evaluation_uses_requested_path(tmp_path, monkeypatch):
    from tile_based_training.components import model_evaluation

    seen = []
    monkeypatch.setattr(model_evaluation, "load_model", lambda path: seen.append(path))
    requested = tmp_path / "custom.keras"
    model_evaluation.Evaluation.load_model(requested)
    assert seen == [requested]


def test_import_has_no_filesystem_side_effects(tmp_path):
    subprocess.run(
        [sys.executable, "-c", "import tile_based_training; import tile_based_training.constants"],
        cwd=tmp_path,
        check=True,
    )
    assert list(tmp_path.iterdir()) == []


def test_mlflow_artifact_survives_cwl_output_move(tmp_path, monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace

    import matplotlib.pyplot as plt
    import mlflow
    from tile_based_training.components.model_evaluation import Evaluation

    working = tmp_path / "temporary-cwl-output"
    working.mkdir()
    model = working / "trained_model.keras"
    model.write_bytes(b"checkpoint artifact")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + str(tmp_path / "tracking.db"))
    monkeypatch.chdir(working)
    evaluation = Evaluation(SimpleNamespace(path_of_model=model, all_params={"EPOCHS": 1}))
    evaluation.score = [1.0, 0.5, 0.5, 0.5]
    monkeypatch.setattr(evaluation, "plot_confusion_matrix", plt.figure)
    evaluation.log_into_mlflow()
    run_id = json.loads((working / "model-run.json").read_text())["run_id"]
    monkeypatch.chdir(tmp_path)
    shutil.rmtree(working)
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/trained_model.keras"
    )
    assert Path(downloaded).read_bytes() == b"checkpoint artifact"
    plt.close("all")
