"""Export an explicit Keras artifact and record its inference contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import click

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def export_model(
    source: str | Path, output: str | Path, run_id: str | None = None
) -> dict[str, object]:
    """Export and verify an ONNX artifact, writing its inference contract.

    Args:
        source: Explicit Keras checkpoint.
        output: Destination ONNX artifact.
        run_id: Optional MLflow provenance identifier.

    Returns:
        Contract recording the checksum, bands, classes and normalization.

    Raises:
        ValueError: The model input or output shape does not match EuroSAT.
    """
    # Conversion is a CPU operation; do not let Grappler probe unavailable CUDA libraries.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    import tf2onnx

    source, output = Path(source), Path(output)
    model = tf.keras.models.load_model(source, compile=False)
    if model.input_shape != (None, 64, 64, 12) or model.output_shape != (None, 10):
        raise ValueError("Expected 64x64x12 input and ten EuroSAT output classes")
    signature = [tf.TensorSpec([None, 64, 64, 12], tf.float32, name="input")]

    def forward(value: tf.Tensor) -> tf.Tensor:
        return model(value, training=False)

    traced_forward = tf.function(forward, input_signature=signature)
    output.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        traced_forward, input_signature=signature, opset=13, output_path=str(output)
    )
    onnx.checker.check_model(onnx.load(output))
    sample = np.random.default_rng(42).random((2, 64, 64, 12), dtype=np.float32)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
    np.testing.assert_allclose(actual, model(sample, training=False).numpy(), rtol=1e-4, atol=1e-5)
    metadata: dict[str, object] = {
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "architecture": type(model).__name__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mlflow_run_id": run_id,
        "bands": BANDS,
        "classes": CLASSES,
        "input_shape": [None, 64, 64, 12],
        "input_dtype": "float32",
        "normalization": {"divisor": 10000.0},
        "nodata": 10,
        "equivalence": {"rtol": 1e-4, "atol": 1e-5, "samples": 2},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@click.command()
@click.option("--source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--run-id", help="Explicit MLflow run; never selects a model by mutable ranking.")
@click.option("--tracking-uri", envvar="MLFLOW_TRACKING_URI")
@click.option("--output", required=True, type=click.Path(path_type=Path))
def main(source: Path | None, run_id: str | None, tracking_uri: str | None, output: Path) -> None:
    """Export --source model.keras or --run-id from MLflow to --output model.onnx."""
    if bool(source) == bool(run_id):
        raise click.UsageError("Provide exactly one of --source or --run-id")
    try:
        if run_id:
            import mlflow

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            source = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"runs:/{run_id}/model/trained_model.keras"
                )
            )
        if source is None:
            raise click.UsageError("Provide a model source")
        export_model(source, output, run_id)
    except Exception as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
