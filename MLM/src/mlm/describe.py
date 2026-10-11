"""Describe an exported model using the STAC MLM extension."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pystac
from pystac.extensions import mlm
from pystac.extensions.eo import Band, EOExtension
from pystac.extensions.raster import DataType

MLM_SCHEMA_URI = "https://stac-extensions.github.io/mlm/v1.6.0/schema.json"


class MLM16Extension(mlm.ItemMLMExtension):
    """Use the fork's MLM field API with the official 1.6.0 schema.

    The supported field APIs used here are compatible with MLM 1.6.0.
    Overriding the schema accessor registers the requested version through
    PySTAC's extension API without changing serialized extension lists.
    """

    @classmethod
    def get_schema_uri(cls) -> str:
        """Return the schema used to validate model metadata."""
        return MLM_SCHEMA_URI


def describe(model: str | Path, output: str | Path) -> pystac.Item:
    """Save a validated STAC 1.1.0 Item describing an exported ONNX model.

    Args:
        model: ONNX artifact with an adjacent JSON inference contract.
        output: Destination of the model Item.

    Returns:
        The validated model Item, including its artifact and contract assets.

    Raises:
        ValueError: The artifact checksum differs from its contract.
        pystac.STACValidationError: Model metadata fails schema validation.
    """
    pystac.set_stac_version("1.1.0")
    model, output = Path(model).resolve(), Path(output).resolve()
    contract = json.loads(model.with_suffix(".json").read_text())
    if hashlib.sha256(model.read_bytes()).hexdigest() != contract["sha256"]:
        raise ValueError("Model checksum does not match its inference contract")
    item = pystac.Item(
        "eurosat-model",
        None,
        None,
        datetime.now(UTC),
        {"model:sha256": contract["sha256"], "model:source_run": contract["mlflow_run_id"]},
    )
    MLM16Extension.add_to(item)
    extension = MLM16Extension(item)
    extension.apply(
        name="EuroSAT tile classifier",
        architecture=contract["architecture"],
        tasks=[mlm.TaskType.SCENE_CLASSIFICATION],
        input=[
            mlm.ModelInput.create(
                "input",
                contract["bands"],
                mlm.InputStructure.create(
                    [-1, 64, 64, 12], ["batch", "height", "width", "bands"], DataType.FLOAT32
                ),
                pre_processing_function=mlm.ProcessingExpression.create(
                    format="python",
                    expression=f'input.astype("float32") / {contract["normalization"]["divisor"]}',
                ),
            )
        ],
        output=[
            mlm.ModelOutput.create(
                "probabilities",
                [mlm.TaskType.SCENE_CLASSIFICATION],
                mlm.ResultStructure.create([-1, 10], ["batch", "class"], DataType.FLOAT32),
                description="Class order: " + ", ".join(contract["classes"]),
            )
        ],
        framework="TensorFlow",
        framework_version="2.21.0",
    )
    item.add_asset(
        "model",
        pystac.Asset(
            str(model),
            media_type="application/octet-stream",
            roles=["mlm:model"],
        ),
    )
    mlm.AssetGeneralMLMExtension(item.assets["model"]).apply(artifact_type="onnx")
    # MLM named input bands must also have an EO, Raster, or core band definition.
    EOExtension.ext(item.assets["model"], add_if_missing=True).bands = [
        Band.create(name=band) for band in contract["bands"]
    ]
    item.add_asset(
        "contract",
        pystac.Asset(
            str(model.with_suffix(".json")), media_type=pystac.MediaType.JSON, roles=["metadata"]
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    item.set_self_href(str(output))
    item.validate()
    item.save_object()
    return item
