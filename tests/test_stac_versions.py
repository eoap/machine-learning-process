"""Validate the requested STAC and MLM contracts independently of TensorFlow."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pystac
import pytest
from mlm.describe import MLM_SCHEMA_URI, describe
from pystac.extensions.eo import EOExtension
from pystac.extensions.mlm import AssetGeneralMLMExtension


def test_model_item_uses_mlm_16_and_stac_11(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model artifact")
    model.with_suffix(".json").write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "mlflow_run_id": "explicit-run",
                "architecture": "Sequential",
                "bands": [
                    "B01",
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B09",
                    "B11",
                    "B12",
                ],
                "classes": [f"class-{index}" for index in range(10)],
                "normalization": {"divisor": 10000.0},
            }
        )
    )
    pystac.set_stac_version("1.0.0")
    item = describe(model, tmp_path / "model-stac.json")
    saved = json.loads((tmp_path / "model-stac.json").read_text())
    assert saved["stac_version"] == "1.1.0"
    assert MLM_SCHEMA_URI in item.stac_extensions
    assert not any("mlm/v1.4" in uri for uri in item.stac_extensions)
    assert MLM_SCHEMA_URI in item.validate()
    reloaded = pystac.read_file(str(tmp_path / "model-stac.json"))
    assert MLM_SCHEMA_URI in reloaded.validate()
    assert [band.name for band in EOExtension.ext(item.assets["model"]).bands] == saved[
        "properties"
    ]["mlm:input"][0]["bands"]
    model_input = saved["properties"]["mlm:input"][0]
    assert model_input["input"]["dim_order"] == ["batch", "height", "width", "bands"]
    assert (
        model_input["pre_processing_function"]["expression"] == 'input.astype("float32") / 10000.0'
    )
    with pytest.raises(pystac.errors.RequiredPropertyMissing):
        AssetGeneralMLMExtension(item.assets["model"]).artifact_type = None
    AssetGeneralMLMExtension(item.assets["model"]).artifact_type = ""
    with pytest.raises(pystac.STACValidationError):
        item.validate()


def test_training_catalog_and_items_use_stac_11(tmp_path):
    path = Path(__file__).parents[1] / "scripts/create-training-fixture.py"
    spec = importlib.util.spec_from_file_location("training_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pystac.set_stac_version("1.0.0")
    catalog = pystac.read_file(str(module.create_fixture(tmp_path / "training")))
    catalog.validate_all()
    assert catalog.to_dict()["stac_version"] == "1.1.0"
    assert len(list(catalog.get_items(recursive=True))) == 100
    assert all(
        item.to_dict()["stac_version"] == "1.1.0" for item in catalog.get_items(recursive=True)
    )


def test_repository_catalog_validates():
    path = Path(__file__).parents[1] / "training/app-package/EUROSAT-Training-Dataset/catalog.json"
    catalog = pystac.read_file(str(path))
    catalog.validate()
    collection = pystac.read_file(str(path.parent / "EUROSAT-Training-Dataset/collection.json"))
    collection.validate()
    assert collection.to_dict()["stac_version"] == "1.1.0"
    assert catalog.to_dict()["stac_version"] == "1.1.0"
