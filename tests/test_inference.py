import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pystac
import pytest
import rasterio
from make_inference.ml_helper import item_filter_assets, predict, sliding


def test_required_bands_and_order(fixture_item):
    item = pystac.read_file(str(fixture_item))
    assert list(item_filter_assets(item)) == [
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
    ]
    del item.assets["B01"]
    with pytest.raises(ValueError, match="B01"):
        item_filter_assets(item)


def test_edge_coverage():
    coverage = np.zeros((70, 73), dtype=int)
    for window in sliding(coverage.shape, 64, fixed=False):
        y, x, h, w = map(int, (window.row_off, window.col_off, window.height, window.width))
        coverage[y : y + h, x : x + w] += 1
    assert np.all(coverage == 1)


def test_normalization_and_nodata():
    class Session:
        def run(self, names, inputs):
            value = inputs["input"]
            assert value.dtype == np.float32
            assert value.shape == (1, 64, 64, 12)
            assert value.max() == 1
            return [np.array([[0.1, 0.9]])]

    data = np.full((12, 64, 64), 10000, dtype=np.uint16)
    data[:, 0, 0] = 0
    classified = predict(data, Session(), "input", "output")
    assert classified[0, 0] == 10
    assert np.all(classified[1:] == 1)


def test_real_bundled_onnx():
    from make_inference import main

    path = Path(main.__file__).parent / "model/model.onnx"
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    result = predict(
        np.full((12, 64, 64), 1000),
        session,
        session.get_inputs()[0].name,
        session.get_outputs()[0].name,
    )
    assert result.shape == (64, 64)
    assert set(np.unique(result)) <= set(range(10))


def test_cli_stac_raster_roundtrip(fixture_item, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "unrelated.tif"
    sentinel.write_bytes(b"user file")
    subprocess.run(
        [str(Path(sys.executable).with_name("make-inference")), "-i", str(fixture_item)],
        cwd=output,
        check=True,
        capture_output=True,
        text=True,
    )
    catalog = pystac.read_file(str(output / "catalog.json"))
    item = next(catalog.get_items(recursive=True))
    assert item.datetime == pystac.read_file(str(fixture_item)).datetime
    with rasterio.open(item.assets["fixture"].get_absolute_href()) as src:
        assert src.shape == (70, 73)
        assert src.crs.to_epsg() == 32632
        assert src.transform.a == 10
        data = src.read(1)
        assert data[0, 0] == src.nodata == 10
        assert np.all(data[-1, 1:] < 10)
        assert set(np.unique(data)) <= set(range(11))
    assert sentinel.read_bytes() == b"user file"


def test_all_nodata_statistics_are_finite():
    from make_inference.ml_helper import _get_stats

    statistics = _get_stats(np.ma.masked_all((64, 64)))
    assert statistics == {"statistics": {"valid_percent": 0.0}}
