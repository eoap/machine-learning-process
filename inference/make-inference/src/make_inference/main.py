"""Classify a Sentinel-2 STAC item with an explicit or bundled ONNX model."""

import tempfile
from contextlib import ExitStack
from pathlib import Path

import click
import numpy as np
import onnxruntime as ort
import pystac
import rasterio

from .ml_helper import (
    create_stac_catalog,
    item_filter_assets,
    predict,
    resize_and_convert_to_cog,
    save_overview,
    save_prediction,
    sliding,
    stack_separated_bands,
)


@click.command()
@click.option("--input_reference", "-i", required=True, help="STAC item or catalog path/URL.")
@click.option(
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ONNX model; defaults to the packaged EuroSAT model.",
)
def run_inference(input_reference, model):
    """Write a georeferenced classification and self-contained STAC catalog."""
    reference = Path(input_reference)
    item = pystac.read_file(
        str(reference / "catalog.json") if reference.is_dir() else input_reference
    )
    if isinstance(item, pystac.Catalog):
        item = next(item.get_items(recursive=True), None)
    if not isinstance(item, pystac.Item):
        raise click.ClickException("Input must contain a STAC Item")
    model = model or Path(__file__).parent / "model/model.onnx"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Only intermediates created by this invocation are cleaned up.
    with tempfile.TemporaryDirectory(prefix="inference-") as scratch, ExitStack() as stack:
        stack.enter_context(
            rasterio.Env(
                CPL_VSIL_CURL_USE_HEAD="NO",
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                GDAL_HTTP_TIMEOUT="30",
            )
        )
        assets = item_filter_assets(item)
        srcs = {}
        for key, href in assets.items():
            local = resize_and_convert_to_cog(key, href, output_dir=scratch)
            srcs[key] = stack.enter_context(rasterio.open(local))
        reference_src = srcs["B02"]
        for src in srcs.values():
            if (src.shape, src.crs, src.transform) != (
                reference_src.shape,
                reference_src.crs,
                reference_src.transform,
            ):
                raise click.ClickException("All bands must share the same grid after resampling")
        meta = reference_src.meta.copy()
        prediction = np.full(reference_src.shape, 10, dtype=np.uint8)
        for window in sliding(reference_src.shape, 64, fixed=False):
            block = stack_separated_bands(window, srcs)
            classified = predict(block, session, input_name, output_name)
            h, w = int(window.height), int(window.width)
            y, x = int(window.row_off), int(window.col_off)
            prediction[y : y + h, x : x + w] = classified[:h, :w]
        save_prediction(prediction, f"{item.id}_classified.tif", meta.copy())
        save_overview(prediction, f"overview-{item.id}_classified.tif", meta.copy())
        create_stac_catalog(item)


def main():
    run_inference()


if __name__ == "__main__":
    main()
