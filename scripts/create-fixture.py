"""Create portable twelve-band STAC data, including partial edge tiles."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pystac
import rasterio
from pystac.extensions.eo import Band, EOExtension
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]


def create_fixture(root: str | Path) -> Path:
    """Write and validate a STAC 1.1.0 Item with twelve local band assets.

    Args:
        root: Directory receiving the Item and its raster assets.

    Returns:
        Path to the validated Item document.
    """
    pystac.set_stac_version("1.1.0")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    bounds = transform_bounds("EPSG:32632", "EPSG:4326", 500000, 5099300, 500730, 5100000)
    west, south, east, north = bounds
    geometry = {
        "type": "Polygon",
        "coordinates": [
            [[west, south], [east, south], [east, north], [west, north], [west, south]]
        ],
    }
    item = pystac.Item("fixture", geometry, list(bounds), datetime(2025, 1, 1, tzinfo=UTC), {})
    for index, band in enumerate(BANDS):
        path = root / f"{band}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=70,
            width=73,
            count=1,
            dtype="uint16",
            crs="EPSG:32632",
            transform=from_origin(500000, 5100000, 10, 10),
            nodata=0,
        ) as dst:
            data = np.full((70, 73), 1000 + index * 100, dtype=np.uint16)
            data[0, 0] = 0
            dst.write(data, 1)
        item.add_asset(
            band,
            pystac.Asset(
                f"./{band}.tif",
                media_type=pystac.MediaType.GEOTIFF,
                roles=["data"],
            ),
        )
        EOExtension.ext(item.assets[band], add_if_missing=True).bands = [Band.create(name=band)]
    item.set_self_href(str(root / "item.json"))
    item.validate()
    item.save_object()
    return root / "item.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/fixture"))
    print(create_fixture(parser.parse_args().output))
