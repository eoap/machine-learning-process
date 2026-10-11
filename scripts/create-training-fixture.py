"""Small local EuroSAT-shaped fixture; validates execution rather than accuracy."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pystac
import rasterio
from rasterio.transform import from_origin
from tile_based_training.export import CLASSES


def create_fixture(root: str | Path) -> Path:
    """Create and validate a local STAC 1.1.0 training catalog.

    Args:
        root: Directory receiving synthetic chips and catalog documents.

    Returns:
        Path to the catalog document.
    """
    pystac.set_stac_version("1.1.0")
    root = Path(root).resolve()
    catalog = pystac.Catalog(
        "training-fixture", "Synthetic execution fixture, not an accuracy benchmark"
    )
    random = np.random.default_rng(42)
    for label in CLASSES:
        directory = root / label
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(10):
            path = directory / f"{index}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=13,
                dtype="uint16",
                crs="EPSG:32632",
                transform=from_origin(500000, 5100000, 10, 10),
            ) as dst:
                dst.write(random.integers(1, 10000, (13, 64, 64), dtype=np.uint16))
            item = pystac.Item(
                f"{label}-{index}",
                None,
                None,
                datetime(2025, 1, 1, tzinfo=UTC),
                {"class": label},
            )
            item.add_asset(
                "image",
                pystac.Asset(str(path), roles=["data"], media_type=pystac.MediaType.GEOTIFF),
            )
            catalog.add_item(item)
    catalog.normalize_hrefs(str(root))
    catalog.validate_all()
    catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
    return root / "catalog.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/training-fixture"))
    print(create_fixture(parser.parse_args().output))
