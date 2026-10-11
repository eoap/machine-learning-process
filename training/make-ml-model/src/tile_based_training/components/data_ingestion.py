# from . import logger
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pystac
import requests

# import botocore
from sklearn.model_selection import train_test_split
from tile_based_training import logger
from tile_based_training.entity.config_entity import DataIngestionConfig
from tqdm import tqdm


# UPDATING src's component
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _local_samples(self, items: list[pystac.Item]) -> list[dict[str, str]]:
        """Select a deterministic sample of each configured class from local Items."""
        samples = []
        for class_name in self.config.data_classes:
            selected = [item for item in items if item.properties.get("class") == class_name]
            if len(selected) < 5:
                raise ValueError(f"Training fixture needs at least five items for {class_name}")
            for item in selected[: self.config.samples_per_class]:
                href = item.assets["image"].get_absolute_href()
                if href is None:
                    raise ValueError(f"Item {item.id} requires a resolvable image asset")
                samples.append({"url": href, "label": class_name})
        return samples

    def _geoparquet_samples(self, table: pa.Table) -> list[dict[str, str]]:
        """Select deterministic image references from a STAC GeoParquet table."""
        samples = []
        for class_name in tqdm(self.config.data_classes):
            selected = (
                table.filter(pc.starts_with(table["id"], class_name + "_"))
                .sort_by("id")
                .slice(0, self.config.samples_per_class)
            )
            if selected.num_rows < 5:
                raise ValueError(f"At least five samples are required for {class_name}")
            for row in selected.select(["assets"]).to_pylist():
                asset = pystac.Asset.from_dict(row["assets"]["image"])
                href = asset.href
                # Archive metadata has no implemented extension in the required fork.
                if asset.extra_fields.get("archive:href"):
                    href += "/" + asset.extra_fields["archive:href"]
                samples.append({"url": href, "label": class_name})
        return samples

    def stac_loader(self) -> list[dict[str, str]]:
        """Load class-balanced image references from Items or a GeoParquet collection.

        Returns:
            Image URLs and their configured class labels.

        Raises:
            ValueError: A class has fewer than five samples or required assets are absent.
        """
        logger.info("Accessing STAC endpoint")
        catalog = pystac.read_file(self.config.stac_reference)
        if not isinstance(catalog, pystac.Catalog):
            raise ValueError("Training input must be a STAC Catalog or Collection")
        local_items = list(catalog.get_items(recursive=True))
        if local_items:
            return self._local_samples(local_items)
        collection = next(iter(catalog.get_children()))
        if not isinstance(collection, pystac.Collection):
            raise ValueError("Training catalog requires a STAC Collection")
        parquet_href = next(
            (
                asset.get_absolute_href()
                for asset in collection.get_assets().values()
                if asset.media_type == "application/vnd.apache.parquet"
            ),
            None,
        )
        if parquet_href is None:
            raise ValueError("Training collection requires a STAC GeoParquet asset")
        if parquet_href.startswith(("http://", "https://")):
            response = requests.get(parquet_href, timeout=60)
            response.raise_for_status()
            table = pq.read_table(io.BytesIO(response.content))
        else:
            table = pq.read_table(parquet_href)
        samples = self._geoparquet_samples(table)
        logger.info(f"{len(samples)} items are loaded")
        return samples

    def data_spliting(self, full_dataset, split_name="train"):

        X = [x["url"] for x in full_dataset]
        y = [x["label"] for x in full_dataset]
        random_state = 42
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train
        )
        logger.info(
            f"Train size: {len(X_train)} , Valid size: {len(X_val)}, Test size: {len(X_test)}"
        )
        dataset = {
            "train": {"url": X_train, "label": y_train},
            "val": {"url": X_val, "label": y_val},
            "test": {"url": X_test, "label": y_test},
        }

        with (Path(self.config.root_dir) / "splitted_data.json").open("w") as stream:
            json.dump(dataset, stream)

        return dataset
