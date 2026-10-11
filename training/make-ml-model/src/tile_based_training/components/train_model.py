import logging
import random

import numpy as np
import tensorflow as tf
from tile_based_training.utils.common import augmentation, rasterio_read
from tqdm import tqdm

tf.config.run_functions_eagerly(True)


logger = logging.getLogger(__name__)


class Training:
    def __init__(self, config):
        self.config = config
        self.model = None

        self.class_names = [
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
        self.label_lookup = {name: idx for idx, name in enumerate(self.class_names)}

    def get_base_model(self):
        self.model = tf.keras.models.load_model(self.config.base_model_path)

    def fetch_image(self, url):
        image = rasterio_read(str(url))  # e.g., (12, 64, 64)
        image = augmentation(image)  # Apply augmentations
        image = np.transpose(image, (1, 2, 0))  # to (64, 64, 12)
        return image.astype(np.float32)

    def _process_data(self, url):
        image = self.fetch_image(url)
        label_str = url.split("/")[-2]
        label_id = self.label_lookup[label_str]
        label = tf.one_hot(label_id, depth=10)
        return image, label

    def ___getitem__(self, index):
        url = self.config.train_data["url"][index]
        return self._process_data(url)

    def _load_data_as_arrays(self, urls):
        images = []
        labels = []

        for url in tqdm(urls, total=len(urls), desc="Loading data"):
            image, label = self._process_data(url)
            images.append(image)
            labels.append(label)

        images = np.stack(images)  # shape: (N, 64, 64, 12)
        labels = tf.stack(labels)  # keep as Tensor

        return images, labels

    def train_model(self, device_name):
        train_paths = self.config.train_data["url"]
        val_paths = self.config.val_data["url"]

        random.shuffle(train_paths)

        x_train, y_train = self._load_data_as_arrays(train_paths)
        x_val, y_val = self._load_data_as_arrays(val_paths)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.trained_model_path) + "/trained_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        logger.info(f"Device is: {device_name}, Built with CUDA: {tf.test.is_built_with_cuda()}")

        self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=self.config.params_epochs,
            batch_size=self.config.params_batch_size,
            callbacks=[checkpoint],
            verbose=1,
        )
        try:
            del x_train, y_train, x_val, y_val
            tf.keras.backend.clear_session()
            logger.info("Training completed and session cleared.")

        except Exception as e:
            logger.error(f"Error during deletion: {e}")

    def process(self, url):
        image, label = tf.py_function(
            func=self._process_data, inp=[url], Tout=(tf.float32, tf.float32)
        )
        image.set_shape([64, 64, 12])
        label.set_shape([10])
        return image, label

    def train_valid_dataloader(self, file_paths):
        file_paths = tf.constant(file_paths)
        dataset = tf.data.Dataset.from_tensor_slices(file_paths)
        dataset = dataset.map(self.process, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.config.params_batch_size)
        dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
        return dataset
