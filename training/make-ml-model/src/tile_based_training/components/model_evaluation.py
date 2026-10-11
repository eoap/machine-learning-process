import json
import os
from pathlib import Path

import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf

# from src.pipeline.stage_03_model_training import history
from keras.models import load_model
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn import metrics

from tile_based_training.config.configuration import EvaluationConfig
from tile_based_training.utils.common import create_directories, rasterio_read


class Evaluation:
    def __init__(self, config: EvaluationConfig):
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

    @staticmethod
    def load_model(path: Path):
        return load_model(path)

    def _load_data_as_arrays(self, urls):
        images = []
        labels = []

        for url in urls:
            image, label = self._process_data(url)
            images.append(image)
            labels.append(label)

        images = np.stack(images)  # shape: (N, 64, 64, 12)
        labels = tf.stack(labels).numpy()  # shape: (N, 10)
        return images, labels

    def fetch_image(self, url):
        image = rasterio_read(str(url))  # e.g., (12, 64, 64)
        image = image / 10000.0  # Evaluation must not randomly augment inputs
        image = np.transpose(image, (1, 2, 0))  # to (64, 64, 12)
        return image.astype(np.float32)

    def _process_data(self, url):
        image = self.fetch_image(url)
        label_str = url.split("/")[-2]
        label_id = self.label_lookup[label_str]
        label = tf.one_hot(label_id, depth=10)
        return image, label

    def evaluation(self):
        create_directories([Path("./mlruns")])
        self.model = self.load_model(self.config.path_of_model)

        test_paths = self.config.test_data_paths["url"]

        x_test, y_test = self._load_data_as_arrays(test_paths)

        self.score = self.model.evaluate(x_test, y_test)
        y_pred = self.model.predict(x_test)

        self.y_true = np.argmax(y_test, axis=1)
        self.y_pred_amax = np.argmax(y_pred, axis=1)

        self.matrix = metrics.confusion_matrix(self.y_true, self.y_pred_amax)

        sample_input = x_test[0:1]  # ✅ keep batch dim
        sample_output = self.model.predict(sample_input)
        self.signature = infer_signature(sample_input, sample_output)
        self.save_scores()
        return (x_test, y_test), self.matrix

    def save_scores(self):
        print(
            {
                "new_experiment": {
                    "test_loss": self.score[0],
                    "test_accuracy": self.score[1],
                    "test_precision": self.score[2],
                    "test_recall": self.score[3],
                }
            }
        )

    def plot_confusion_matrix(self):
        class_names = np.unique(self.y_true)
        sns.set(font_scale=1.2)  # Adjust font size for better readability
        fig, ax = plt.subplots()
        # Create a heatmap
        sns.heatmap(
            self.matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )

        # Add labels and title
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        # Show the plot
        plt.show()
        return fig

    def log_into_mlflow(self):
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        tracking_uri = mlflow.get_tracking_uri()
        if (
            tracking_uri.startswith("sqlite:///")
            and mlflow.get_experiment_by_name("EuroSAT_classification") is None
        ):
            database = Path(tracking_uri[len("sqlite:///") :]).resolve()
            artifacts = database.parent / "mlartifacts"
            artifacts.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment("EuroSAT_classification", artifact_location=artifacts.as_uri())
        mlflow.set_experiment("EuroSAT_classification")
        with mlflow.start_run() as run:
            mlflow.log_params(self.config.all_params)
            mlflow.log_figure(self.plot_confusion_matrix(), "Confusion_Matrix.png")
            mlflow.log_metrics(
                {
                    "loss": self.score[0],
                    "test_accuracy": self.score[1],
                    "test_precision": self.score[2],
                    "test_recall": self.score[3],
                }
            )
            # A raw Keras artifact keeps run-based export stable across server versions.
            mlflow.log_artifact(str(self.config.path_of_model), artifact_path="model")
            Path("model-run.json").write_text(
                json.dumps(
                    {
                        "run_id": run.info.run_id,
                        "tracking_uri": mlflow.get_tracking_uri(),
                        "model_path": str(Path(self.config.path_of_model).resolve()),
                    },
                    indent=2,
                )
            )
