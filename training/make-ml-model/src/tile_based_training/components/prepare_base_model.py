from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from tile_based_training.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    @staticmethod
    def _model_architecture(
        input_shape: tuple[int, int, int] = (64, 64, 12),
        kernel_regularizer: str | tf.keras.regularizers.Regularizer | None = None,
        number_of_classes: int = 10,
        lr: float = 0.01,
        epsilon: float = 1e-7,
        momentum: float = 0.9,
        decay: float | None = None,
        loss: str = "categorical_crossentropy",
        optimizer: str = "Adam",
    ) -> tf.keras.Model:
        """Build and compile the configured EuroSAT convolutional classifier."""
        if kernel_regularizer == "None":
            kernel_regularizer = None
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_regularizer=kernel_regularizer))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, kernel_regularizer=kernel_regularizer))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(number_of_classes))
        model.add(Activation("softmax"))

        optimizers = {
            "Adam": tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon),
            "SGD": tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, weight_decay=decay),
            "RMSprop": tf.keras.optimizers.RMSprop(
                learning_rate=lr,
                epsilon=epsilon,
                weight_decay=decay,
                ema_momentum=momentum,
            ),
        }
        opt = optimizers[optimizer]
        metrics = [
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]

        model.compile(loss=loss, optimizer=opt, metrics=metrics)
        model.summary()
        return model

    def base_model(self):
        self.model = self._model_architecture(
            kernel_regularizer=self.config.params_kernel_regularizer,
            number_of_classes=self.config.params_classes,
            lr=self.config.params_learning_rate,
            epsilon=self.config.params_epsilon,
            momentum=self.config.params_momentum,
            decay=self.config.params_decay,
            loss=self.config.params_loss,
            optimizer=self.config.params_optimizer,
        )

        self.save_model(path=self.config.model_path, model=self.model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
