from dataclasses import dataclass
from pathlib import Path


# for equality checks
@dataclass(frozen=True)  # for equality checks
class DataIngestionConfig:
    root_dir: Path
    stac_reference: str
    local_data_file: Path
    data_classes: list
    samples_per_class: int


@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    model_path: Path

    params_learning_rate: float
    params_classes: int
    params_decay: float
    params_loss: str
    params_optimizer: str
    params_kernel_regularizer: str
    params_momentum: float
    params_epsilon: float


# Entity
@dataclass(frozen=True)
class TrainingConfig:
    # will used in case of downloading data
    train_data_dir: Path
    val_data_dir: Path
    # will used in case of reading data directly from s3
    train_data: dict
    val_data: dict

    trained_model_path: Path
    base_model_path: Path
    params_epochs: int
    params_batch_size: int

    classes_number: int


@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    test_data: Path
    all_params: dict  # comes from parameters.yml
    params_batch_size: int
    test_data_paths: dict
