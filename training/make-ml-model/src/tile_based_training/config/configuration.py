from pathlib import Path

from box import ConfigBox

from tile_based_training.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, config_dict
from tile_based_training.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
)
from tile_based_training.utils.common import create_directories, load_json, read_yaml


# CONFIGURATION MANAGER
class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = (
            read_yaml(config_filepath) if Path(config_filepath).exists() else ConfigBox(config_dict)
        )
        self.params = read_yaml(params_filepath)

        create_directories([self.config.output_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            stac_reference=self.params.stac_reference,
            local_data_file=config.local_data_file,
            data_classes=self.config.data_ingestion.DATA_CLASSES,
            samples_per_class=self.params.SAMPLES_PER_CLASS,
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_learning_rate=self.params.LEARNING_RATE,
            params_classes=self.params.CLASSES,
            params_decay=self.params.DECAY,
            params_loss=self.params.LOSS,
            params_optimizer=self.params.OPTIMIZER,
            params_kernel_regularizer=self.params.REGULARIZER,
            params_momentum=self.params.MEMENTUM,
            params_epsilon=self.params.EPSILON,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training
        prepare_base_model = self.config.prepare_base_model
        train_data = load_json(Path(training_config.all_data_urls + "/splitted_data.json"))["train"]
        val_data = load_json(Path(training_config.all_data_urls + "/splitted_data.json"))["val"]
        # load params
        params = self.params
        create_directories([Path(training_config.trained_model_path)])

        training_config = TrainingConfig(
            # will be used in case of downloading data
            val_data_dir=Path(training_config.val_data_dir),
            train_data_dir=Path(training_config.train_data_dir),
            # will be used in case of reading data directly from s3
            train_data=train_data,
            val_data=val_data,
            trained_model_path=Path(training_config.trained_model_path),
            base_model_path=Path(prepare_base_model.model_path),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            classes_number=self.params.CLASSES,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        test_data = load_json(Path(self.config.evaluation.all_data_urls + "/splitted_data.json"))[
            "test"
        ]

        eval_config = EvaluationConfig(
            path_of_model=self.config.evaluation.trained_model_path,
            test_data=self.config.evaluation.data_dir,
            all_params=self.params,
            params_batch_size=self.params.BATCH_SIZE,
            test_data_paths=test_data,
        )
        return eval_config
