import os
import sys
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=0'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tile_based_training import logger
import warnings
import click
from pathlib import Path
from tile_based_training.utils.common import (
    write_yaml,
    configure_gpu,
)
from tile_based_training.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from tile_based_training.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTRainingPipeline,
)
from tile_based_training.pipeline.stage_03_model_training import ModelTRainingPipeline
from tile_based_training.pipeline.stage_04_model_evaluation import (
    ModelEvaluationPipeline,
)

warnings.filterwarnings("ignore")


@click.command(
    short_help="making a tile-based classification on a sentinel-2 L1C data ",
    help="A selected model with highest evaluation metrics will making an inference on a sentinel-2 L1C data",
)
@click.option(
    "--stac_reference",
    "--sr",
    "stac_reference",
    help="The url which point to STAC input reference",
    default= "https://raw.githubusercontent.com/eoap/machine-learning-process/main/training/app-package/EUROSAT-Training-Dataset/catalog.json",
    required=True,
    show_default=True,
)
@click.option(
    "--BATCH_SIZE",
    "--b",
    "BATCH_SIZE",
    help="BATCH_SIZE",
    required=False,
    default=4,
    show_default=True,
    type=int,
)
@click.option(
    "--CLASSES",
    "--c",
    "CLASSES",
    help="Number of classes to train",
    required=False,
    default=10,
    show_default=True,
    type=int,
)
@click.option(
    "--DECAY",
    "--d",
    "DECAY",
    help="DECAY - model metadata",
    required=False,
    default=0.1,
    show_default=True,
    type=float,
)
@click.option("--EPOCHS", "--ep","EPOCHS", help="Number of epochs", required=False, default=5, type=int)
@click.option(
    "--EPSILON",
    "--e",
    "EPSILON",
    help="EPSILON - model metadata",
    required=False,
    default=0.000001,
    show_default=True,
    type=float,
)
@click.option(
    "--LEARNING_RATE",
    "--lr",
    "LEARNING_RATE",
    help="LEARNING_RATE",
    required=False,
    default=0.0001,
    show_default=True,
    type=float,
)
@click.option(
    "--LOSS",
    "--lo",
    "LOSS",
    help="loss function",
    required=False,
    default="categorical_crossentropy",
    show_default=True,
    type=str,
)
@click.option(
    "--MEMENTUM",
    "--m",
    "MEMENTUM",
    help="MEMENTUM - model metadata",
    required=False,
    default=0.95,
    show_default=True,
    type=float,
)
@click.option(
    "--OPTIMIZER",
    "--o",
    "OPTIMIZER",
    help="OPTIMIZER",
    required=False,
    default="Adam",
    show_default=True,
    type=str,
)
@click.option(
    "--REGULARIZER",
    "--r",
    "REGULARIZER",
    help="REGULARIZER",
    required=False,
    default="None",
    type=str,
)
@click.option(
    "--SAMPLES_PER_CLASS",
    "--s",
    "SAMPLES_PER_CLASS",
    help="number of sample for each class to train model based on",
    required=False,
    default=10,
    show_default=True,
    type=int,
)
@click.option(
    "--indexing_flag",
    "--idx",
    "indexing_flag",
    help="A flag to enable data ingestion pipeline",
    required=False,
    is_flag=True,
    default=False,
    show_default=True,
    type=bool,
)
@click.pass_context
def run_tile_based_classification_training(ctx, **kwargs):
    device_name = configure_gpu()
    logger.info(f"MLFLOW URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    logger.info(f"{os.getcwd()}")
    logger.info(
        f"\n=================================================================\nDevice name is: {device_name} \n================================================================="
    )
    from pprint import pprint
    pprint(kwargs)
    write_yaml(path_to_yaml=Path("params.yaml"), args=kwargs)

    #s3_bucket_config()
    # First step
    STAGE_NAME = "Data Ingestion stage"
   
    # try:
    #     logger.info(f"\n=================================================================\n>>>>>> stage {STAGE_NAME} started <<<<<<")
    #     obj = DataIngestionTrainingPipeline()
    #     obj.main()
    #     logger.info(
    #         f"\n=================================================================\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n================================================================="
    #     )
    # except Exception as e:
    #     logger.exception(e)
    #     raise e
    
    
    STAGE_NAME = "Prepare Base Model"
    logger.info(f"\n=================================================================\n>>>>>> stage {STAGE_NAME} started <<<<<<")

    try:
        obj = PrepareBaseModelTRainingPipeline()
        obj.main()
        logger.info(
            f"\n=================================================================\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n================================================================="
        )
    except Exception as e:
        logger.exception(e)
        raise e
    STAGE_NAME = "Training Model"

    try:
        logger.info(f"\n=================================================================\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTRainingPipeline()
        obj.main(device_name)
        logger.info(
            f"\n=================================================================\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
    except Exception as e:
        logger.exception(e)
        raise e
    sys.exit(0)
    STAGE_NAME = "Evaluating Model"

    try:
        logger.info(f"\n=================================================================\n>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(
            f"\n=================================================================\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
        )
    except Exception as e:
        logger.exception(e)
        raise e


def main():
    run_tile_based_classification_training()


if __name__ == "__main__":
    # Log the experiment
    main()


# # First step
# STAGE_NAME = "Data Ingestion stage"

# try:
#     if not (os.path.isfile("./output/data_ingestion/splitted_data.json")):
#         logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
#         obj = DataIngestionTrainingPipeline()
#         obj.main()
#         logger.info(
#             f"\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n================================================================="
#         )
#     else:
#         logger.info(
#             f"\n>>> stage {STAGE_NAME} is completed in previous attempts <<<\n================================================================="
#         )
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Prepare Base Model"

# try:
#     logger.info(f"\n>>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = PrepareBaseModelTRainingPipeline()
#     obj.main()
#     logger.info(
#         f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n================================================================="
#     )
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Training Model"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = ModelTRainingPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e


# STAGE_NAME = "Evaluating Model"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     obj = ModelEvaluationPipeline()
#     obj.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
