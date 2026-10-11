from tile_based_training import logger
from tile_based_training.components.prepare_base_model import PrepareBaseModel
from tile_based_training.config.configuration import ConfigurationManager

STAGE_NAME = "Prepare Base Model"


class PrepareBaseModelTRainingPipeline:
    def __init__(self):
        pass

    def main(self):
        # tf.keras.backend.clear_session()
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.base_model()


if __name__ == "__main__":
    try:
        logger.info("***********************")
        logger.info("***********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTRainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
