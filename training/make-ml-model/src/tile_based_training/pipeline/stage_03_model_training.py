from tile_based_training import logger
from tile_based_training.components.train_model import Training
from tile_based_training.config.configuration import ConfigurationManager

STAGE_NAME = "Training Model"


class ModelTRainingPipeline:
    def __init__(self):
        pass

    def main(self, device_name):
        # tf.keras.backend.clear_session()
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_model(device_name)


if __name__ == "__main__":
    try:
        logger.info("***********************")
        logger.info("***********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTRainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
