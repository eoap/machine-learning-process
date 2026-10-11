from tile_based_training import logger
from tile_based_training.components.model_evaluation import Evaluation
from tile_based_training.config.configuration import ConfigurationManager

STAGE_NAME = "Evaluating Model"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        # tf.keras.backend.clear_session()
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(eval_config)
        test_dataset, conf_mat = evaluation.evaluation()
        evaluation.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info("***********************")
        logger.info("***********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
