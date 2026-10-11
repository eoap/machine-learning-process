from tile_based_training import logger
from tile_based_training.components.data_ingestion import DataIngestion
from tile_based_training.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            print(data_ingestion_config)
            data_ingestion = DataIngestion(config=data_ingestion_config)
            all_urls = data_ingestion.stac_loader()
            data_ingestion.data_spliting(all_urls)

            ### Uncomment this only if you want to download the data locally
            # data_ingestion.data_downloader(dataset["train"],split_name="train")
            # data_ingestion.data_downloader(dataset["val"],split_name="val")
            # data_ingestion.data_downloader(dataset["test"],split_name="test")
        except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
