from cnnClassifier import logger 
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_basemodel import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline

# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = DataIngestionTrainingPipeline()
    # obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} Completed <<<<<<\n\n")
except Exception as e:

    logger.exception(e)
    raise 


# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Prepare Base Model Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    # obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} Completed<<<<<<\n\n")
except Exception as e:

    logger.exception(e)
    raise 


# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} Completed<<<<<<\n\n")
except Exception as e:

    logger.exception(e)
    raise 