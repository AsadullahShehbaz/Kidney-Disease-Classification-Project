from cnnClassifier import logger 
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME}<<<<<<\n\n")
except Exception as e:

    logger.exception(e)
    raise 

