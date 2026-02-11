from cnnClassifier import logger 
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline
# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} Completed <<<<<<\n\n")
except Exception as e:

    logger.exception(e)
    raise 


# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Prepare Base Model Training Stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME}<<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
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

STAGE_NAME = "Evaluation Stage"
try:
    logger.info(f"*"*30)
    logger.info(f">>>>>>>>>>>>>>> {STAGE_NAME} Started <<<<<<<<<<<<<<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>>>>>>>>>>>>>> {STAGE_NAME} Completed <<<<<<<<<<<<<<<")
    
except Exception  as e:
    logger.exception(e)
    raise e 