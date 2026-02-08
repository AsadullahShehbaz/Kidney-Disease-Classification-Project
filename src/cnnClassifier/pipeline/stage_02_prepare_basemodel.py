from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger 

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass 

    def main(self):
        # 1.Create config object to get hyper-parameters 
        config  = ConfigurationManager()
    
        # 2.Get base model parameter values 
        prepare_base_model_config = config.get_prepare_base_model_config()

        # 3.Create model object by passing parameter values 
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # 4.Get base model object by downloading automatically and save locally
        prepare_base_model.get_base_model()
        logger.info(f"Base model saved at {prepare_base_model_config.updated_base_model_path}")
        
        # 5.Update parameters of basemodel and create our classifier CNN Architecture
        prepare_base_model.update_base_model() 

if __name__ == "__main__":
    logger.info(">>>>> stage {STAGE_NAME} started <<<<<")

    obj = PrepareBaseModelTrainingPipeline()
    obj.main()

    logger.info(">>>>> stage {STAGE_NAME} completed <<<<<")
