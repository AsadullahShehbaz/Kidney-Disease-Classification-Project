from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training import Training
from cnnClassifier import logger 

STAGE_NAME = "Training"

class ModelTrainingPipeline:

    def __init__(self):
        pass 

    def main(self):

        # Initialize config object 
        config = ConfigurationManager()

        # Get all training related configuration values 
        training_config = config.get_training_config()

        # Initialize the Training class with configuration setting
        training = Training(config=training_config)

        # Load the prepared VGG16 model from disk 
        training.get_base_model()

        # Create data generator objects for feeding images to model during training
        training.train_valid_generator()

        # Train the VGG16 model on kidney CT scan images
        training.train()

if __name__ == "__main__":
    try:
        logger.info(f"*"*20)
        logger.info(f">>>>>>>>> {STAGE_NAME} STARTED <<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>>{STAGE_NAME} completed <<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e 
    