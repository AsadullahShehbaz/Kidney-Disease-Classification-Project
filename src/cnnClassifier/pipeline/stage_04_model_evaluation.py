from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_evaluation import Evaluation
from cnnClassifier import logger 


STAGE_NAME = "Evaluation Stage"

class EvaluationPipeline: 

    def __init__(self):
        """
        Initializes the EvaluationPipeline class.

        This class is responsible for evaluating a model that has been trained and saving the evaluation metrics to a JSON file.

        :return: None
        """
        pass 

    def main(self):
        """
        Evaluate a model and save the evaluation metrics to a JSON file.

        :return: None
        """
        
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()
        evaluation = Evaluation(config=eval_config)
        evaluation.evaluation()
        evaluation.save_score()
        # evaluation.log_into_mlflow()

if __name__ == "__main__":
    try:
        logger.info(f"*"*50)
        logger.info(f">>>>>>>>>>>>{STAGE_NAME} started <<<<<<<<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>>>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e