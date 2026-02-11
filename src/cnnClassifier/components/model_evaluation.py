import tensorflow as tf
from pathlib import Path 
import mlflow
import mlflow.keras
from urllib.parse import urlparse 
from cnnClassifier.entitiy.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json
from dotenv import load_dotenv
load_dotenv()
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config  # Store config with model path, image size, batch size

    def _valid_generator(self):
        # Image preprocessing settings
        datagenerator_kwargs = dict(
            rescale=1./255,  # Normalize pixel values from [0,255] to [0,1]
            validation_split=0.30  # Reserve 30% of data for validation
        )

        # Data loading settings
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Resize images to (height, width)
            batch_size=self.config.params_batch_size,  # Number of images per batch
            interpolation="bilinear"  # Image resizing method
        )

        # Create ImageDataGenerator for validation data
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Load validation images from directory
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,  # Path to data folder
            subset="validation",  # Use validation split
            shuffle=False,  # Keep order for consistent evaluation
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """Load saved Keras model from disk"""
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        # Load the trained model
        self.model = self.load_model(self.config.path_of_model)
        # Prepare validation data
        self._valid_generator()
        # Evaluate model and get loss/metrics
        self.score = self.model.evaluate(self.valid_generator)  # Fixed: was 'model', should be 'self.model'

        # self.save_score()

    def save_score(self):
        scores = {"loss":self.score[0],"accuracy":self.score[1]}
        save_json(path=Path("scores.json"),data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)

        tracking_uri_type_store = urlparse(mlflow.get_tracking_uri()).scheme 

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss":self.score[0],"accuracy":self.score[1]}
            )

            # Model registry does not work with file store 
            if tracking_uri_type_store != "file":

                # Register the model 
                mlflow.keras.log_model(self.model,"model")
            else:
                mlflow.keras.log_model(self.model,"model")

