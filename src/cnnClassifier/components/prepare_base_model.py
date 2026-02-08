import os 
from pathlib import Path 
import urllib.request as request 
from zipfile import ZipFile 
import tensorflow as tf 
from cnnClassifier.entitiy.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    """Download VGG16 base model from website of Keras application"""
    
    def __init__(self, config: PrepareBaseModelConfig):
        """
        Initialize the class with configuration settings
        Config will take all information about base model from config/config.yaml and params.yaml
        
        Args:
            config: Contains all model settings like image size, weights, paths, etc.
        """
        self.config = config 
    
    def get_base_model(self):
        """
        Download the pre-trained VGG16 model from Keras
        This model has already been trained on ImageNet (1 million+ images)
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            # Image dimensions: (height, width, channels) e.g., (224, 224, 3) for RGB
            input_shape=self.config.params_image_size,
            
            # 'imagenet' means use pre-trained weights, None means random initialization
            weights=self.config.params_weight,
            
            # include_top=False removes the original classification head (last layers)
            # We do this to add our own custom classifier for our specific task
            include_top=self.config.params_include_top
        )

        # Save the downloaded base model to disk for future use
        # Path comes from config/config.yaml -> base_model_path
        self.save_model(path=self.config.base_model_path, model=self.model)
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Customize the VGG16 model for our specific classification task
        
        Args:
            model: The base VGG16 model
            classes: Number of categories we want to classify (e.g., 5 for cat/dog/bird/fish/rabbit)
            freeze_all: If True, don't update ANY layers during training (only train new layers)
            freeze_till: Freeze layers except the last 'n' layers (e.g., 5 means unfreeze last 5)
            learning_rate: How big steps the model takes during learning (e.g., 0.001)
        """
        
        # STEP 1: Decide which layers to freeze (not update during training)
        if freeze_all:
            # Lock ALL base model layers - we only train the new classifier we'll add
            # Good when you have little data and trust the pre-trained features
            for layer in model.layers:
                layer.trainable = False 
                
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze all layers EXCEPT the last 'freeze_till' layers
            # Example: if freeze_till=5, freeze everything except last 5 layers
            # This is called "fine-tuning" - adjust some layers to your data
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False 

        # STEP 2: Add new layers for our custom classification task
        
        # Flatten: Convert 3D feature maps to 1D vector
        # Example: (7, 7, 512) becomes (25088,) - one long list of numbers
        flatten_in = tf.keras.layers.Flatten()(model.output)

        # Dense: Our custom classification layer
        # Creates output neurons equal to number of classes
        # Softmax converts outputs to probabilities (all add up to 1.0)
        # Example: [0.7, 0.2, 0.1] means 70% sure it's class 1
        prediction = tf.keras.layers.Dense(
            units=classes,  # Number of output neurons = number of categories
            activation="softmax"  # Converts raw scores to probabilities
        )(flatten_in)

        # STEP 3: Combine base model + new layers into complete model
        # Creates the full pipeline: Input → VGG16 → Flatten → Dense → Output
        full_model = tf.keras.models.Model(
            inputs=model.input,  # Use VGG16's input layer
            outputs=prediction    # Use our new prediction layer as output
        )

        # STEP 4: Configure how the model will learn
        full_model.compile(
            # SGD = Stochastic Gradient Descent (learning algorithm)
            # Learning rate controls how fast the model learns
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            
            # Loss function: measures how wrong predictions are
            # CategoricalCrossentropy is standard for multi-class problems
            loss=tf.keras.losses.CategoricalCrossentropy(),
            
            # Metrics: what to track during training (accuracy = % correct)
            metrics=["accuracy"]
        )
        
        # Print model architecture: shows all layers, parameters, and connections
        # Helpful for debugging and understanding model structure
        full_model.summary()

        # Return the complete, ready-to-train model
        return full_model
    
    def update_base_model(self):

        # Initialize full model 
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)


    @staticmethod
    def save_model(path: Path, model: tf.keras.models):
            model.save(path)