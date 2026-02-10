from cnnClassifier.entitiy.config_entity import TrainingConfig
import os 
import urllib.request as request 
from zipfile import ZipFile 
from pathlib import Path
import tensorflow as tf 
import time 

class Training:
    """
    Handles the complete training process for the kidney disease classifier
    
    This class takes care of:
    1. Loading the prepared VGG16 model
    2. Creating data generators for training and validation
    3. Training the model on kidney CT scan images
    4. Saving the trained model 
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the Training class with configuration settings
        
        Args:
            config: TrainingConfig object containing all training parameters like:
                   - paths to data and model
                   - epochs, batch size
                   - image size
                   - augmentation settings
        """
        self.config = config 
    
    def get_base_model(self):
        """
        Load the prepared VGG16 model from disk
        
        This model was created in the previous stage (prepare_base_model)
        It already has:
        - Pre-trained VGG16 layers (frozen)
        - Custom classification head added on top
        - Compiled and ready for training
        
        The model is loaded from: artifacts/prepare_base_model/base_model_updated.h5
        """
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        """
        Create data generators for feeding images to the model during training
        
        Data generators:
        - Load images from disk in batches (not all at once - saves memory!)
        - Apply preprocessing (normalization, resizing)
        - Apply data augmentation to training set (increases dataset variety)
        - Split data into training (80%) and validation (20%) sets
        
        Why use generators?
        - Memory efficient: Only loads one batch at a time
        - Real-time augmentation: Creates new variations on-the-fly
        - Automatic batching: No need to manually create batches
        """
        
        # ==================== COMMON SETTINGS FOR BOTH GENERATORS ====================
        
        # Dictionary of settings applied to BOTH training and validation generators
        datagenerator_kwargs = dict(
            # Rescale pixel values from [0, 255] to [0, 1]
            # Neural networks work better with normalized inputs
            # Example: pixel value 255 becomes 1.0, value 127 becomes 0.498
            rescale = 1./255,
            
            # Split dataset: 80% training, 20% validation
            # Validation set is used to check if model is overfitting
            validation_split = 0.20
        )

        # Dictionary of settings for how images flow through the generators
        dataflow_kwargs = dict(
            # Resize all images to (224, 224) - VGG16's required input size
            # [:-1] removes the last element (channels), so [224, 224, 3] becomes [224, 224]
            target_size = self.config.params_image_size[:-1],
            
            # Number of images to load in each batch
            # Smaller batch = less memory, but noisier training
            # Larger batch = more memory, but stabler training
            # 16 is a good balance for most systems
            batch_size = self.config.params_batch_size,
            
            # Method for resizing images
            # "bilinear" = smooth interpolation, good quality
            # Other options: "nearest", "bicubic"
            interpolation = "bilinear"
        )

        # ==================== VALIDATION GENERATOR ====================
        
        # Create validation data generator
        # Validation images get NO augmentation - we want to test on real, unchanged images
        # Only rescaling is applied (normalize to [0, 1])
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs  # ** unpacks the dictionary into keyword arguments
        )

        # Create the actual generator that loads validation images
        self.valid_generator = valid_datagenerator.flow_from_directory(
            # Path to data folder containing Normal/ and Tumor/ subfolders
            directory=self.config.training_data,
            
            # Use the validation split (20% of data)
            subset="validation",
            
            # Don't shuffle validation data - we want consistent evaluation
            # Shuffling would give slightly different accuracy each time
            shuffle=False, 
            
            # Apply the dataflow settings (target_size, batch_size, etc.)
            **dataflow_kwargs
        )
        # Output example: "Found 40 images belonging to 2 classes."

        # ==================== TRAINING GENERATOR ====================
        
        # Check if data augmentation is enabled in params.yaml
        if self.config.params_is_augmentation:
            # DATA AUGMENTATION: Create artificial variations of training images
            # This helps prevent overfitting and makes model more robust
            
            # Why augmentation?
            # - Increases effective dataset size (1000 images → 10,000+ variations)
            # - Model learns to recognize kidneys from different angles, positions
            # - Prevents memorization of training images
            
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                # Randomly rotate images up to 40 degrees left or right
                # Example: Original image → Rotated 15° clockwise
                # Helps model recognize kidneys at different orientations
                rotation_range=40,
                
                # Randomly flip images horizontally (left ↔ right)
                # Example: Kidney on left side → Kidney on right side
                # Medical images can be from either kidney!
                horizontal_flip=True,
                
                # Randomly shift image horizontally by up to 20% of width
                # Example: 224px image → shift up to 45 pixels left/right
                # Helps model handle kidneys not perfectly centered
                width_shift_range=0.2,
                
                # Randomly shift image vertically by up to 20% of height
                # Example: 224px image → shift up to 45 pixels up/down
                height_shift_range=0.2,
                
                # Apply shear transformation (slanting effect)
                # Example: Rectangle → Parallelogram
                # Range: 0.2 means up to 20% shear
                # Helps with images taken at angles
                shear_range=0.2,
                
                # Randomly zoom in/out by up to 20%
                # Example: Zoom in 10% → closer view of kidney
                # Helps model recognize kidneys at different scales
                zoom_range=0.2,
                
                # Also apply rescaling and validation_split
                **datagenerator_kwargs
            )
            
            # IMPORTANT: Each image is augmented RANDOMLY each epoch
            # Same image looks different every time it's fed to the model!
            # This creates essentially unlimited training variations
            
        else:
            # If augmentation is disabled, use same generator as validation
            # Training images only get rescaled, no transformations
            # Useful for: debugging, fast prototyping, or very large datasets
            train_datagenerator = valid_datagenerator
    
        # Create the actual generator that loads training images
        self.train_generator = train_datagenerator.flow_from_directory(
            # Path to data folder
            directory=self.config.training_data,
            
            # Use the training split (80% of data)
            subset="training",
            
            # Shuffle training data - important for good learning!
            # Model sees images in different order each epoch
            # Prevents learning based on image order
            shuffle=True,
            
            # Apply the dataflow settings
            **dataflow_kwargs
        )
        # Output example: "Found 160 images belonging to 2 classes."
        
        # FINAL RESULT:
        # self.train_generator: Loads augmented training images in batches
        # self.valid_generator: Loads original validation images in batches

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """
        Save the trained Keras model to disk
        
        @staticmethod means this method doesn't need access to self
        It's a utility function that can be called independently
        
        Args:
            path: Where to save the model (e.g., artifacts/training/model.h5)
            model: The trained Keras model object
        
        Saved format: HDF5 (.h5 file)
        Contains:
        - Model architecture (layers, connections)
        - Trained weights (learned parameters)
        - Optimizer state (for resuming training)
        - Compilation settings (loss function, metrics)
        
        This .h5 file can be loaded later for predictions or further training
        """
        model.save(path)

    # def train(self, callback_list: list):
    def train(self):
        """
        Train the VGG16 model on kidney CT scan images
        
        This is the main training loop that:
        1. Feeds images to the model batch by batch
        2. Model makes predictions
        3. Calculates loss (how wrong predictions are)
        4. Updates weights to improve
        5. Repeats for multiple epochs
        6. Saves the trained model
        
        Args:
            callback_list: List of Keras callbacks (e.g., early stopping, model checkpointing)
                          Callbacks are functions called during training to:
                          - Save best model
                          - Stop if not improving
                          - Log metrics to MLflow
                          - Reduce learning rate
        """
        
        # ==================== CALCULATE TRAINING STEPS ====================
        
        # Calculate how many batches (steps) in one epoch for TRAINING data
        # Formula: total_images // batch_size
        # Example: 160 training images, batch_size=16 → 160 // 16 = 10 steps per epoch
        # 
        # Why // (floor division)?
        # - If 165 images and batch_size=16 → 165 // 16 = 10 steps
        # - Last 5 images are dropped (can't form complete batch)
        # - This ensures all batches have exactly the same size
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size 
        
        # Calculate how many batches (steps) for VALIDATION data
        # Example: 40 validation images, batch_size=16 → 40 // 16 = 2 steps
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size 
        
        # ==================== TRAIN THE MODEL ====================
        
        # model.fit() is the main training function
        # It runs the entire training loop:
        # For each epoch:
        #   For each training batch:
        #     1. Forward pass: predict → calculate loss
        #     2. Backward pass: calculate gradients
        #     3. Update weights
        #   For each validation batch:
        #     1. Predict (no weight updates)
        #     2. Calculate metrics
        
        self.model.fit(
            # Training data generator
            # Provides batches of (images, labels) automatically
            # Example batch: 16 images of shape (224, 224, 3) + 16 labels [0 or 1]
            self.train_generator,
            
            # Number of times to iterate over the ENTIRE dataset
            # Example: epochs=10 means model sees each image 10 times
            # (Actually more due to augmentation - each time looks different!)
            # 
            # What happens each epoch:
            # Epoch 1: Model is terrible (random weights)
            # Epoch 5: Model is learning patterns
            # Epoch 10: Model is good at recognizing kidneys
            # Epoch 20+: Might start overfitting (memorizing instead of learning)
            epochs = self.config.params_epochs,
            
            # Number of batches to process in one epoch
            # We calculated this above: total_training_images // batch_size
            # Example: 10 steps means model processes 10 batches per epoch
            steps_per_epoch = self.steps_per_epoch,
            
            # Number of validation batches to process after each epoch
            # Used to check model performance on unseen data
            # Example: 2 steps means process 2 batches of validation data
            validation_steps = self.validation_steps,
            
            # Validation data generator
            # After each epoch, model is evaluated on this data
            # This gives us validation accuracy and loss
            # If validation loss increases → model is overfitting!
            validation_data = self.valid_generator,
            
            # Callbacks are executed at specific points during training
            # Common callbacks:
            # - ModelCheckpoint: Save model when validation accuracy improves
            # - EarlyStopping: Stop training if no improvement for N epochs
            # - TensorBoard: Log metrics for visualization
            # - MLflowCallback: Log experiments to MLflow
            # - ReduceLROnPlateau: Reduce learning rate if stuck
            # callbacks = callback_list
        )
        
        # TRAINING OUTPUT EXAMPLE:
        # Epoch 1/10
        # 10/10 [======] - 45s - loss: 0.6932 - accuracy: 0.5000 - val_loss: 0.6931 - val_accuracy: 0.5000
        # Epoch 2/10  
        # 10/10 [======] - 42s - loss: 0.5234 - accuracy: 0.7500 - val_loss: 0.4523 - val_accuracy: 0.8000
        # ...
        # Epoch 10/10
        # 10/10 [======] - 41s - loss: 0.1234 - accuracy: 0.9500 - val_loss: 0.1567 - val_accuracy: 0.9250
        
        # ==================== SAVE TRAINED MODEL ====================
        
        # After training is complete, save the final model
        # This model now has updated weights and can classify kidney images
        # File size: ~100-500 MB (VGG16 is large!)
        self.save_model(
            path = self.config.trained_model_path,  # artifacts/training/model.h5
            model = self.model  
        )
        
        # Success! The model is now trained and saved
        # Next steps:
        # 1. Load this model for evaluation
        # 2. Test on new kidney images
        # 3. Deploy as web app (app.py)