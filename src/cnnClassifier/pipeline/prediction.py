"""
src/cnnClassifier/pipeline/predictions.py
"""
import numpy as np  # Library for numerical operations (like arrays and math)
from tensorflow.keras.models import load_model  # Import function to load trained models
from tensorflow.keras.preprocessing import image  # Tools to process images for our model
import os  # Library to work with file paths and directories


class PredictionPipeline:
    """
    A simple class to predict if a brain scan image shows a tumor or is normal.
    Think of it like a doctor that looks at X-ray images and gives a diagnosis!
    """
    
    def __init__(self, filename):
        # Constructor - runs when we create a new PredictionPipeline object
        # filename is the path to the image we want to analyze
        self.filename = filename 
    
    def predict(self):
        # Main prediction method - this does all the magic!
        
        # Step 1: Load our trained model from the saved location
        # The model was trained earlier and saved in artifacts/training/model.keras
        model = load_model(os.path.join("model", "model.keras"))
        
        # Step 2: Prepare the image for prediction (same size model expects)
        imagename = self.filename  # Get the image path
        test_image = image.load_img(imagename, target_size=(224, 224))
        # Load image and resize to 224x224 pixels (standard size for this model)
        
        test_image = image.img_to_array(test_image)
        # Convert image to numerical array that model can understand
        
        test_image = np.expand_dims(test_image, axis=0)
        # Add batch dimension - model expects [1, 224, 224, 3] shape, not [224, 224, 3]
        
        # Step 3: Get model's prediction
        result = np.argmax(model.predict(test_image), axis=1)
        # model.predict() gives probabilities like [0.2, 0.8]
        # argmax picks the highest value index: 0=Normal, 1=Tumor
        print(result)  # Debug print to see raw prediction (0 or 1)
        
        # Step 4: Convert number to human-readable result
        if result[0] == 1:
            prediction = 'Tumor'
            return [{"image": prediction}]  # Return tumor prediction
        
        else:
            prediction = 'Normal'
            return [{"image": prediction}]  # Return normal prediction
