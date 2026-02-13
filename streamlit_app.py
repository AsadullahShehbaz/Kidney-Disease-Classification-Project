"""
Simple Streamlit Demo for Brain Tumor Detection
Upload an image and get instant prediction!
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Configure the page
st.set_page_config(
    page_title="Kidney Disease Detector",
    page_icon="🧠",
    layout="centered"
)

# Title and description
st.title("🧠 Kidney Disease Tumor Detection")
st.write("Upload a Kidney Disease scan image to detect if it contains a tumor or is normal")

# Load model (with caching to load only once)
@st.cache_resource
def load_my_model():
    """Load the trained model - only runs once"""
    return load_model(os.path.join("model", "model.keras"))

# File uploader
uploaded_file = st.file_uploader(
    "Choose a brain scan image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Add a predict button
    if st.button("🔍 Predict", type="primary"):
        with st.spinner("Analyzing image..."):
            # Save uploaded file temporarily
            temp_path = "temp_image.jpg"
            img.save(temp_path)
            
            # Load and preprocess image
            test_image = image.load_img(temp_path, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            # Load model and predict
            model = load_my_model()
            result = np.argmax(model.predict(test_image), axis=1)
            
            # Clean up temp file
            os.remove(temp_path)
            
            # Show results
            st.markdown("---")
            if result[0] == 1:
                st.error("⚠️ Prediction: **TUMOR DETECTED**")
                st.write("The model detected signs of a tumor in the scan.")
            else:
                st.success("✅ Prediction: **NORMAL**")
                st.write("The model indicates this is a normal brain scan.")
            
            # Show confidence (optional)
            st.info("💡 **Note**: This is an AI prediction for educational purposes. Always consult medical professionals for real diagnosis.")

else:
    # Show instructions when no file is uploaded
    st.info("👆 Please upload a kidney disease scan image to get started")
    
    # Optional: Add example or demo section
    with st.expander("ℹ️ How to use"):
        st.write("""
        1. Click 'Browse files' above
        2. Select a brain scan image (JPG, JPEG, or PNG)
        3. Click the 'Predict' button
        4. View the AI's prediction!
        """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")