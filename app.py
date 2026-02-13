# =============================================================================
# IMPORTS SECTION - Bringing in the tools we need
# =============================================================================

# FastAPI core imports - The main framework for building our web API
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
# - FastAPI: Main application class
# - UploadFile: Handles file uploads efficiently (images in our case)
# - File: Marks a parameter as a file upload in the API
# - Request: Contains information about incoming HTTP requests
# - HTTPException: Used to return error responses (like 400, 500)

# Response classes - Different ways to send data back to the user
from fastapi.responses import JSONResponse, HTMLResponse
# - JSONResponse: Sends data in JSON format {"key": "value"}
# - HTMLResponse: Sends HTML pages (like a website)

# Template and static file handling
from fastapi.templating import Jinja2Templates  # Renders HTML with dynamic data
from fastapi.staticfiles import StaticFiles     # Serves CSS, JS, images

# File and path operations
from pathlib import Path  # Modern way to handle file paths (better than strings)
import shutil            # File operations (copy, move, delete)
import uuid              # Generates unique IDs (prevents filename conflicts)
import os                # Operating system operations

# Custom ML pipeline - Your trained CNN model
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier import logger  # Logs events (like print but better for production)


# =============================================================================
# APP INITIALIZATION - Setting up the FastAPI application
# =============================================================================

# Create FastAPI app instance
# This is the main object that handles all requests and responses
app = FastAPI(
    title="Kidney Disease Classifier CNN API",           # Shows in API docs at /docs
    description="FastAPI backend for CNN-based kidney tumor classification",
    version="1.0"  # API version number
)

# Get the current file's directory path
# Example: If this file is at /home/user/project/api/main.py
# Then BASE_DIR will be /home/user/project/api
BASE_DIR = Path(__file__).resolve().parent
# __file__ = current file path
# .resolve() = converts to absolute path (full path from root)
# .parent = gets the parent directory

# Create uploads folder path
# The "/" operator combines paths: BASE_DIR + "uploads"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create the uploads directory if it doesn't exist
# parents=True: Create parent folders if needed
# exist_ok=True: Don't throw error if folder already exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Setup Jinja2 for rendering HTML templates
# This looks for HTML files in the "templates" folder
# Example: templates/index.html, templates/about.html
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files (CSS, JavaScript, images)
# This makes files in "static" folder accessible via /static/ URL
# Example: static/style.css becomes accessible at http://localhost:8000/static/style.css
app.mount(
    "/static",                                    # URL prefix
    StaticFiles(directory=str(BASE_DIR / "static")),  # Folder to serve
    name="static"                                 # Internal name for this mount
)


# =============================================================================
# ROUTES - URL endpoints that users can access
# =============================================================================

# -------------------------
# Route 1: Home Page
# -------------------------
@app.get("/", response_class=HTMLResponse)
# @app.get = Handle GET requests (when user visits URL in browser)
# "/" = Root URL (http://localhost:8000/)
# response_class=HTMLResponse = Tell FastAPI we're returning HTML, not JSON

async def home(request: Request):
    """
    Renders the main HTML page with the upload form
    
    Args:
        request: Contains info about the HTTP request (URL, headers, etc.)
    
    Returns:
        HTMLResponse with rendered index.html template
    """
    logger.info("Rendering Home Page")  # Log this event for debugging
    
    # Render and return the HTML template
    return templates.TemplateResponse(
        "index.html",           # Template file name (from templates folder)
        {"request": request}    # Pass request object to template (required by Jinja2)
    )


# -------------------------
# Route 2: Image Prediction API
# -------------------------
@app.post("/predict")
# @app.post = Handle POST requests (when form submits data)
# "/predict" = URL endpoint (http://localhost:8000/predict)

async def predict_image(file: UploadFile = File(...)):
    """
    Accepts an image file, saves it, and returns CNN prediction
    
    Args:
        file: Uploaded image file (UploadFile object)
              File(...) means this parameter is REQUIRED
    
    Returns:
        JSON response with prediction result or error message
    """
    try:
        # Log that we received a prediction request
        logger.info("Prediction request received")

        # ----------------
        # Step 1: Validate file type (security check)
        # ----------------
        # Check if uploaded file is actually an image
        # content_type examples: "image/jpeg", "image/png", "application/pdf"
        if not file.content_type.startswith("image/"):
            # If not an image, return 400 Bad Request error
            raise HTTPException(
                status_code=400,  # HTTP status code (400 = Bad Request)
                detail="Only image files are allowed"  # Error message
            )

        # ----------------
        # Step 2: Generate unique filename
        # ----------------
        # uuid.uuid4() generates a random unique ID
        # Example: "a3f2e9d1-4b5c-6789-scan.jpg"
        # This prevents problems if two users upload files with same name
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        
        # Create full file path: UPLOAD_DIR + unique_filename
        # Example: /home/user/project/api/uploads/a3f2e9d1-4b5c-6789-scan.jpg
        file_path = UPLOAD_DIR / unique_filename

        # ----------------
        # Step 3: Save uploaded file to disk
        # ----------------
        # Open file in write-binary mode ("wb")
        with file_path.open("wb") as buffer:
            # Copy uploaded file to our server in chunks (memory efficient)
            # file.file = the uploaded file stream
            # buffer = our destination file
            shutil.copyfileobj(file.file, buffer)
            # This reads and writes in small chunks (like 64KB at a time)
            # Good for large files - doesn't load entire file into memory

            logger.info(f"File saved at: {file_path}")
            # ----------------
            # Step 4: Run ML prediction
            # ----------------
            # Initialize the CNN prediction pipeline with saved image path
            predictor = PredictionPipeline(str(file_path))

            # Run prediction (returns array like: [{'image': 'Tumor'}])
            prediction = predictor.predict()

            logger.info(f"Prediction successful: {prediction}")

            # ✅ FIX: Extract simple string from prediction result
            # Convert [{'image': 'Tumor'}] → "Tumor"
            prediction_result = prediction
            if isinstance(prediction, list) and len(prediction) > 0:
                # Extract from array
                if isinstance(prediction[0], dict):
                    prediction_result = prediction[0].get('image', str(prediction[0]))
                else:
                    prediction_result = str(prediction[0])
            elif isinstance(prediction, dict):
                prediction_result = prediction.get('image', str(prediction))
            else:
                prediction_result = str(prediction)

            logger.info(f"Extracted prediction: {prediction_result}")

            # ----------------
            # Step 5: Return success response
            # ----------------
            return JSONResponse(
                content={
                    "status": "success",
                    "prediction": prediction_result  # Now returns: "Tumor" or "Normal"
                }
            )

    except Exception as e:
        raise HTTPException(
             status_code=500,
             detail={
            "status": "error",
            "message": "Internal Server Error",
            "error": str(e)
        }

        )            
# =============================================================================
# SERVER STARTUP - Run the application
# =============================================================================

if __name__ == "__main__":
    # This block only runs when you execute this file directly
    # Not when importing it as a module
    
    import uvicorn  # ASGI server (runs FastAPI apps)
    
    # Note: You're creating a NEW FastAPI instance here (bug!)
    # You should use the 'app' variable defined above instead
    # app = FastAPI()  # ❌ Remove this line
    
    # Run the server
    uvicorn.run(
        "app:app",                    # The FastAPI app to run
        host="0.0.0.0",        # Listen on all network interfaces (accessible from other devices)
                               # Use "127.0.0.1" for localhost only
        port=8000        # Port number (access at http://localhost:8000)           # Auto-restart server when code changes (useful for development)
                               # Set to False in production
    )
    
    # Alternative ways to run:
    # 1. Command line: uvicorn main:app --reload
    # 2. Command line: python main.py


# =============================================================================
# HOW TO TEST THIS API
# =============================================================================
"""
1. Start the server:
   python main.py

2. Open browser and go to:
   http://localhost:8000/          → See the upload form
   http://localhost:8000/docs      → Interactive API documentation (Swagger UI)
   http://localhost:8000/redoc     → Alternative API documentation

3. Test prediction:
   - Upload an image through the web form
   - Or use curl: curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
   - Or use Postman to send POST request with file

4. Expected folder structure:
   project/
   ├── main.py              ← This file
   ├── uploads/             ← Uploaded images saved here
   ├── templates/
   │   └── index.html       ← HTML upload form
   ├── static/
   │   ├── style.css        ← CSS styles
   │   └── script.js        ← JavaScript code
   └── cnnClassifier/       ← Your ML model code
       └── pipeline/
           └── prediction.py
"""