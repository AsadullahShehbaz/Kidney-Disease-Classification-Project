# 🏥 Kidney Disease Classification - Deep Learning Project
### Complete Guide for Beginners | VGG16 CNN Classifier with MLflow & DVC

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Project Architecture](#-project-architecture)
3. [Directory Structure Explained](#-directory-structure-explained)
4. [Step-by-Step Workflow](#-step-by-step-workflow)
5. [Installation & Setup](#-installation--setup)
6. [Understanding Each Component](#-understanding-each-component)
7. [MLflow Integration](#-mlflow-integration)
8. [DVC Integration](#-dvc-integration)
9. [AWS Deployment](#-aws-deployment)
10. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

### What Does This Project Do?
This project classifies kidney CT scan images into two categories:
- **Normal Kidney** 🟢
- **Diseased Kidney (Tumor/Stone/Cyst)** 🔴

### Technologies Used
- **Deep Learning**: TensorFlow/Keras with VGG16 (Transfer Learning)
- **Experiment Tracking**: MLflow (tracks model performance)
- **Pipeline Management**: DVC (Data Version Control)
- **Deployment**: AWS EC2 + Docker
- **Web Interface**: Flask application

### Why This Structure?
This is a **production-grade** ML project structure used by companies like:
- Google, Amazon, Microsoft
- AI startups and consulting firms
- Professional freelance projects

**NOT just a Jupyter notebook!** This is how **real AI Engineers** build deployable systems.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER UPLOADS KIDNEY IMAGE                     │
│                            (Web UI)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK APP (app.py)                            │
│              Receives image → Processes → Predicts               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 TRAINED VGG16 MODEL                              │
│        (artifacts/training/model.h5)                             │
│                                                                  │
│  Input: 224x224 RGB Image                                       │
│  Output: [probability_normal, probability_diseased]              │
└─────────────────────────────────────────────────────────────────┘

                    HOW WAS THIS MODEL CREATED?
                               ↓

┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
│                                                                  │
│  Stage 1: Data Ingestion                                        │
│  ├── Download dataset from Drive/S3                             │
│  └── Extract and organize images                                │
│                                                                  │
│  Stage 2: Prepare Base Model                                    │
│  ├── Load pre-trained VGG16                                     │
│  ├── Remove top layers                                          │
│  └── Add custom classification head                             │
│                                                                  │
│  Stage 3: Model Training                                        │
│  ├── Load training data                                         │
│  ├── Data augmentation                                          │
│  ├── Train model with fine-tuning                               │
│  └── Save trained model                                         │
│                                                                  │
│  Stage 4: Model Evaluation                                      │
│  ├── Test on validation set                                     │
│  ├── Calculate accuracy, loss                                   │
│  └── Log results to MLflow                                      │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 EXPERIMENT TRACKING                              │
│                                                                  │
│  MLflow: Logs metrics, parameters, models                       │
│  DVC: Tracks data versions and pipeline stages                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Directory Structure Explained

```
Kidney-Disease-Classification/
│
├── 📁 .github/                          # GitHub Actions (CI/CD automation)
│   └── workflows/
│       ├── .gitkeep                     # Keeps empty folder in Git
│       └── main.yaml                    # Auto-deployment workflow
│
├── 📁 config/                           # ⚙️ CONFIGURATION FILES
│   ├── config.yaml                      # Paths & directories (WHERE things are)
│   └── secrets.yaml                     # API keys, passwords (optional)
│
├── 📁 research/                         # 🧪 EXPERIMENTATION ZONE
│   ├── 01_data_ingestion.ipynb         # Test data download
│   ├── 02_prepare_base_model.ipynb     # Test VGG16 setup
│   ├── 03_model_training.ipynb         # Test training process
│   └── 04_model_evaluation.ipynb       # Test evaluation
│   
│   # 💡 Workflow: Experiment here → Then convert to modular code
│
├── 📁 src/cnnClassifier/               # 🧠 MAIN APPLICATION CODE
│   │
│   ├── 📄 __init__.py                  # Makes this a Python package
│   │
│   ├── 📁 components/                  # 🔧 BUILDING BLOCKS (Core Logic)
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          # Downloads kidney CT images
│   │   ├── prepare_base_model.py      # Sets up VGG16 architecture
│   │   ├── model_training.py          # Trains the CNN
│   │   └── model_evaluation.py        # Tests model accuracy
│   │   
│   │   # Each file = One responsibility
│   │   # Example: If download fails, check data_ingestion.py only
│   │
│   ├── 📁 utils/                       # 🛠️ HELPER FUNCTIONS (Reusable Tools)
│   │   ├── __init__.py
│   │   └── common.py                  # read_yaml(), save_json(), create_directories()
│   │   
│   │   # Used by ALL components
│   │   # Write once, use everywhere!
│   │
│   ├── 📁 config/                      # 🎛️ CONFIGURATION MANAGER
│   │   ├── __init__.py
│   │   └── configuration.py           # Reads config.yaml & params.yaml
│   │   
│   │   # Central brain: "What settings does each component need?"
│   │
│   ├── 📁 pipeline/                    # 🔄 COMPLETE WORKFLOWS
│   │   ├── __init__.py
│   │   ├── stage_01_data_ingestion.py          # Pipeline: Download data
│   │   ├── stage_02_prepare_base_model.py      # Pipeline: Setup model
│   │   ├── stage_03_model_training.py          # Pipeline: Train model
│   │   ├── stage_04_model_evaluation.py        # Pipeline: Evaluate
│   │   └── predict.py                          # Prediction pipeline
│   │   
│   │   # Connects components into end-to-end processes
│   │
│   ├── 📁 entity/                      # 📋 DATA BLUEPRINTS
│   │   ├── __init__.py
│   │   └── config_entity.py           # DataIngestionConfig, TrainingConfig, etc.
│   │   
│   │   # Defines structure: "What data does each component expect?"
│   │
│   └── 📁 constants/                   # 🔒 FIXED VALUES (Never Change)
│       └── __init__.py                # CONFIG_FILE_PATH, PARAMS_FILE_PATH
│
├── 📁 templates/                        # 🌐 WEB INTERFACE
│   └── index.html                      # Upload image → Get prediction
│
├── 📁 artifacts/                        # 💾 GENERATED OUTPUTS (Git ignored)
│   ├── data_ingestion/                 # Downloaded & extracted data
│   ├── prepare_base_model/             # VGG16 base model files
│   ├── training/                       # Trained model (model.h5)
│   └── evaluation/                     # Evaluation results (scores.json)
│   
│   # Created automatically during training
│   # Not uploaded to GitHub (too large)
│
├── 📄 config.yaml                       # ⚙️ PATHS & DIRECTORIES
├── 📄 params.yaml                       # 🎚️ MODEL HYPERPARAMETERS
├── 📄 dvc.yaml                          # 📊 DVC PIPELINE DEFINITION
├── 📄 requirements.txt                  # 📦 PYTHON DEPENDENCIES
├── 📄 setup.py                          # 📦 MAKES PROJECT INSTALLABLE
├── 📄 main.py                           # ▶️ TRAINING ENTRY POINT
├── 📄 app.py                            # 🌐 FLASK WEB APP
├── 📄 Dockerfile                        # 🐳 DOCKER IMAGE RECIPE
├── 📄 .dvcignore                        # Ignore files for DVC
├── 📄 .gitignore                        # Ignore files for Git
└── 📄 README.md                         # 📖 THIS FILE!
```

---

## 🔄 Step-by-Step Workflow

### **Complete Development Workflow:**

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: SETUP PROJECT STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────┤
│ Run: python template.py                                         │
│ Creates all folders and files automatically                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: DEFINE CONFIGURATIONS                                   │
├─────────────────────────────────────────────────────────────────┤
│ Edit config/config.yaml:                                        │
│   - Where to store data? (artifacts/data_ingestion)             │
│   - Where to save model? (artifacts/training/model.h5)          │
│                                                                  │
│ Edit params.yaml:                                               │
│   - Image size: [224, 224, 3]                                   │
│   - Learning rate: 0.001                                        │
│   - Epochs: 10                                                  │
│   - Batch size: 16                                              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: EXPERIMENT IN JUPYTER                                   │
├─────────────────────────────────────────────────────────────────┤
│ research/01_data_ingestion.ipynb:                               │
│   - Test downloading dataset                                    │
│   - Test extraction                                             │
│   - Verify data structure                                       │
│                                                                  │
│ Once working → Convert to component!                            │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: CREATE DATA STRUCTURES                                  │
├─────────────────────────────────────────────────────────────────┤
│ Edit entity/config_entity.py:                                   │
│                                                                  │
│ @dataclass                                                      │
│ class DataIngestionConfig:                                      │
│     root_dir: Path                                              │
│     source_URL: str                                             │
│     local_data_file: Path                                       │
│     unzip_dir: Path                                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: UPDATE CONFIGURATION MANAGER                            │
├─────────────────────────────────────────────────────────────────┤
│ Edit config/configuration.py:                                   │
│                                                                  │
│ def get_data_ingestion_config(self):                            │
│     config = self.config.data_ingestion                         │
│     create_directories([config.root_dir])                       │
│     return DataIngestionConfig(...)                             │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: BUILD COMPONENT                                         │
├─────────────────────────────────────────────────────────────────┤
│ Create components/data_ingestion.py:                            │
│                                                                  │
│ class DataIngestion:                                            │
│     def __init__(self, config):                                 │
│         self.config = config                                    │
│                                                                  │
│     def download_file(self):                                    │
│         # Download logic                                        │
│                                                                  │
│     def extract_zip_file(self):                                 │
│         # Extraction logic                                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: CREATE PIPELINE                                         │
├─────────────────────────────────────────────────────────────────┤
│ Create pipeline/stage_01_data_ingestion.py:                     │
│                                                                  │
│ class DataIngestionTrainingPipeline:                            │
│     def main(self):                                             │
│         config = ConfigurationManager()                         │
│         data_config = config.get_data_ingestion_config()        │
│         data_ingestion = DataIngestion(data_config)             │
│         data_ingestion.download_file()                          │
│         data_ingestion.extract_zip_file()                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: UPDATE MAIN.PY                                          │
├─────────────────────────────────────────────────────────────────┤
│ Edit main.py:                                                   │
│                                                                  │
│ STAGE_NAME = "Data Ingestion"                                   │
│ try:                                                            │
│     logger.info(f">>>>> stage {STAGE_NAME} started")            │
│     pipeline = DataIngestionTrainingPipeline()                  │
│     pipeline.main()                                             │
│     logger.info(f">>>>> stage {STAGE_NAME} completed")          │
│ except Exception as e:                                          │
│     logger.exception(e)                                         │
│     raise e                                                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: REPEAT FOR ALL STAGES                                   │
├─────────────────────────────────────────────────────────────────┤
│ - Stage 2: Prepare Base Model                                   │
│ - Stage 3: Model Training                                       │
│ - Stage 4: Model Evaluation                                     │
│                                                                  │
│ Each follows same pattern: config → entity → component →        │
│ pipeline → main.py                                              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 10: SETUP DVC PIPELINE                                     │
├─────────────────────────────────────────────────────────────────┤
│ Edit dvc.yaml to define stage dependencies                      │
│ Run: dvc repro (executes entire pipeline)                       │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 11: CREATE WEB APP                                         │
├─────────────────────────────────────────────────────────────────┤
│ Edit app.py (Flask application)                                 │
│ Create prediction endpoint                                      │
│ Test locally                                                    │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 12: DEPLOY TO AWS                                          │
├─────────────────────────────────────────────────────────────────┤
│ - Dockerize application                                         │
│ - Push to ECR                                                   │
│ - Deploy on EC2                                                 │
│ - Setup CI/CD with GitHub Actions                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 💻 Installation & Setup

### **Prerequisites:**
- Python 3.8 or higher
- Git installed
- Anaconda/Miniconda (recommended)
- 4GB+ RAM
- Internet connection

### **Step 1: Clone Repository**
```bash
git clone https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project
cd Kidney-Disease-Classification-Deep-Learning-Project
```

### **Step 2: Create Virtual Environment**

**Option A: Using Conda (Recommended)**
```bash
# Create environment
conda create -n kidney_classifier python=3.8 -y

# Activate environment
conda activate kidney_classifier
```

**Option B: Using venv**
```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**What gets installed?**
```txt
tensorflow==2.12.0          # Deep learning framework
pandas==2.0.3              # Data manipulation
numpy==1.24.3              # Numerical computing
matplotlib==3.7.1          # Visualization
Flask==2.3.2               # Web framework
mlflow==2.5.0              # Experiment tracking
dvc==3.15.0                # Data version control
PyYAML==6.0                # YAML file reading
python-box==7.0.1          # Dictionary with dot notation
ensure==1.0.2              # Type validation
```

### **Step 4: Initialize DVC**
```bash
dvc init
```

This creates:
- `.dvc/` folder (DVC configuration)
- `.dvcignore` (files to ignore)

---

## 🧩 Understanding Each Component

### **1. config.yaml - The Address Book**

```yaml
# config/config.yaml

# Root directory for all outputs
artifacts_root: artifacts

# Stage 1: Data Ingestion Configuration
data_ingestion:
  root_dir: artifacts/data_ingestion              # Where to store downloaded data
  source_URL: https://drive.google.com/file/d/1vlhZ5c7abcdef/  # Dataset URL
  local_data_file: artifacts/data_ingestion/data.zip  # Downloaded zip location
  unzip_dir: artifacts/data_ingestion             # Where to extract

# Stage 2: Base Model Preparation
prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5  # Initial VGG16
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5  # After adding custom layers

# Stage 3: Model Training
training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5  # Final trained model

# Stage 4: Model Evaluation  
evaluation:
  root_dir: artifacts/evaluation
  mlflow_uri: https://dagshub.com/yourname/kidney-classifier.mlflow  # MLflow server
```

**💡 Why separate file?**
- Change paths without touching code
- Easy to switch between local/cloud storage
- Team members use same structure

---

### **2. params.yaml - The Control Panel**

```yaml
# params.yaml

# Image preprocessing
IMAGE_SIZE: [224, 224, 3]  # VGG16 requires 224x224 RGB images
BATCH_SIZE: 16             # Images per training batch
INCLUDE_TOP: False         # Remove VGG16's original classifier

# Pre-trained weights
WEIGHTS: imagenet          # Use ImageNet pre-trained weights
CLASSES: 2                 # Normal vs Diseased

# Training hyperparameters
EPOCHS: 10                 # Training iterations
LEARNING_RATE: 0.001       # How fast model learns
AUGMENTATION: True         # Apply data augmentation?

# Transfer learning strategy
FREEZE_ALL: True           # Freeze VGG16 layers?
FREEZE_TILL: null          # Or freeze specific number of layers
```

**💡 Experimentation made easy:**
```bash
# Try different learning rates without changing code!
# Just edit params.yaml:
LEARNING_RATE: 0.01   # Fast learning
# vs
LEARNING_RATE: 0.0001 # Slow, stable learning
```

---

### **3. Entity - Data Blueprints**

```python
# src/cnnClassifier/entity/config_entity.py

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for data ingestion stage
    frozen=True makes it immutable (can't be changed after creation)
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for VGG16 model preparation"""
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training"""
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for model evaluation"""
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
```

**💡 Why use dataclasses?**
```python
# ❌ Without dataclass (error-prone)
config = {
    "root_dir": "artifacts/data",
    "source_url": "https://..."  # Typo: source_url vs source_URL
}
# No error! But breaks later

# ✅ With dataclass (safe)
config = DataIngestionConfig(
    root_dir=Path("artifacts/data"),
    source_url="https://..."  # Error: unexpected keyword argument
)
# Catches mistakes immediately!
```

---

### **4. Configuration Manager - The Brain**

```python
# src/cnnClassifier/config/configuration.py

from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig
)

class ConfigurationManager:
    """
    Central manager for all configurations
    Reads YAML files and creates config objects
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        # Read configuration files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create root artifacts directory
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Returns configuration for data ingestion stage
        """
        config = self.config.data_ingestion
        
        # Create directory for this stage
        create_directories([config.root_dir])
        
        # Create and return config object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        
        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """Returns configuration for base model preparation"""
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )
        
        return prepare_base_model_config
    
    # Similar methods for training and evaluation configs...
```

**💡 How it works:**
```python
# Usage in pipeline:
config_manager = ConfigurationManager()
data_config = config_manager.get_data_ingestion_config()

print(data_config.source_URL)  # Access with dot notation
print(data_config.root_dir)
```

---

### **5. Components - The Workers**

#### **Component Example: Data Ingestion**

```python
# src/cnnClassifier/components/data_ingestion.py

import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    """
    Handles downloading and extracting kidney disease dataset
    """
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize with configuration
        
        Args:
            config: DataIngestionConfig object with paths and URLs
        """
        self.config = config
    
    def download_file(self):
        """
        Download dataset from Google Drive or other source
        """
        # Check if file already exists
        if not os.path.exists(self.config.local_data_file):
            logger.info("Downloading data...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"Downloaded {filename} with info:\n{headers}")
        else:
            file_size = get_size(Path(self.config.local_data_file))
            logger.info(f"File already exists. Size: {file_size}")
    
    def extract_zip_file(self):
        """
        Extract downloaded zip file
        Creates: Normal/ and Tumor/ folders with images
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted to: {unzip_path}")
```

**💡 Usage:**
```python
# In pipeline:
config = ConfigurationManager()
data_config = config.get_data_ingestion_config()
data_ingestion = DataIngestion(config=data_config)

# Execute
data_ingestion.download_file()
data_ingestion.extract_zip_file()
```

---

#### **Component Example: Prepare Base Model**

```python
# src/cnnClassifier/components/prepare_base_model.py

import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    """
    Downloads VGG16 and adds custom classification head
    """
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        """
        Download pre-trained VGG16 model from Keras
        This model was trained on ImageNet (1.4M images)
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,  # 'imagenet'
            include_top=self.config.params_include_top  # False
        )
        
        # Save base model
        self.save_model(
            path=self.config.base_model_path,
            model=self.model
        )
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Add custom layers on top of VGG16 for kidney classification
        
        Args:
            model: VGG16 base model
            classes: Number of output classes (2: Normal/Diseased)
            freeze_all: Freeze all VGG16 layers?
            freeze_till: Freeze layers except last n
            learning_rate: Training learning rate
        """
        # STEP 1: Freeze VGG16 layers (transfer learning)
        if freeze_all:
            # Don't update VGG16 weights during training
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            # Freeze all except last freeze_till layers
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False
        
        # STEP 2: Add custom classification head
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)
        
        # STEP 3: Create full model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )
        
        # STEP 4: Compile model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        
        full_model.summary()
        return full_model
    
    def update_base_model(self):
        """
        Create full model with custom head
        """
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        # Save updated model
        self.save_model(
            path=self.config.updated_base_model_path,
            model=self.full_model
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save Keras model to disk"""
        model.save(path)
```

---

#### **Component Example: Model Training**

```python
# src/cnnClassifier/components/model_training.py

import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    """
    Trains the kidney disease classifier
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_base_model(self):
        """Load the prepared model"""
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def train_valid_generator(self):
        """
        Create data generators for training and validation
        Applies data augmentation to increase dataset size
        """
        datagenerator_kwargs = dict(
            rescale=1./255,  # Normalize pixel values to [0,1]
            validation_split=0.20  # 20% for validation
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # (224, 224)
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        # Validation generator (no augmentation)
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        
        # Training generator (with augmentation if enabled)
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,      # Rotate images randomly
                horizontal_flip=True,   # Flip images horizontally
                width_shift_range=0.2,  # Shift images horizontally
                height_shift_range=0.2, # Shift images vertically
                shear_range=0.2,        # Shear transformation
                zoom_range=0.2,         # Zoom in/out
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator
        
        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )
    
    def train(self):
        """
        Train the model on kidney CT scan images
        """
        # Calculate steps per epoch
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        
        # Train model
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )
        
        # Save trained model
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
```

---

### **6. Pipeline - Connecting Everything**

```python
# src/cnnClassifier/pipeline/stage_01_data_ingestion.py

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    """
    Complete pipeline for data ingestion
    Downloads and extracts kidney disease dataset
    """
    def __init__(self):
        pass
    
    def main(self):
        """Execute data ingestion pipeline"""
        # Get configuration
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Create component
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Execute steps
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
```

---

### **7. main.py - Orchestrating All Stages**

```python
# main.py

from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import EvaluationPipeline

# STAGE 1: Data Ingestion
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# STAGE 2: Prepare Base Model
STAGE_NAME = "Prepare Base Model"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# STAGE 3: Model Training
STAGE_NAME = "Training"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# STAGE 4: Model Evaluation
STAGE_NAME = "Evaluation"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
```

---

## 📊 MLflow Integration

### **What is MLflow?**
MLflow tracks your experiments:
- What hyperparameters did you use?
- What accuracy did you get?
- Which model performed best?

### **Setup MLflow with DagsHub:**

**1. Create DagsHub Account:**
- Go to [dagshub.com](https://dagshub.com/)
- Sign up with GitHub

**2. Create Repository:**
- New Repository → Connect to GitHub repo
- DagsHub creates MLflow tracking server

**3. Get Credentials:**
```
MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/kidney-classifier.mlflow
MLFLOW_TRACKING_USERNAME=yourusername
MLFLOW_TRACKING_PASSWORD=your_token_here
```

**4. Set Environment Variables:**

**Windows (CMD):**
```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/kidney-classifier.mlflow
set MLFLOW_TRACKING_USERNAME=yourusername
set MLFLOW_TRACKING_PASSWORD=your_token
```

**Linux/Mac:**
```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/yourusername/kidney-classifier.mlflow
export MLFLOW_TRACKING_USERNAME=yourusername
export MLFLOW_TRACKING_PASSWORD=your_token
```

**5. View MLflow UI:**
```bash
# Local UI
mlflow ui

# Open browser: http://localhost:5000
```

### **How MLflow is Used in Project:**

```python
# src/cnnClassifier/components/model_evaluation.py

import mlflow
import mlflow.keras
from urllib.parse import urlparse

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluation(self):
        """Evaluate model on validation set"""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
    
    def log_into_mlflow(self):
        """Log metrics and model to MLflow"""
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config.all_params)
            
            # Log metrics
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # Log model
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model"
                )
            else:
                mlflow.keras.log_model(self.model, "model")
```

**What Gets Logged:**
- ✅ Hyperparameters (learning rate, epochs, etc.)
- ✅ Metrics (accuracy, loss)
- ✅ Model file (.h5)
- ✅ Training time
- ✅ System info

---

## 📦 DVC Integration

### **What is DVC?**
DVC (Data Version Control) is like Git for:
- Large datasets
- Model files
- ML pipelines

### **Why Use DVC?**
```
Without DVC:
❌ Can't track which data version produced which model
❌ Can't reproduce experiments
❌ Large files bloat Git repository

With DVC:
✅ Track data versions
✅ Reproduce any experiment
✅ Share data efficiently
✅ Define pipeline dependencies
```

### **DVC Commands:**

```bash
# Initialize DVC
dvc init

# Track data file
dvc add artifacts/data_ingestion/data.zip

# This creates data.zip.dvc file (small, goes in Git)
# Actual data.zip is tracked by DVC

# Define pipeline in dvc.yaml
# Then run pipeline:
dvc repro

# View pipeline graph:
dvc dag
```

### **dvc.yaml Structure:**

```yaml
# dvc.yaml

stages:
  data_ingestion:
    cmd: python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
    deps:
      - src/cnnClassifier/pipeline/stage_01_data_ingestion.py
      - config/config.yaml
    outs:
      - artifacts/data_ingestion/Kidney-ct-scan-image
  
  prepare_base_model:
    cmd: python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
    deps:
      - src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
      - config/config.yaml
    params:
      - IMAGE_SIZE
      - INCLUDE_TOP
      - CLASSES
      - WEIGHTS
      - LEARNING_RATE
    outs:
      - artifacts/prepare_base_model
  
  training:
    cmd: python src/cnnClassifier/pipeline/stage_03_model_training.py
    deps:
      - src/cnnClassifier/pipeline/stage_03_model_training.py
      - config/config.yaml
      - artifacts/data_ingestion/Kidney-ct-scan-image
      - artifacts/prepare_base_model
    params:
      - IMAGE_SIZE
      - EPOCHS
      - BATCH_SIZE
      - AUGMENTATION
    outs:
      - artifacts/training/model.h5
  
  evaluation:
    cmd: python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
    deps:
      - src/cnnClassifier/pipeline/stage_04_model_evaluation.py
      - config/config.yaml
      - artifacts/data_ingestion/Kidney-ct-scan-image
      - artifacts/training/model.h5
    params:
      - IMAGE_SIZE
      - BATCH_SIZE
    metrics:
      - scores.json:
          cache: false
```

**What This Does:**
- Defines 4 stages
- Each stage has dependencies (deps)
- When deps change, stage re-runs
- Outputs (outs) are cached
- Parameters (params) are tracked

**Run Pipeline:**
```bash
# Run all stages
dvc repro

# DVC automatically:
# - Checks what changed
# - Runs only necessary stages
# - Caches outputs

# View what will run:
dvc status
```

---

## ☁️ AWS Deployment

### **Deployment Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                         GITHUB                                   │
│                    (Source Code)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ Push code
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GITHUB ACTIONS                                 │
│                  (CI/CD Pipeline)                                │
│                                                                  │
│  1. Run tests                                                    │
│  2. Build Docker image                                           │
│  3. Push to ECR                                                  │
│  4. Deploy to EC2                                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS ECR                                       │
│            (Docker Image Registry)                               │
│                                                                  │
│  kidney-classifier:latest                                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           │ Pull image
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AWS EC2                                       │
│                (Virtual Server)                                  │
│                                                                  │
│  Docker Container Running:                                       │
│  - Flask App (app.py)                                            │
│  - Trained Model (model.h5)                                      │
│  - Port 8080                                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        USERS                                     │
│              Access via: http://ec2-ip:8080                      │
└─────────────────────────────────────────────────────────────────┘
```

### **Step-by-Step Deployment:**

#### **1. Create IAM User**

```bash
# AWS Console → IAM → Users → Add User

User Name: kidney-classifier-deployer

Permissions:
✅ AmazonEC2ContainerRegistryFullAccess
✅ AmazonEC2FullAccess

# Download credentials:
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=abc123...
```

#### **2. Create ECR Repository**

```bash
# AWS Console → ECR → Create Repository

Repository name: kidney-classifier
Region: us-east-1

# Note the URI:
URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/kidney-classifier
```

#### **3. Create EC2 Instance**

```bash
# AWS Console → EC2 → Launch Instance

Name: kidney-classifier-server
AMI: Ubuntu Server 22.04 LTS
Instance type: t2.medium (4GB RAM)
Key pair: Create new (download .pem file)
Security Group:
  - Allow SSH (port 22)
  - Allow HTTP (port 80)
  - Allow Custom TCP (port 8080)

Launch instance
```

#### **4. Install Docker on EC2**

```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Activate changes
newgrp docker

# Verify
docker --version
```

#### **5. Configure EC2 as Self-Hosted Runner**

```bash
# GitHub → Your Repo → Settings → Actions → Runners → New self-hosted runner

# Follow instructions on EC2:
# 1. Download runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz

# 2. Extract
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz

# 3. Configure
./config.sh --url https://github.com/yourusername/kidney-classifier --token YOUR_TOKEN

# 4. Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

#### **6. Setup GitHub Secrets**

```bash
# GitHub → Your Repo → Settings → Secrets → Actions → New repository secret

Add these secrets:
AWS_ACCESS_KEY_ID: AKIA...
AWS_SECRET_ACCESS_KEY: abc123...
AWS_REGION: us-east-1
AWS_ECR_LOGIN_URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com
ECR_REPOSITORY_NAME: kidney-classifier
```

#### **7. Create GitHub Actions Workflow**

```yaml
# .github/workflows/main.yaml

name: Deploy to AWS

on:
  push:
    branches:
      - main

jobs:
  build-and-deploy:
    runs-on: self-hosted
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Login to ECR
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_REGION: ${{ secrets.AWS_REGION }}
        run: |
          aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin ${{ secrets.AWS_ECR_LOGIN_URI }}
      
      - name: Build Docker image
        run: |
          docker build -t ${{ secrets.ECR_REPOSITORY_NAME }}:latest .
      
      - name: Tag Docker image
        run: |
          docker tag ${{ secrets.ECR_REPOSITORY_NAME }}:latest ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
      
      - name: Push to ECR
        run: |
          docker push ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
      
      - name: Pull and run on EC2
        run: |
          docker pull ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
          docker stop kidney-classifier || true
          docker rm kidney-classifier || true
          docker run -d -p 8080:8080 --name kidney-classifier ${{ secrets.AWS_ECR_LOGIN_URI }}/${{ secrets.ECR_REPOSITORY_NAME }}:latest
```

#### **8. Create Dockerfile**

```dockerfile
# Dockerfile

FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8080

# Run app
CMD ["python3", "app.py"]
```

#### **9. Deploy!**

```bash
# Push code to GitHub
git add .
git commit -m "Setup deployment"
git push origin main

# GitHub Actions automatically:
# 1. Builds Docker image
# 2. Pushes to ECR
# 3. Deploys to EC2

# Access app:
# http://your-ec2-ip:8080
```

---

## 🐛 Troubleshooting

### **Common Errors & Solutions:**

#### **Error 1: `BoxKeyError: 'artifact_roots'`**
```python
# Problem: Typo in config.yaml

# Solution: Check config.yaml first line
artifact_roots: artifacts  # Must be exactly this
```

#### **Error 2: `ModuleNotFoundError: No module named 'cnnClassifier'`**
```bash
# Problem: Package not installed

# Solution: Install in editable mode
pip install -e .
```

#### **Error 3: `OOM (Out of Memory) during training`**
```yaml
# Problem: Batch size too large

# Solution: Reduce in params.yaml
BATCH_SIZE: 8  # Instead of 16 or 32
```

#### **Error 4: `Unable to download data`**
```python
# Problem: Google Drive link requires authentication

# Solution: Use direct download link
# OR manually download and place in artifacts/data_ingestion/
```

#### **Error 5: `Docker build fails`**
```bash
# Problem: Large model file in image

# Solution: Use .dockerignore
# Create .dockerignore:
research/
*.ipynb
.git/
```

#### **Error 6: `MLflow connection error`**
```bash
# Problem: Environment variables not set

# Solution: Export variables before running
export MLFLOW_TRACKING_URI=https://dagshub.com/...
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...
```

---

## 🎓 Learning Path for Beginners

### **Week 1-2: Understanding Structure**
- ✅ Read this README fully
- ✅ Understand each file's purpose
- ✅ Run the project locally
- ✅ Experiment in research/ notebooks

### **Week 3-4: Modify Components**
- ✅ Change hyperparameters in params.yaml
- ✅ Try different learning rates
- ✅ Modify data augmentation
- ✅ Track experiments with MLflow

### **Week 5-6: Build Your Own**
- ✅ Use this structure for a new project
- ✅ Change to different dataset (cats/dogs, flowers, etc.)
- ✅ Modify model (try ResNet50 instead of VGG16)
- ✅ Deploy your own model

### **Week 7-8: Advanced**
- ✅ Add more evaluation metrics
- ✅ Implement early stopping
- ✅ Try transfer learning fine-tuning
- ✅ Create API endpoints

---

## 📚 Additional Resources

### **Learn More:**
- **TensorFlow Tutorial**: https://www.tensorflow.org/tutorials
- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **DVC Tutorial**: https://dvc.org/doc/start
- **Flask Tutorial**: https://flask.palletsprojects.com/

### **Similar Projects to Practice:**
1. **Plant Disease Classification** (same structure)
2. **Chest X-Ray Classification** (COVID detection)
3. **Skin Cancer Classification** (melanoma detection)
4. **Brain Tumor Classification** (MRI images)

---

## 🎯 Key Takeaways

### **What Makes This Project Professional:**

✅ **Modular Code**: Each component does one thing well
✅ **Configuration Management**: Easy to change settings
✅ **Experiment Tracking**: Know what works (MLflow)
✅ **Version Control**: Track data and code (Git + DVC)
✅ **Reproducibility**: Anyone can reproduce results
✅ **Deployment Ready**: Docker + AWS + CI/CD
✅ **Scalable**: Easy to add features

### **Why This Structure Matters:**

**In Interviews:**
```
Interviewer: "Tell me about your projects"

❌ Bad: "I built a CNN in Jupyter that got 95% accuracy"

✅ Good: "I built a production-ready kidney disease classifier with:
- Modular pipeline architecture
- MLflow experiment tracking
- DVC for data versioning  
- Docker containerization
- AWS deployment with CI/CD
- 95% accuracy on validation set"
```

**In Jobs:**
```
Manager: "We need to add a new feature"

With modular structure:
✅ "Sure, I'll add a new component and pipeline stage"
   (Takes 2-3 hours)

Without structure:
❌ "I need to rewrite the entire notebook..."
   (Takes 2-3 days and might break everything)
```

---

## 🏆 Final Words

**Congratulations!** You've learned a production-grade ML project structure. This is **exactly** how companies like Google, Amazon, and AI startups build ML systems.

**Remember:**
- 📓 Jupyter for **experiments**
- 🏗️ Modular code for **production**
- 🚀 Both skills make you **hirable**

**You're now ahead of 90% of ML students** who only know notebooks!

Keep building, keep learning! 💪🚀

---

**Questions? Issues?**
- Open an issue on GitHub
- Contact instructor
- Check troubleshooting section

**Happy Coding!** 🎉