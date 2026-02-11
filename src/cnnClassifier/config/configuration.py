"""
cnnClassifier.config.configuration

This module contains the ConfigurationManager class, which is responsible for:
- Loading YAML configuration and parameter files
- Creating required artifact directories
- Providing strongly-typed configuration objects to pipeline components
"""

from cnnClassifier.constants import CONFIG_PATH_YAML, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entitiy.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)
from pathlib import Path 
import os 


class ConfigurationManager:
    """
    Central configuration manager for the project.

    Responsibilities:
    - Read configuration and parameter YAML files
    - Initialize artifact directories
    - Generate validated, strongly-typed config objects for each pipeline stage
    """

    def __init__(
        self,
        config_filepath=CONFIG_PATH_YAML,
        params_filepath=PARAMS_FILE_PATH
    ):
        # Load main configuration (paths, URLs, pipeline structure)
        self.config = read_yaml(config_filepath)

        # Load parameters configuration (hyperparameters, constants, etc.)
        self.params = read_yaml(params_filepath)

        # Ensure root artifacts directory exists
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates and returns the configuration required
        for the Data Ingestion stage.
        """
        # Extract data ingestion configuration block
        config = self.config.data_ingestion 

        # Create directory for data ingestion artifacts
        create_directories([config.root_dir])

        # Convert YAML configuration to a dataclass
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    def get_prepare_base_model_config(self)->PrepareBaseModelConfig:
        config = self.config.prepare_base_model 
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weight=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

        
    def get_training_config(self)-> TrainingConfig:

        training = self.config.training
        prepare_base_model = self.config.prepare_base_model 
        params = self.params 
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,"kidney-ct-scan-image")
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config 

    def get_evaluation_config(self)-> EvaluationConfig:

        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.keras",
            training_data="artifacts/data_ingestion/kidney-ct-scan-image",
            mlflow_uri="https://dagshub.com/asadullahcreative/Kidney-Disease-Classification-Project.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )

        return eval_config