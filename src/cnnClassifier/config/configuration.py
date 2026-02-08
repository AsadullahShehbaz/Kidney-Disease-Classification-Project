"""
cnnClassifier.config.configuration

This module contains the ConfigurationManager class, which is responsible for:
- Loading YAML configuration and parameter files
- Creating required artifact directories
- Providing strongly-typed configuration objects to pipeline components
"""

from cnnClassifier.constants import CONFIG_PATH_YAML, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entitiy.config_entity import DataIngestionConfig


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
