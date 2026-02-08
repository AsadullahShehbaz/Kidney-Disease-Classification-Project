"""

cnnClassifier.pipeline.stage_01_data_ingestion

This module defines the pipeline logic for Stage 01: Data Ingestion.
It orchestrates the end-to-end data ingestion workflow by:
- Loading project configuration
- Initializing the DataIngestion component
- Downloading the dataset
- Extracting raw data into the artifacts directory

This stage serves as the entry point of the training pipeline and
ensures that raw data is available for downstream processing stages.
"""


# Import configuration manager to load pipeline settings
from cnnClassifier.config.configuration import ConfigurationManager

# Import Data Ingestion component responsible for data download & extraction
from cnnClassifier.components.data_ingestion import DataIngestion

# Import centralized logger for pipeline tracking
from cnnClassifier import logger

# Stage name used for logging and pipeline monitoring
STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """
    Pipeline class responsible for executing
    the Data Ingestion stage of the project.
    """

    def __init__(self):
        # No initialization required for now
        pass

    def main(self):
        """
        Orchestrates the data ingestion process:
        1. Loads configuration
        2. Downloads dataset
        3. Extracts dataset files
        """
        try:
            # Load project configuration
            config = ConfigurationManager()

            # Retrieve data ingestion specific configuration
            data_ingestion_config = config.get_data_ingestion_config()

            # Initialize Data Ingestion component
            data_ingestion = DataIngestion(config=data_ingestion_config)

            # Download dataset from source
            data_ingestion.download_file()

            # Extract downloaded dataset
            data_ingestion.extract_zip_file()

        except Exception as e:
            # Propagate exception for centralized error handling
            raise e


# Entry point for executing the pipeline stage
if __name__ == "__name__":
    try:
        # Log stage start
        logger.info(f">>>>> Stage {STAGE_NAME} started <<<<<<")

        # Create pipeline object and execute
        obj = DataIngestionTrainingPipeline()
        obj.main()

        # Log stage completion
        logger.info(f">>>>> Stage {STAGE_NAME} ended <<<<<<")

    except Exception as e:
        # Log detailed exception traceback
        logger.exception(e)
        raise
