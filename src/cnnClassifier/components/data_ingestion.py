"""
cnnClassifier.components.data_ingestion

This module contains the DataIngestion component responsible for:
- Downloading the dataset from an external source
- Extracting the downloaded archive into the artifacts directory
"""

import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.entitiy.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    """
    Handles data downloading and extraction for the pipeline.
    """

    def __init__(self, config: DataIngestionConfig):
        # Store data ingestion configuration
        self.config = config

    def download_file(self) -> str:
        """
        Download dataset from Google Drive using gdown.

        Returns:
            str: Path to the downloaded zip file.
        """
        try:
            dataset_url = self.config.source_url
            zip_download_dir = self.config.local_data_file

            # Ensure data ingestion artifact directory exists
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(
                f"Downloading data from {dataset_url} into file {zip_download_dir}"
            )

            # Extract Google Drive file ID from URL
            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?export=download&id="

            # Download the dataset
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)

            logger.info(
                f"Downloaded data successfully: {zip_download_dir} "
                f"(size: {get_size(Path(zip_download_dir))})"
            )

            return zip_download_dir

        except Exception as e:
            logger.exception("Failed to download dataset")
            raise e

    def extract_zip_file(self) -> None:
        """
        Extract downloaded zip file into the specified directory.
        """
        unzip_path = self.config.unzip_dir

        # Create extraction directory if it does not exist
        os.makedirs(unzip_path, exist_ok=True)

        logger.info(f"Extracting zip file to: {unzip_path}")

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info("Extraction completed successfully")
