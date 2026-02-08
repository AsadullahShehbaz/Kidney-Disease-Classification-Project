"""
cnnClassifier.entity.config_entity

This module defines configuration entities used across the project.
DataIngestionConfig stores all parameters required for the data
ingestion stage in a structured and immutable format.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration schema for the Data Ingestion stage.

    Attributes:
        root_dir (Path): Root directory for data ingestion artifacts.
        source_url (str): URL of the dataset to be downloaded.
        local_data_file (Path): Path where the downloaded file is saved.
        unzip_dir (Path): Directory where extracted data will be stored.
    """

    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
