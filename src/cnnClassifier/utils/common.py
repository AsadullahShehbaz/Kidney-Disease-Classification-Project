"""
cnnClassifier.utils.common

Common utility functions used across the project for:
- File and directory handling
- YAML / JSON / binary serialization
- Logging support
- Base64 image encoding and decoding
"""

import os
import json
import yaml
import joblib
import base64
from pathlib import Path
from typing import Any,Sequence

from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from cnnClassifier import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its content as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML configuration file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other file or parsing error.

    Returns:
        ConfigBox: Parsed YAML content with dot-notation access.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("YAML file is empty")

    except Exception as e:
        raise e


# @ensure_annotations
def create_directories(paths: list, verbose: bool = True) -> None:
    """
    Create multiple directories if they do not already exist.

    Args:
        paths (list[Path]): List of directory paths to create.
        verbose (bool): Whether to log directory creation.
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")



def save_json(path: Path, data: dict):
    """
    Save dictionary data to a JSON file.

    Args:
        path (Path): Destination path for JSON file.
        data (dict): Data to be saved.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"JSON file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load JSON data and return it as a ConfigBox.

    Args:
        path (Path): Path to JSON file.

    Returns:
        ConfigBox: JSON content with attribute-style access.
    """
    with open(path, "r") as f:
        content = json.load(f)

    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """
    Save Python object in binary format using joblib.

    Args:
        data (Any): Object to serialize.
        path (Path): Destination path for binary file.
    """
    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load a binary file using joblib.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Deserialized Python object.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    """
    Get file size in kilobytes (KB).

    Args:
        path (Path): Path to file.

    Returns:
        str: Approximate file size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring: str, fileName: str) -> None:
    """
    Decode a Base64-encoded image string and save it as an image file.

    Args:
        imgstring (str): Base64 encoded image string.
        fileName (str): Output image file name.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)


def encodeImageIntoBase64(image_path: str) -> bytes:
    """
    Encode an image file into Base64 format.

    Args:
        image_path (str): Path to image file.

    Returns:
        bytes: Base64 encoded image.
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())
