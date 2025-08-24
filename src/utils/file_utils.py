"""
File operation utilities
"""

import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class FileUtils:
    """Utility functions for file operations"""

    @staticmethod
    def read_json(file_path: Path) -> Dict[str, Any]:
        """Read JSON file and return dictionary"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            raise

    @staticmethod
    def write_json(data: Dict[str, Any], file_path: Path):
        """Write dictionary to JSON file"""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error writing JSON file {file_path}: {e}")
            raise

    @staticmethod
    def ensure_directory(path: Path):
        """Ensure directory exists"""
        path.mkdir(parents=True, exist_ok=True)
