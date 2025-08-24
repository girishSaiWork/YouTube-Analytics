"""
Configuration settings for YouTube Analytics project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Main configuration class"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    RAW_DATA_PATH = DATA_ROOT / "raw"
    PROCESSED_DATA_PATH = DATA_ROOT / "processed"
    OUTPUT_DATA_PATH = DATA_ROOT / "output"

    # Spark configuration
    SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "YouTube Analytics")
    SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")

    # Data sources
    COUNTRIES = ["CA", "DE", "FR", "GB", "IN", "JP", "KR", "MX", "RU", "US"]

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "./logs/youtube_analytics.log")

    @classmethod
    def get_video_file_path(cls, country_code: str) -> Path:
        """Get the path to a country's video CSV file"""
        return cls.RAW_DATA_PATH / "videos" / f"{country_code}videos.csv"

    @classmethod
    def get_category_file_path(cls, country_code: str) -> Path:
        """Get the path to a country's category JSON file"""
        return cls.RAW_DATA_PATH / "categories" / f"{country_code}_category_id.json"
