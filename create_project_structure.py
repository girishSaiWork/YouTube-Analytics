#!/usr/bin/env python3
"""
Script to create the YouTube Analytics project structure
This script will create all necessary folders and basic files for the project
"""

import os
from pathlib import Path

def create_project_structure():
    """Create the complete project structure for YouTube Analytics"""
    
    # Define the project root
    project_name = "YouTube Analytics"
    project_root = Path(project_name)
    
    print(f"Creating project structure for: {project_name}")
    print("=" * 50)
    
    # Define all directories to create
    directories = [
        # Root level
        "",
        
        # Config
        "config",
        
        # Data directories
        "data",
        "data/raw",
        "data/raw/videos",
        "data/raw/categories", 
        "data/processed",
        "data/output",
        "data/schemas",
        
        # Source code
        "src",
        "src/data_ingestion",
        "src/data_processing", 
        "src/analytics",
        "src/utils",
        
        # Notebooks
        "notebooks",
        
        # Tests
        "tests",
        "tests/test_data_processing",
        "tests/test_analytics",
        "tests/fixtures",
        
        # Scripts
        "scripts",
        
        # Documentation
        "docs"
    ]
    
    # Create directories
    print("Creating directories...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        if directory:  # Don't print for root directory
            print(f"  ✓ Created: {directory}")
    
    print(f"\n✅ Successfully created project structure in '{project_name}' folder!")
    print(f"📁 Total directories created: {len([d for d in directories if d])}")
    
    return project_root

def create_init_files(project_root):
    """Create __init__.py files for Python packages"""
    
    print("\nCreating __init__.py files...")
    
    init_files = [
        "config/__init__.py",
        "src/__init__.py", 
        "src/data_ingestion/__init__.py",
        "src/data_processing/__init__.py",
        "src/analytics/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
        "tests/test_data_processing/__init__.py",
        "tests/test_analytics/__init__.py"
    ]
    
    for init_file in init_files:
        file_path = project_root / init_file
        file_path.write_text('"""Package initialization file"""\n', encoding='utf-8')
        print(f"  ✓ Created: {init_file}")
    
    print(f"✅ Created {len(init_files)} __init__.py files!")

def create_basic_files(project_root):
    """Create basic project files"""
    
    print("\nCreating basic project files...")
    
    # README.md
    readme_content = """# YouTube Analytics Project

A PySpark-based data engineering project for analyzing YouTube trending videos data.

## Project Structure

```
YouTube Analytics/
├── config/          # Configuration files
├── data/           # Data storage (raw, processed, output)
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for exploration
├── tests/          # Unit tests
├── scripts/        # Execution scripts
└── docs/           # Documentation
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure your environment: Copy `.env.example` to `.env`
3. Run the data pipeline: `python scripts/run_pipeline.py`

## Data Sources

This project analyzes YouTube trending videos data from multiple countries:
- Video data: CSV files with trending video information
- Category data: JSON files with category ID mappings

## Features

- Multi-country data processing
- Trending pattern analysis
- Category-based insights
- Engagement metrics calculation
"""
    
    (project_root / "README.md").write_text(readme_content, encoding='utf-8')
    print("  ✓ Created: README.md")

    # requirements.txt
    requirements_content = """# Core dependencies
pyspark>=3.5.0
findspark>=2.0.1
python-dotenv>=1.0.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization (for notebooks)
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Development dependencies
pytest>=7.4.0
black>=23.0.0
flake8>=6.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
"""

    (project_root / "requirements.txt").write_text(requirements_content, encoding='utf-8')
    print("  ✓ Created: requirements.txt")
    
    # .env.example
    env_example_content = """# Environment Configuration Example
# Copy this file to .env and update with your values

# Spark Configuration
SPARK_HOME=/path/to/spark
PYSPARK_PYTHON=python
PYSPARK_DRIVER_PYTHON=python

# Data Paths
RAW_DATA_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
OUTPUT_DATA_PATH=./data/output

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/youtube_analytics.log
"""
    
    (project_root / ".env.example").write_text(env_example_content, encoding='utf-8')
    print("  ✓ Created: .env.example")

    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env

# Jupyter Notebook
.ipynb_checkpoints

# Data files (add specific patterns as needed)
data/raw/*.csv
data/processed/*
data/output/*

# Logs
logs/
*.log

# OS
.DS_Store
Thumbs.db

# Spark
metastore_db/
derby.log
spark-warehouse/
"""

    (project_root / ".gitignore").write_text(gitignore_content, encoding='utf-8')
    print("  ✓ Created: .gitignore")
    
    print("✅ Created basic project files!")

def create_placeholder_files(project_root):
    """Create placeholder files and basic templates for the project structure"""

    print("\nCreating placeholder files and templates...")

    # Config files
    settings_content = '''"""
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
'''

    (project_root / "config" / "settings.py").write_text(settings_content, encoding='utf-8')
    print("  ✓ Created: config/settings.py")

    # Spark config
    spark_config_content = '''"""
Spark-specific configuration and session management
"""

from pyspark.sql import SparkSession
from config.settings import Config

class SparkConfig:
    """Spark configuration and session management"""

    _spark_session = None

    @classmethod
    def get_spark_session(cls) -> SparkSession:
        """Get or create Spark session with optimized configuration"""
        if cls._spark_session is None:
            cls._spark_session = SparkSession.builder \\
                .appName(Config.SPARK_APP_NAME) \\
                .master(Config.SPARK_MASTER) \\
                .config("spark.sql.adaptive.enabled", "true") \\
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \\
                .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") \\
                .getOrCreate()

            # Set log level to reduce verbosity
            cls._spark_session.sparkContext.setLogLevel("WARN")

        return cls._spark_session

    @classmethod
    def stop_spark_session(cls):
        """Stop the Spark session"""
        if cls._spark_session:
            cls._spark_session.stop()
            cls._spark_session = None
'''

    (project_root / "config" / "spark_config.py").write_text(spark_config_content, encoding='utf-8')
    print("  ✓ Created: config/spark_config.py")

    # Data schemas
    schemas_content = '''"""
Data schemas for YouTube Analytics project
"""

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, BooleanType, TimestampType

class YouTubeSchemas:
    """Schema definitions for YouTube data"""

    @staticmethod
    def get_video_schema():
        """Schema for video CSV files"""
        return StructType([
            StructField("video_id", StringType(), True),
            StructField("trending_date", StringType(), True),
            StructField("title", StringType(), True),
            StructField("channel_title", StringType(), True),
            StructField("category_id", StringType(), True),
            StructField("publish_time", StringType(), True),
            StructField("tags", StringType(), True),
            StructField("views", StringType(), True),
            StructField("likes", StringType(), True),
            StructField("dislikes", StringType(), True),
            StructField("comment_count", StringType(), True),
            StructField("thumbnail_link", StringType(), True),
            StructField("comments_disabled", StringType(), True),
            StructField("ratings_disabled", StringType(), True),
            StructField("video_error_or_removed", StringType(), True),
            StructField("description", StringType(), True)
        ])

    @staticmethod
    def get_processed_video_schema():
        """Schema for processed video data with proper types"""
        return StructType([
            StructField("video_id", StringType(), False),
            StructField("trending_date", StringType(), True),
            StructField("title", StringType(), True),
            StructField("channel_title", StringType(), True),
            StructField("category_id", IntegerType(), True),
            StructField("category_name", StringType(), True),
            StructField("publish_time", TimestampType(), True),
            StructField("tags", StringType(), True),
            StructField("views", IntegerType(), True),
            StructField("likes", IntegerType(), True),
            StructField("dislikes", IntegerType(), True),
            StructField("comment_count", IntegerType(), True),
            StructField("comments_disabled", BooleanType(), True),
            StructField("ratings_disabled", BooleanType(), True),
            StructField("video_error_or_removed", BooleanType(), True),
            StructField("country", StringType(), True)
        ])
'''

    (project_root / "data" / "schemas" / "youtube_schemas.py").write_text(schemas_content, encoding='utf-8')
    print("  ✓ Created: data/schemas/youtube_schemas.py")

    print("✅ Created configuration and schema files!")

def create_src_files(project_root):
    """Create source code template files"""

    print("\nCreating source code templates...")

    # Utils - Spark utilities
    spark_utils_content = '''"""
Spark utility functions and session management
"""

from pyspark.sql import SparkSession
from config.spark_config import SparkConfig
import logging

logger = logging.getLogger(__name__)

class SparkUtils:
    """Utility functions for Spark operations"""

    @staticmethod
    def get_spark_session() -> SparkSession:
        """Get the configured Spark session"""
        return SparkConfig.get_spark_session()

    @staticmethod
    def stop_spark_session():
        """Stop the Spark session"""
        SparkConfig.stop_spark_session()

    @staticmethod
    def optimize_dataframe(df, num_partitions=None):
        """Optimize DataFrame partitioning"""
        if num_partitions:
            return df.repartition(num_partitions)
        return df.coalesce(1)

    @staticmethod
    def cache_dataframe(df, storage_level="MEMORY_AND_DISK"):
        """Cache DataFrame with specified storage level"""
        return df.cache()
'''

    (project_root / "src" / "utils" / "spark_utils.py").write_text(spark_utils_content, encoding='utf-8')
    print("  ✓ Created: src/utils/spark_utils.py")

    # Utils - File utilities
    file_utils_content = '''"""
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
'''

    (project_root / "src" / "utils" / "file_utils.py").write_text(file_utils_content, encoding='utf-8')
    print("  ✓ Created: src/utils/file_utils.py")

    # Data ingestion - readers
    readers_content = '''"""
Data reading utilities for YouTube Analytics
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from pyspark.sql.functions import lit
from config.settings import Config
from src.utils.file_utils import FileUtils
from data.schemas.youtube_schemas import YouTubeSchemas
import logging

logger = logging.getLogger(__name__)

class YouTubeDataReader:
    """Data reader for YouTube datasets"""

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = Config()

    def read_videos_data(self, country_code: str, schema: StructType = None) -> DataFrame:
        """Read video data for a specific country"""
        file_path = self.config.get_video_file_path(country_code)

        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        logger.info(f"Reading video data for {country_code} from {file_path}")

        if schema is None:
            schema = YouTubeSchemas.get_video_schema()

        df = self.spark.read.csv(
            str(file_path),
            header=True,
            schema=schema,
            multiLine=True,
            escape='"'
        )

        # Add country column
        df = df.withColumn("country", lit(country_code))

        return df

    def read_categories_data(self, country_code: str) -> dict:
        """Read category mapping for a specific country"""
        file_path = self.config.get_category_file_path(country_code)

        if not file_path.exists():
            raise FileNotFoundError(f"Category file not found: {file_path}")

        logger.info(f"Reading category data for {country_code} from {file_path}")

        return FileUtils.read_json(file_path)

    def read_all_countries_data(self) -> DataFrame:
        """Read and combine data from all countries"""
        dfs = []

        for country in self.config.COUNTRIES:
            try:
                df = self.read_videos_data(country)
                dfs.append(df)
                logger.info(f"Successfully loaded data for {country}")
            except FileNotFoundError:
                logger.warning(f"Data file not found for {country}, skipping...")
                continue

        if not dfs:
            raise ValueError("No data files found for any country")

        # Union all DataFrames
        combined_df = dfs[0]
        for df in dfs[1:]:
            combined_df = combined_df.union(df)

        return combined_df
'''

    (project_root / "src" / "data_ingestion" / "readers.py").write_text(readers_content, encoding='utf-8')
    print("  ✓ Created: src/data_ingestion/readers.py")

    print("✅ Created source code template files!")

def create_additional_templates(project_root):
    """Create additional template files for notebooks and scripts"""

    print("\nCreating additional template files...")

    # Main pipeline script
    pipeline_script_content = '''#!/usr/bin/env python3
"""
Main pipeline script for YouTube Analytics
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.settings import Config
from src.utils.spark_utils import SparkUtils
from src.data_ingestion.readers import YouTubeDataReader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main pipeline execution"""
    logger = logging.getLogger(__name__)
    logger.info("Starting YouTube Analytics Pipeline")

    try:
        # Get Spark session
        spark = SparkUtils.get_spark_session()
        logger.info("Spark session created successfully")

        # Initialize data reader
        reader = YouTubeDataReader(spark)

        # Read data for all countries
        logger.info("Reading data for all countries...")
        df = reader.read_all_countries_data()

        logger.info(f"Total records loaded: {df.count()}")
        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        SparkUtils.stop_spark_session()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    setup_logging()
    main()
'''

    (project_root / "scripts" / "run_pipeline.py").write_text(pipeline_script_content, encoding='utf-8')
    print("  ✓ Created: scripts/run_pipeline.py")

    # Data exploration notebook template
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# YouTube Analytics - Data Exploration\\n",
    "\\n",
    "This notebook provides initial exploration of the YouTube trending videos dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "from pathlib import Path\\n",
    "\\n",
    "# Add src to path\\n",
    "sys.path.append(str(Path.cwd().parent / \\"src\\"))\\n",
    "\\n",
    "from config.settings import Config\\n",
    "from src.utils.spark_utils import SparkUtils\\n",
    "from src.data_ingestion.readers import YouTubeDataReader"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize Spark session\\n",
    "spark = SparkUtils.get_spark_session()\\n",
    "print(f\\"Spark version: {spark.version}\\")\\n",
    "print(f\\"Available countries: {Config.COUNTRIES}\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "reader = YouTubeDataReader(spark)\\n",
    "\\n",
    "# Start with one country for exploration\\n",
    "df_us = reader.read_videos_data('US')\\n",
    "print(f\\"US data shape: {df_us.count()} rows, {len(df_us.columns)} columns\\")\\n",
    "df_us.printSchema()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Show sample data\\n",
    "df_us.show(5, truncate=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic statistics\\n",
    "df_us.describe().show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

    (project_root / "notebooks" / "01_data_exploration.ipynb").write_text(notebook_content, encoding='utf-8')
    print("  ✓ Created: notebooks/01_data_exploration.ipynb")

    # Documentation files
    data_dict_content = '''# Data Dictionary

## Video Data Files (CSVs)

| Column | Type | Description |
|--------|------|-------------|
| video_id | String | Unique identifier for the video |
| trending_date | String | Date when video was trending (YY.DD.MM format) |
| title | String | Video title |
| channel_title | String | Name of the YouTube channel |
| category_id | String | Category ID (maps to category JSON) |
| publish_time | String | Video publication timestamp |
| tags | String | Video tags (pipe-separated) |
| views | String | Number of views |
| likes | String | Number of likes |
| dislikes | String | Number of dislikes |
| comment_count | String | Number of comments |
| thumbnail_link | String | URL to video thumbnail |
| comments_disabled | String | Whether comments are disabled |
| ratings_disabled | String | Whether ratings are disabled |
| video_error_or_removed | String | Whether video has errors or was removed |
| description | String | Video description |

## Category Data Files (JSONs)

Contains mapping of category IDs to category names for each country.

Structure:
```json
{
  "items": [
    {
      "id": "1",
      "snippet": {
        "title": "Film & Animation"
      }
    }
  ]
}
```

## Countries Available

- CA: Canada
- DE: Germany
- FR: France
- GB: Great Britain
- IN: India
- JP: Japan
- KR: South Korea
- MX: Mexico
- RU: Russia
- US: United States
'''

    (project_root / "docs" / "data_dictionary.md").write_text(data_dict_content, encoding='utf-8')
    print("  ✓ Created: docs/data_dictionary.md")

    print("✅ Created additional template files!")

if __name__ == "__main__":
    print("🚀 YouTube Analytics Project Structure Creator")
    print("=" * 50)
    
    try:
        # Create the project structure
        project_root = create_project_structure()
        
        # Create __init__.py files
        create_init_files(project_root)
        
        # Create basic files
        create_basic_files(project_root)

        # Create placeholder files and templates
        create_placeholder_files(project_root)

        # Create source code templates
        create_src_files(project_root)

        # Create additional templates
        create_additional_templates(project_root)

        print("\n" + "=" * 50)
        print("🎉 PROJECT SETUP COMPLETE!")
        print("=" * 50)
        print(f"📁 Project created at: {project_root.absolute()}")
        print("\nNext steps:")
        print("1. Navigate to the project folder: cd 'YouTube Analytics'")
        print("2. Copy your YouTube data to data/raw/")
        print("3. Create a virtual environment: python -m venv venv")
        print("4. Activate it: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/Mac)")
        print("5. Install dependencies: pip install -r requirements.txt")
        print("6. Copy .env.example to .env and configure your settings")
        
    except Exception as e:
        print(f"❌ Error creating project structure: {e}")
        raise
