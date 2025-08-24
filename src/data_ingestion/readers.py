"""
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

    def read_categories_data(self, country_code: str) -> DataFrame:
        """Read category mapping for a specific country as DataFrame"""
        file_path = self.config.get_category_file_path(country_code)

        if not file_path.exists():
            raise FileNotFoundError(f"Category file not found: {file_path}")

        logger.info(f"Reading category data for {country_code} from {file_path}")

        df = self.spark.read.json(
            str(file_path),
            multiLine=True
        )

        return df

    def read_and_flatten_categories_data(self, country_code: str) -> DataFrame:
        """Read and flatten category mapping for a specific country"""
        from src.data_processing.transformers import YouTubeDataTransformer

        logger.info(f"Reading and flattening category data for {country_code}")

        # Read raw JSON
        raw_df = self.read_categories_data(country_code)

        # Flatten the JSON structure
        flattened_df = YouTubeDataTransformer.json_flattener(raw_df)

        # Extract clean category mapping
        category_mapping = YouTubeDataTransformer.extract_category_mapping(flattened_df)

        return category_mapping

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
