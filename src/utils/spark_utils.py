"""
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
