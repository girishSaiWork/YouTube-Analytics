"""
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
            cls._spark_session = SparkSession.builder \
                .appName(Config.SPARK_APP_NAME) \
                .master(Config.SPARK_MASTER) \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
                .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer") \
                .config("spark.hadoop.io.nativeio.native", "false") \
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
