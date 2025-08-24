"""
Loader for processed YouTube Analytics data
"""

from pyspark.sql import SparkSession, DataFrame
from config.settings import Config
from pathlib import Path
from src.utils.pandas_io import PandasIOHandler, should_use_pandas_io
import logging

logger = logging.getLogger(__name__)

class ProcessedDataLoader:
    """Loader for processed YouTube Analytics data from parquet files"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.config = Config()
        self.pandas_io = PandasIOHandler(spark)
        self.use_pandas_io = should_use_pandas_io()
    
    def load_processed_videos(self, countries: list = None) -> DataFrame:
        """
        Load processed video data using environment-appropriate method

        Args:
            countries: List of countries to filter. If None, loads all countries

        Returns:
            DataFrame with processed video data
        """
        # Define possible file paths
        video_path_parquet = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos.parquet"
        video_path_csv = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos.csv"
        video_path_spark_parquet = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos"
        video_path_spark_csv = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos_csv"

        df = None

        # Try Pandas I/O first if on Windows
        if self.use_pandas_io:
            logger.info("Attempting to load using Pandas I/O (Windows compatibility)")

            # Try pandas parquet first
            if video_path_parquet.exists():
                logger.info(f"Loading from pandas parquet: {video_path_parquet}")
                df = self.pandas_io.load_parquet_as_spark_df(str(video_path_parquet))

            # Try pandas CSV if parquet failed
            if df is None and video_path_csv.exists():
                logger.info(f"Loading from pandas CSV: {video_path_csv}")
                df = self.pandas_io.load_csv_as_spark_df(str(video_path_csv))

        # Try native Spark I/O if pandas failed or not on Windows
        if df is None:
            logger.info("Using native Spark I/O")

            # Try Spark parquet format
            if video_path_spark_parquet.exists():
                logger.info(f"Loading from Spark parquet: {video_path_spark_parquet}")
                try:
                    df = self.spark.read.parquet(str(video_path_spark_parquet))
                except Exception as e:
                    if "UnsatisfiedLinkError" in str(e):
                        logger.warning("Spark parquet read failed due to Windows native library issue")
                    else:
                        raise

            # Try Spark CSV format
            if df is None and video_path_spark_csv.exists():
                logger.info(f"Loading from Spark CSV: {video_path_spark_csv}")
                try:
                    df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(video_path_spark_csv))
                except Exception as e:
                    if "UnsatisfiedLinkError" in str(e):
                        logger.warning("Spark CSV read failed due to Windows native library issue")
                    else:
                        raise

        if df is None:
            raise FileNotFoundError(f"No processed video data found in any supported format")

        # Filter by countries if specified
        if countries:
            logger.info(f"Filtering data for countries: {countries}")
            df = df.filter(df.country.isin(countries))

        logger.info(f"Loaded {df.count()} processed video records")
        return df
    
    def load_category_mappings(self, countries: list = None) -> DataFrame:
        """
        Load category mappings from parquet or CSV files

        Args:
            countries: List of countries to filter. If None, loads all countries

        Returns:
            DataFrame with category mappings
        """
        category_path = self.config.PROCESSED_DATA_PATH / "category_mappings"
        category_path_csv = self.config.PROCESSED_DATA_PATH / "category_mappings_csv"

        # Try parquet first, then CSV fallback
        if category_path.exists():
            logger.info(f"Loading category mappings from parquet: {category_path}")
            try:
                df = self.spark.read.parquet(str(category_path))
            except Exception as e:
                if "UnsatisfiedLinkError" in str(e):
                    logger.warning("Parquet read failed due to Windows native library issue. Trying CSV fallback.")
                    if category_path_csv.exists():
                        logger.info(f"Loading category mappings from CSV: {category_path_csv}")
                        df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(category_path_csv))
                    else:
                        raise FileNotFoundError(f"Neither parquet nor CSV category mappings found")
                else:
                    raise
        elif category_path_csv.exists():
            logger.info(f"Loading category mappings from CSV: {category_path_csv}")
            df = self.spark.read.option("header", "true").option("inferSchema", "true").csv(str(category_path_csv))
        else:
            raise FileNotFoundError(f"Category mappings not found at: {category_path} or {category_path_csv}")

        # Filter by countries if specified
        if countries:
            logger.info(f"Filtering categories for countries: {countries}")
            df = df.filter(df.country.isin(countries))

        logger.info(f"Loaded {df.count()} category mappings")
        return df
    
    def load_data_for_analysis(self, countries: list = None, include_categories: bool = True) -> dict:
        """
        Load all data needed for analysis
        
        Args:
            countries: List of countries to filter. If None, loads all countries
            include_categories: Whether to include category mappings
            
        Returns:
            Dictionary with 'videos' and optionally 'categories' DataFrames
        """
        logger.info("Loading data for analysis")
        
        result = {}
        
        # Load video data
        result['videos'] = self.load_processed_videos(countries)
        
        # Load category mappings if requested
        if include_categories:
            try:
                result['categories'] = self.load_category_mappings(countries)
            except FileNotFoundError:
                logger.warning("Category mappings not found, skipping...")
                result['categories'] = None
        
        logger.info("Data loading for analysis completed")
        return result
    
    def get_available_countries(self) -> list:
        """
        Get list of countries available in processed data
        
        Returns:
            List of country codes available in the data
        """
        try:
            df = self.load_processed_videos()
            countries = [row.country for row in df.select("country").distinct().collect()]
            logger.info(f"Available countries in processed data: {countries}")
            return sorted(countries)
        except FileNotFoundError:
            logger.warning("No processed data found")
            return []
    
    def get_data_summary(self) -> dict:
        """
        Get summary information about the processed data
        
        Returns:
            Dictionary with data summary statistics
        """
        try:
            videos_df = self.load_processed_videos()
            
            summary = {
                "total_videos": videos_df.count(),
                "countries": self.get_available_countries(),
                "date_range": {
                    "earliest": videos_df.agg({"trending_date": "min"}).collect()[0][0],
                    "latest": videos_df.agg({"trending_date": "max"}).collect()[0][0]
                },
                "unique_channels": videos_df.select("channel_title").distinct().count(),
                "unique_categories": videos_df.select("category_name").distinct().count() if "category_name" in videos_df.columns else 0
            }
            
            logger.info(f"Data summary: {summary}")
            return summary
            
        except FileNotFoundError:
            logger.warning("No processed data found for summary")
            return {"error": "No processed data available"}
    
    def validate_processed_data(self) -> dict:
        """
        Validate the integrity of processed data
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "videos_data_exists": False,
            "categories_data_exists": False,
            "data_quality": {}
        }
        
        # Check if video data exists
        video_path = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos"
        if video_path.exists():
            validation_results["videos_data_exists"] = True
            
            # Basic data quality checks
            try:
                df = self.load_processed_videos()
                validation_results["data_quality"] = {
                    "total_records": df.count(),
                    "null_video_ids": df.filter(df.video_id.isNull()).count(),
                    "duplicate_records": df.count() - df.dropDuplicates(["video_id", "trending_date"]).count(),
                    "countries_count": df.select("country").distinct().count()
                }
            except Exception as e:
                validation_results["data_quality"]["error"] = str(e)
        
        # Check if category data exists
        category_path = self.config.PROCESSED_DATA_PATH / "category_mappings"
        if category_path.exists():
            validation_results["categories_data_exists"] = True
        
        logger.info(f"Data validation results: {validation_results}")
        return validation_results
