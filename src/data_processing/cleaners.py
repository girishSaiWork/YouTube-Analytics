"""
Data cleaning utilities for YouTube Analytics
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, isnan, isnull, trim, length, to_timestamp
import logging

logger = logging.getLogger(__name__)

class YouTubeDataCleaner:
    """Data cleaning utilities for YouTube datasets"""
    
    @staticmethod
    def remove_null_video_ids(df: DataFrame) -> DataFrame:
        """
        Remove rows with null or empty video IDs
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with null video IDs removed
        """
        logger.info("Removing rows with null or empty video IDs")
        
        initial_count = df.count()
        
        cleaned_df = df.filter(
            col("video_id").isNotNull() & 
            (trim(col("video_id")) != "") &
            (length(col("video_id")) > 0)
        )
        
        final_count = cleaned_df.count()
        removed_count = initial_count - final_count
        
        logger.info(f"Removed {removed_count} rows with invalid video IDs. Remaining: {final_count}")
        
        return cleaned_df
    
    @staticmethod
    def clean_text_columns(df: DataFrame, text_columns: list = None) -> DataFrame:
        """
        Clean text columns by trimming whitespace and handling nulls
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to clean. If None, cleans common text columns
            
        Returns:
            DataFrame with cleaned text columns
        """
        if text_columns is None:
            text_columns = ["title", "channel_title", "description"]
        
        logger.info(f"Cleaning text columns: {text_columns}")
        
        for col_name in text_columns:
            if col_name in df.columns:
                logger.info(f"Cleaning column: {col_name}")
                df = df.withColumn(col_name, 
                                  when(col(col_name).isNull(), "")
                                  .otherwise(trim(col(col_name))))
        
        logger.info("Text columns cleaned successfully")
        return df
    
    @staticmethod
    def handle_negative_metrics(df: DataFrame) -> DataFrame:
        """
        Handle negative values in metric columns (views, likes, etc.)
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with negative metrics set to 0
        """
        logger.info("Handling negative values in metric columns")
        
        metric_columns = ["views", "likes", "dislikes", "comment_count"]
        
        for col_name in metric_columns:
            if col_name in df.columns:
                logger.info(f"Cleaning negative values in: {col_name}")
                df = df.withColumn(col_name, 
                                  when(col(col_name) < 0, 0)
                                  .otherwise(col(col_name)))
        
        logger.info("Negative metrics handled successfully")
        return df
    
    @staticmethod
    def parse_publish_time(df: DataFrame, timestamp_column: str = "publish_time") -> DataFrame:
        """
        Parse publish_time string to proper timestamp
        
        Args:
            df: Input DataFrame
            timestamp_column: Name of the timestamp column
            
        Returns:
            DataFrame with parsed timestamp
        """
        logger.info(f"Parsing timestamp column: {timestamp_column}")
        
        df_parsed = df.withColumn(
            f"{timestamp_column}_parsed",
            to_timestamp(col(timestamp_column), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        )
        
        logger.info("Timestamp parsing completed")
        return df_parsed
    
    @staticmethod
    def remove_duplicates(df: DataFrame, subset_columns: list = None) -> DataFrame:
        """
        Remove duplicate rows based on specified columns
        
        Args:
            df: Input DataFrame
            subset_columns: Columns to consider for duplicates. If None, uses video_id and trending_date
            
        Returns:
            DataFrame with duplicates removed
        """
        if subset_columns is None:
            subset_columns = ["video_id", "trending_date"]
        
        logger.info(f"Removing duplicates based on columns: {subset_columns}")
        
        initial_count = df.count()
        
        # Check if all subset columns exist
        existing_columns = [col for col in subset_columns if col in df.columns]
        if not existing_columns:
            logger.warning("No subset columns found in DataFrame. Using all columns for deduplication.")
            deduplicated_df = df.distinct()
        else:
            deduplicated_df = df.dropDuplicates(existing_columns)
        
        final_count = deduplicated_df.count()
        removed_count = initial_count - final_count
        
        logger.info(f"Removed {removed_count} duplicate rows. Remaining: {final_count}")
        
        return deduplicated_df
    
    @staticmethod
    def validate_data_quality(df: DataFrame) -> dict:
        """
        Perform data quality checks and return summary
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Performing data quality validation")
        
        total_rows = df.count()
        
        quality_metrics = {
            "total_rows": total_rows,
            "null_video_ids": df.filter(col("video_id").isNull()).count(),
            "empty_titles": df.filter((col("title").isNull()) | (trim(col("title")) == "")).count(),
            "negative_views": df.filter(col("views") < 0).count() if "views" in df.columns else 0,
            "negative_likes": df.filter(col("likes") < 0).count() if "likes" in df.columns else 0,
        }
        
        # Calculate percentages (create a copy to avoid modifying dict during iteration)
        metrics_copy = quality_metrics.copy()
        for metric, count in metrics_copy.items():
            if metric != "total_rows" and total_rows > 0:
                percentage = (count / total_rows) * 100
                quality_metrics[f"{metric}_percentage"] = round(percentage, 2)
        
        logger.info(f"Data quality validation completed: {quality_metrics}")
        
        return quality_metrics
    
    @staticmethod
    def apply_full_cleaning_pipeline(df: DataFrame) -> DataFrame:
        """
        Apply the complete data cleaning pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Fully cleaned DataFrame
        """
        logger.info("Starting full data cleaning pipeline")
        
        # Step 1: Remove null video IDs
        df = YouTubeDataCleaner.remove_null_video_ids(df)
        
        # Step 2: Clean text columns
        df = YouTubeDataCleaner.clean_text_columns(df)
        
        # Step 3: Handle negative metrics
        df = YouTubeDataCleaner.handle_negative_metrics(df)
        
        # Step 4: Parse timestamps
        df = YouTubeDataCleaner.parse_publish_time(df)
        
        # Step 5: Remove duplicates
        df = YouTubeDataCleaner.remove_duplicates(df)
        
        logger.info("Full data cleaning pipeline completed")
        
        return df
