"""
Feature Engineering module for YouTube Analytics
Contains functions for creating complex metrics and transformations
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, DoubleType
import logging

logger = logging.getLogger(__name__)


class YouTubeFeatureEngineer:
    """Feature engineering class for YouTube Analytics data"""
    
    @staticmethod
    def calculate_engagement_score(df: DataFrame) -> DataFrame:
        """
        Calculate engagement_score using the formula:
        engagement_score = ((likes * 0.5) + (dislikes * 0.2) + (comment_count * 0.3)) / views
        
        Handles division-by-zero errors by setting engagement_score to 0 when views is 0 or null.
        
        Args:
            df: Input DataFrame with likes, dislikes, comment_count, and views columns
            
        Returns:
            DataFrame with added engagement_score column
        """
        logger.info("Calculating engagement_score...")
        
        df_with_engagement = df.withColumn(
            "engagement_score",
            F.when(
                (F.col("views").isNull()) | (F.col("views") == 0),
                0.0
            ).otherwise(
                (
                    (F.coalesce(F.col("likes"), F.lit(0)) * 0.5) +
                    (F.coalesce(F.col("dislikes"), F.lit(0)) * 0.2) +
                    (F.coalesce(F.col("comment_count"), F.lit(0)) * 0.3)
                ) / F.col("views")
            ).cast(DoubleType())
        )
        
        logger.info("Engagement score calculated successfully")
        return df_with_engagement
    
    @staticmethod
    def calculate_days_to_trend(df: DataFrame) -> DataFrame:
        """
        Calculate the number of days between trending_date and publish_time.
        
        Args:
            df: Input DataFrame with trending_date and publish_time_parsed columns
            
        Returns:
            DataFrame with added days_to_trend column
        """
        logger.info("Calculating days_to_trend...")
        
        # Parse trending_date (format: YY.DD.MM) and use existing publish_time_parsed
        df_with_dates = df.withColumn(
            "trending_date_parsed",
            F.to_date(F.col("trending_date"), "yy.dd.MM")
        ).withColumn(
            "publish_date_parsed",
            F.to_date(F.col("publish_time_parsed"))
        )
        
        # Calculate days_to_trend
        df_with_days_to_trend = df_with_dates.withColumn(
            "days_to_trend",
            F.when(
                F.col("trending_date_parsed").isNull() | F.col("publish_date_parsed").isNull(),
                None
            ).otherwise(
                F.datediff(F.col("trending_date_parsed"), F.col("publish_date_parsed"))
            ).cast(IntegerType())
        )
        
        logger.info("Days to trend calculated successfully")
        return df_with_days_to_trend
    
    @staticmethod
    def calculate_trending_rank(df: DataFrame, partition_cols: list = None, order_col: str = "views") -> DataFrame:
        """
        Calculate trending_rank using PySpark Window Functions.
        Ranks videos within each trending_date and category_name based on views in descending order.
        
        Args:
            df: Input DataFrame
            partition_cols: List of columns to partition by (default: ["trending_date", "category_name"])
            order_col: Column to order by (default: "views")
            
        Returns:
            DataFrame with added trending_rank column
        """
        logger.info("Calculating trending_rank using Window Functions...")
        
        if partition_cols is None:
            partition_cols = ["trending_date", "category_name"]
        
        # Define window specification
        window_spec = Window.partitionBy(*partition_cols).orderBy(F.desc(order_col))
        
        # Calculate trending_rank
        df_with_rank = df.withColumn(
            "trending_rank",
            F.row_number().over(window_spec)
        )
        
        logger.info("Trending rank calculated successfully")
        return df_with_rank
    
    @staticmethod
    def add_all_features(df: DataFrame) -> DataFrame:
        """
        Add all engineered features to the DataFrame in one go.
        
        Args:
            df: Input DataFrame with required columns
            
        Returns:
            DataFrame with all engineered features added
        """
        logger.info("Adding all engineered features...")
        
        # Apply all feature engineering steps
        df_with_features = df
        df_with_features = YouTubeFeatureEngineer.calculate_engagement_score(df_with_features)
        df_with_features = YouTubeFeatureEngineer.calculate_days_to_trend(df_with_features)
        df_with_features = YouTubeFeatureEngineer.calculate_trending_rank(df_with_features)
        
        logger.info("All features added successfully")
        return df_with_features
    
    @staticmethod
    def get_feature_statistics(df: DataFrame) -> dict:
        """
        Get comprehensive statistics for all engineered features.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Dictionary containing statistics for each feature
        """
        logger.info("Calculating feature statistics...")
        
        stats = {}
        
        # Engagement Score Statistics
        if "engagement_score" in df.columns:
            engagement_stats = df.select(
                F.count("engagement_score").alias("count"),
                F.mean("engagement_score").alias("mean"),
                F.stddev("engagement_score").alias("stddev"),
                F.min("engagement_score").alias("min"),
                F.max("engagement_score").alias("max"),
                F.expr("percentile_approx(engagement_score, 0.5)").alias("median")
            ).collect()[0]
            stats["engagement_score"] = engagement_stats.asDict()
        
        # Days to Trend Statistics
        if "days_to_trend" in df.columns:
            days_stats = df.select(
                F.count("days_to_trend").alias("count"),
                F.mean("days_to_trend").alias("mean"),
                F.stddev("days_to_trend").alias("stddev"),
                F.min("days_to_trend").alias("min"),
                F.max("days_to_trend").alias("max"),
                F.expr("percentile_approx(days_to_trend, 0.5)").alias("median")
            ).collect()[0]
            stats["days_to_trend"] = days_stats.asDict()
        
        # Trending Rank Statistics
        if "trending_rank" in df.columns:
            rank_stats = df.select(
                F.count("trending_rank").alias("count"),
                F.mean("trending_rank").alias("mean"),
                F.stddev("trending_rank").alias("stddev"),
                F.min("trending_rank").alias("min"),
                F.max("trending_rank").alias("max"),
                F.expr("percentile_approx(trending_rank, 0.5)").alias("median")
            ).collect()[0]
            stats["trending_rank"] = rank_stats.asDict()
        
        logger.info("Feature statistics calculated successfully")
        return stats
    
    @staticmethod
    def analyze_feature_correlations(df: DataFrame) -> DataFrame:
        """
        Analyze correlations between engineered features and original metrics.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            DataFrame with correlation analysis results
        """
        logger.info("Analyzing feature correlations...")
        
        # Calculate correlations between features
        correlations = []
        
        # Engagement score vs trending rank
        if "engagement_score" in df.columns and "trending_rank" in df.columns:
            engagement_by_rank = df.groupBy("trending_rank").agg(
                F.avg("engagement_score").alias("avg_engagement_score"),
                F.count("*").alias("video_count")
            ).filter(F.col("trending_rank") <= 10).orderBy("trending_rank")
            correlations.append(("engagement_vs_rank", engagement_by_rank))
        
        # Days to trend by category
        if "days_to_trend" in df.columns and "category_name" in df.columns:
            days_by_category = df.filter(F.col("days_to_trend").isNotNull()).groupBy("category_name").agg(
                F.avg("days_to_trend").alias("avg_days_to_trend"),
                F.count("*").alias("video_count")
            ).orderBy("avg_days_to_trend")
            correlations.append(("days_by_category", days_by_category))
        
        logger.info("Feature correlation analysis completed")
        return correlations
    
    @staticmethod
    def identify_top_performers(df: DataFrame, 
                              engagement_threshold: float = 0.01,
                              days_threshold: int = 2,
                              rank_threshold: int = 3) -> DataFrame:
        """
        Identify top performing videos based on engineered features.
        
        Args:
            df: DataFrame with engineered features
            engagement_threshold: Minimum engagement score
            days_threshold: Maximum days to trend
            rank_threshold: Maximum trending rank
            
        Returns:
            DataFrame with top performing videos
        """
        logger.info("Identifying top performing videos...")
        
        top_performers = df.filter(
            (F.col("engagement_score") > engagement_threshold) &
            (F.col("days_to_trend") <= days_threshold) &
            (F.col("trending_rank") <= rank_threshold)
        )
        
        logger.info(f"Found {top_performers.count()} top performing videos")
        return top_performers
