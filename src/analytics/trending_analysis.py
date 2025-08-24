"""
Trending analysis utilities for YouTube Analytics
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, count, avg, max as spark_max, min as spark_min, sum as spark_sum, desc, asc
import logging

logger = logging.getLogger(__name__)

class TrendingAnalyzer:
    """Analytics for YouTube trending videos"""
    
    @staticmethod
    def top_videos_by_views(df: DataFrame, limit: int = 10) -> DataFrame:
        """
        Get top videos by view count
        
        Args:
            df: Input DataFrame with video data
            limit: Number of top videos to return
            
        Returns:
            DataFrame with top videos sorted by views
        """
        logger.info(f"Getting top {limit} videos by views")
        
        top_videos = (df.select("video_id", "title", "channel_title", "views", "country", "category_name")
                       .orderBy(desc("views"))
                       .limit(limit))
        
        return top_videos
    
    @staticmethod
    def top_channels_by_trending_count(df: DataFrame, limit: int = 10) -> DataFrame:
        """
        Get top channels by number of trending videos
        
        Args:
            df: Input DataFrame with video data
            limit: Number of top channels to return
            
        Returns:
            DataFrame with top channels by trending video count
        """
        logger.info(f"Getting top {limit} channels by trending count")
        
        top_channels = (df.groupBy("channel_title")
                         .agg(count("video_id").alias("trending_videos_count"),
                              spark_sum("views").alias("total_views"),
                              avg("views").alias("avg_views"))
                         .orderBy(desc("trending_videos_count"))
                         .limit(limit))
        
        return top_channels
    
    @staticmethod
    def category_performance_analysis(df: DataFrame) -> DataFrame:
        """
        Analyze performance metrics by category
        
        Args:
            df: Input DataFrame with video data and category names
            
        Returns:
            DataFrame with category performance metrics
        """
        logger.info("Analyzing category performance")
        
        category_stats = (df.groupBy("category_name")
                           .agg(count("video_id").alias("video_count"),
                                avg("views").alias("avg_views"),
                                avg("likes").alias("avg_likes"),
                                avg("dislikes").alias("avg_dislikes"),
                                avg("comment_count").alias("avg_comments"),
                                spark_max("views").alias("max_views"),
                                spark_min("views").alias("min_views"))
                           .orderBy(desc("avg_views")))
        
        return category_stats
    
    @staticmethod
    def country_comparison_analysis(df: DataFrame) -> DataFrame:
        """
        Compare trending patterns across countries
        
        Args:
            df: Input DataFrame with video data from multiple countries
            
        Returns:
            DataFrame with country comparison metrics
        """
        logger.info("Analyzing trending patterns by country")
        
        country_stats = (df.groupBy("country")
                          .agg(count("video_id").alias("total_trending_videos"),
                               avg("views").alias("avg_views"),
                               avg("likes").alias("avg_likes"),
                               avg("comment_count").alias("avg_comments"),
                               spark_max("views").alias("highest_views"))
                          .orderBy(desc("avg_views")))
        
        return country_stats
    
    @staticmethod
    def engagement_rate_analysis(df: DataFrame) -> DataFrame:
        """
        Calculate engagement rates (likes + comments / views)
        
        Args:
            df: Input DataFrame with video data
            
        Returns:
            DataFrame with engagement rate metrics
        """
        logger.info("Calculating engagement rates")
        
        engagement_df = (df.withColumn("engagement_rate", 
                                      (col("likes") + col("comment_count")) / col("views"))
                          .withColumn("like_rate", col("likes") / col("views"))
                          .withColumn("comment_rate", col("comment_count") / col("views"))
                          .select("video_id", "title", "channel_title", "views", "likes", 
                                 "comment_count", "engagement_rate", "like_rate", "comment_rate",
                                 "category_name", "country")
                          .orderBy(desc("engagement_rate")))
        
        return engagement_df
    
    @staticmethod
    def trending_duration_analysis(df: DataFrame) -> DataFrame:
        """
        Analyze how long videos stay trending (requires multiple trending dates per video)
        
        Args:
            df: Input DataFrame with video data
            
        Returns:
            DataFrame with trending duration analysis
        """
        logger.info("Analyzing trending duration")
        
        trending_duration = (df.groupBy("video_id", "title", "channel_title")
                            .agg(count("trending_date").alias("days_trending"),
                                 spark_max("views").alias("peak_views"))
                            .orderBy(desc("days_trending")))
        
        return trending_duration
    
    @staticmethod
    def generate_trending_summary_report(df: DataFrame) -> dict:
        """
        Generate a comprehensive summary report of trending data
        
        Args:
            df: Input DataFrame with video data
            
        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating trending summary report")
        
        total_videos = df.count()
        total_views = df.agg(spark_sum("views")).collect()[0][0]
        total_likes = df.agg(spark_sum("likes")).collect()[0][0]
        total_comments = df.agg(spark_sum("comment_count")).collect()[0][0]
        
        avg_views = df.agg(avg("views")).collect()[0][0]
        avg_likes = df.agg(avg("likes")).collect()[0][0]
        avg_comments = df.agg(avg("comment_count")).collect()[0][0]
        
        unique_channels = df.select("channel_title").distinct().count()
        unique_categories = df.select("category_name").distinct().count()
        unique_countries = df.select("country").distinct().count()
        
        summary = {
            "total_videos": total_videos,
            "total_views": int(total_views) if total_views else 0,
            "total_likes": int(total_likes) if total_likes else 0,
            "total_comments": int(total_comments) if total_comments else 0,
            "avg_views": round(avg_views, 2) if avg_views else 0,
            "avg_likes": round(avg_likes, 2) if avg_likes else 0,
            "avg_comments": round(avg_comments, 2) if avg_comments else 0,
            "unique_channels": unique_channels,
            "unique_categories": unique_categories,
            "unique_countries": unique_countries
        }
        
        logger.info(f"Summary report generated: {summary}")
        
        return summary
