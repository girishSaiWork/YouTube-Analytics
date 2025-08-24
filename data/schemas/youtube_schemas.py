"""
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
