"""
Data transformation utilities for YouTube Analytics
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import regexp_replace, col, explode_outer, split, array, lit
from pyspark.sql.types import ArrayType, StructType
import logging

logger = logging.getLogger(__name__)

class YouTubeDataTransformer:
    """Data transformation utilities for YouTube datasets"""
    
    @staticmethod
    def clean_tags_column(df: DataFrame, tags_column: str = "tags") -> DataFrame:
        """
        Clean and transform the tags column from pipe-separated string to array format
        
        Args:
            df: Input DataFrame
            tags_column: Name of the tags column to clean
            
        Returns:
            DataFrame with cleaned tags column
        """
        logger.info(f"Cleaning tags column: {tags_column}")
        
        # Replace pipes with commas and remove quotes
        df_cleaned = (df.withColumn(tags_column, regexp_replace(tags_column, "\\|", ","))
                       .withColumn(tags_column, regexp_replace(tags_column, '"', "")))
        
        logger.info("Tags column cleaned successfully")
        return df_cleaned
    
    @staticmethod
    def convert_tags_to_array(df: DataFrame, tags_column: str = "tags") -> DataFrame:
        """
        Convert comma-separated tags string to array
        
        Args:
            df: Input DataFrame
            tags_column: Name of the tags column to convert
            
        Returns:
            DataFrame with tags as array type
        """
        logger.info(f"Converting tags column to array: {tags_column}")
        
        df_array = df.withColumn(tags_column, split(col(tags_column), ","))
        
        logger.info("Tags column converted to array successfully")
        return df_array
    
    @staticmethod
    def json_flattener(df: DataFrame, depth: int = 0, max_depth: int = 10) -> DataFrame:
        """
        Recursively flattens a DataFrame with nested JSON structures.
        
        This function handles complex nested JSON by:
        1. Detecting ArrayType and StructType columns
        2. Exploding arrays using explode_outer (preserves nulls)
        3. Flattening structs by creating new columns for each field
        4. Recursively processing until no complex fields remain
        
        Args:
            df: The DataFrame to flatten
            depth: Current recursion depth (for logging)
            max_depth: Maximum recursion depth to prevent infinite loops
            
        Returns:
            DataFrame: The flattened DataFrame with all complex fields expanded into separate columns.
        """
        if depth > max_depth:
            logger.warning(f"Maximum recursion depth {max_depth} reached. Stopping flattening.")
            return df
            
        indent = "  " * depth  # for pretty logging

        # Detect complex fields (StructType or ArrayType)
        complex_fields = {}
        for field in df.schema.fields:
            if isinstance(field.dataType, (ArrayType, StructType)):
                complex_fields[field.name] = field.dataType

        if not complex_fields:
            logger.info(f"{indent}No more complex fields found at depth {depth}.")
            return df

        logger.info(f"{indent}Found complex fields at depth {depth}: {list(complex_fields.keys())}")

        for field_name, field_type in complex_fields.items():
            if isinstance(field_type, StructType):
                logger.info(f"{indent}Flattening StructType column '{field_name}'")
                for sub_field in field_type.fields:
                    new_col_name = f"{field_name}_{sub_field.name}"
                    logger.info(f"{indent}  Creating column: {new_col_name}")
                    df = df.withColumn(new_col_name, col(f"{field_name}.{sub_field.name}"))
                df = df.drop(field_name)
                logger.info(f"{indent}Dropped original Struct column '{field_name}'")
            elif isinstance(field_type, ArrayType):
                logger.info(f"{indent}Exploding ArrayType column '{field_name}' with explode_outer")
                df = df.withColumn(field_name, explode_outer(col(field_name)))

        logger.info(f"{indent}Recursing into flattened DataFrame at depth {depth + 1}")
        return YouTubeDataTransformer.json_flattener(df, depth=depth + 1, max_depth=max_depth)
    
    @staticmethod
    def extract_category_mapping(flattened_category_df: DataFrame) -> DataFrame:
        """
        Extract clean category ID to name mapping from flattened category DataFrame
        
        Args:
            flattened_category_df: Flattened category DataFrame from json_flattener
            
        Returns:
            DataFrame with columns: category_id, category_name
        """
        logger.info("Extracting category mapping from flattened DataFrame")
        
        category_mapping = (flattened_category_df
                           .select(col("items_id").alias("category_id"),
                                  col("items_snippet_title").alias("category_name"))
                           .distinct())
        
        logger.info(f"Extracted {category_mapping.count()} category mappings")
        return category_mapping
    
    @staticmethod
    def add_category_names(video_df: DataFrame, category_mapping_df: DataFrame) -> DataFrame:
        """
        Join video data with category names
        
        Args:
            video_df: Video DataFrame with category_id column
            category_mapping_df: Category mapping DataFrame with category_id and category_name
            
        Returns:
            DataFrame with category names added
        """
        logger.info("Adding category names to video data")
        
        # Join on category_id
        enriched_df = video_df.join(
            category_mapping_df,
            video_df.category_id == category_mapping_df.category_id,
            "left"
        ).drop(category_mapping_df.category_id)  # Remove duplicate category_id column
        
        logger.info("Category names added successfully")
        return enriched_df
    
    @staticmethod
    def convert_numeric_columns(df: DataFrame) -> DataFrame:
        """
        Convert string columns that should be numeric to proper types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with proper numeric types
        """
        logger.info("Converting string columns to numeric types")
        
        # Define columns that should be numeric
        numeric_columns = ["views", "likes", "dislikes", "comment_count"]
        
        for col_name in numeric_columns:
            if col_name in df.columns:
                logger.info(f"Converting {col_name} to integer")
                df = df.withColumn(col_name, col(col_name).cast("integer"))
        
        logger.info("Numeric conversion completed")
        return df
    
    @staticmethod
    def convert_boolean_columns(df: DataFrame) -> DataFrame:
        """
        Convert string boolean columns to proper boolean types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with proper boolean types
        """
        logger.info("Converting string columns to boolean types")
        
        # Define columns that should be boolean
        boolean_columns = ["comments_disabled", "ratings_disabled", "video_error_or_removed"]
        
        for col_name in boolean_columns:
            if col_name in df.columns:
                logger.info(f"Converting {col_name} to boolean")
                df = df.withColumn(col_name, 
                                  (col(col_name) == "True") | (col(col_name) == "true"))
        
        logger.info("Boolean conversion completed")
        return df
