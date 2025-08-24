"""
Complete data processing pipeline for YouTube Analytics
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit
from src.data_ingestion.readers import YouTubeDataReader
from src.data_processing.transformers import YouTubeDataTransformer
from src.data_processing.cleaners import YouTubeDataCleaner
from config.settings import Config
from src.utils.pandas_io import PandasIOHandler, should_use_pandas_io
import logging

logger = logging.getLogger(__name__)

class YouTubeDataPipeline:
    """Complete data processing pipeline for YouTube Analytics"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.reader = YouTubeDataReader(spark)
        self.config = Config()
        self.pandas_io = PandasIOHandler(spark)
        self.use_pandas_io = should_use_pandas_io()
    
    def process_single_country_data(self, country_code: str, apply_cleaning: bool = True) -> DataFrame:
        """
        Process data for a single country with full transformation pipeline
        
        Args:
            country_code: Country code (e.g., 'US', 'CA')
            apply_cleaning: Whether to apply data cleaning steps
            
        Returns:
            Processed DataFrame for the country
        """
        logger.info(f"Processing data for country: {country_code}")
        
        # Step 1: Read raw video data
        df = self.reader.read_videos_data(country_code)
        logger.info(f"Loaded {df.count()} raw videos for {country_code}")
        
        # Step 2: Apply transformations
        df = self._apply_transformations(df)
        
        # Step 3: Add category names
        df = self._add_category_names(df, country_code)
        
        # Step 4: Apply cleaning if requested
        if apply_cleaning:
            df = self._apply_cleaning(df)
        
        logger.info(f"Processing completed for {country_code}")
        return df
    
    def process_all_countries_data(self, apply_cleaning: bool = True) -> DataFrame:
        """
        Process data for all available countries
        
        Args:
            apply_cleaning: Whether to apply data cleaning steps
            
        Returns:
            Combined DataFrame for all countries
        """
        logger.info("Processing data for all countries")
        
        country_dfs = []
        
        for country in self.config.COUNTRIES:
            try:
                df = self.process_single_country_data(country, apply_cleaning=False)
                country_dfs.append(df)
                logger.info(f"Successfully processed {country}")
            except Exception as e:
                logger.warning(f"Failed to process {country}: {e}")
                continue
        
        if not country_dfs:
            raise ValueError("No country data could be processed")
        
        # Combine all countries
        logger.info("Combining data from all countries")
        combined_df = country_dfs[0]
        for df in country_dfs[1:]:
            combined_df = combined_df.union(df)
        
        # Apply cleaning to combined dataset if requested
        if apply_cleaning:
            combined_df = self._apply_cleaning(combined_df)
        
        logger.info(f"Combined processing completed. Total records: {combined_df.count()}")
        return combined_df
    
    def _apply_transformations(self, df: DataFrame) -> DataFrame:
        """Apply data transformations"""
        logger.info("Applying data transformations")
        
        # Clean tags column
        df = YouTubeDataTransformer.clean_tags_column(df)
        
        # Convert data types
        df = YouTubeDataTransformer.convert_numeric_columns(df)
        df = YouTubeDataTransformer.convert_boolean_columns(df)
        
        return df
    
    def _add_category_names(self, df: DataFrame, country_code: str) -> DataFrame:
        """Add category names to the DataFrame"""
        logger.info(f"Adding category names for {country_code}")
        
        try:
            # Get category mapping
            category_mapping = self.reader.read_and_flatten_categories_data(country_code)
            
            # Join with video data
            df = YouTubeDataTransformer.add_category_names(df, category_mapping)
            
        except Exception as e:
            logger.warning(f"Could not add category names for {country_code}: {e}")
            # Add a default category_name column if joining fails
            df = df.withColumn("category_name", lit("Unknown"))
        
        return df
    
    def _apply_cleaning(self, df: DataFrame) -> DataFrame:
        """Apply data cleaning steps"""
        logger.info("Applying data cleaning")
        
        # Apply full cleaning pipeline
        df = YouTubeDataCleaner.apply_full_cleaning_pipeline(df)
        
        # Validate data quality
        quality_metrics = YouTubeDataCleaner.validate_data_quality(df)
        logger.info(f"Data quality metrics: {quality_metrics}")
        
        return df
    
    def save_processed_data(self, df: DataFrame, output_path: str, format: str = "parquet") -> None:
        """
        Save processed data to specified path using environment-appropriate method

        Args:
            df: DataFrame to save
            output_path: Path to save the data
            format: Output format (parquet, csv, json)
        """
        logger.info(f"Saving processed data to {output_path} in {format} format")

        # Use Pandas I/O for Windows to avoid Hadoop native library issues
        if self.use_pandas_io and format.lower() in ["parquet", "csv"]:
            logger.info("Using Pandas I/O for Windows compatibility")

            if format.lower() == "parquet":
                success = self.pandas_io.save_spark_df_as_parquet(df, output_path)
            elif format.lower() == "csv":
                success = self.pandas_io.save_spark_df_as_csv(df, output_path)

            if success:
                logger.info(f"Data saved successfully using Pandas to {output_path}")
                return
            else:
                logger.warning("Pandas save failed, falling back to Spark native")

        # Use native Spark I/O (Linux/Docker/WSL)
        logger.info("Using native Spark I/O")

        # Ensure output directory exists
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            if format.lower() == "parquet":
                df.write.mode("overwrite").parquet(output_path)
            elif format.lower() == "csv":
                df.write.mode("overwrite").option("header", "true").csv(output_path)
            elif format.lower() == "json":
                df.write.mode("overwrite").json(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Data saved successfully to {output_path}")

        except Exception as e:
            if "UnsatisfiedLinkError" in str(e):
                logger.error(f"Native Spark save failed: {e}")
                logger.error("Consider running in Docker/WSL for full Spark compatibility")
                raise
            else:
                raise

    def save_category_mappings(self, countries: list = None) -> None:
        """
        Save category mappings for all countries to processed data folder

        Args:
            countries: List of countries to process. If None, processes all countries
        """
        if countries is None:
            countries = self.config.COUNTRIES

        logger.info(f"Saving category mappings for countries: {countries}")

        all_categories = []

        for country in countries:
            try:
                # Get flattened category mapping
                category_df = self.reader.read_and_flatten_categories_data(country)
                # Add country column
                category_df = category_df.withColumn("country", lit(country))
                all_categories.append(category_df)
                logger.info(f"Processed categories for {country}")
            except Exception as e:
                logger.warning(f"Failed to process categories for {country}: {e}")
                continue

        if all_categories:
            # Combine all category mappings
            combined_categories = all_categories[0]
            for cat_df in all_categories[1:]:
                combined_categories = combined_categories.union(cat_df)

            # Save to processed data folder
            output_path = str(self.config.PROCESSED_DATA_PATH / "category_mappings")
            self.save_processed_data(combined_categories, output_path, "parquet")
            logger.info(f"Saved category mappings for {len(all_categories)} countries")
        else:
            logger.warning("No category mappings could be processed")

    def save_category_mappings_with_pandas(self, countries: list = None) -> None:
        """
        Save category mappings in both parquet and CSV formats using pandas

        Args:
            countries: List of countries to process. If None, processes all countries
        """
        if countries is None:
            countries = self.config.COUNTRIES

        logger.info(f"Processing category mappings for countries: {countries}")

        all_categories = []

        for country in countries:
            try:
                # Get flattened category mapping
                category_df = self.reader.read_and_flatten_categories_data(country)
                # Add country column
                category_df = category_df.withColumn("country", lit(country))
                all_categories.append(category_df)
                logger.info(f"Processed categories for {country}")
            except Exception as e:
                logger.warning(f"Failed to process categories for {country}: {e}")
                continue

        if all_categories:
            # Combine all category mappings
            combined_categories = all_categories[0]
            for cat_df in all_categories[1:]:
                combined_categories = combined_categories.union(cat_df)

            # Convert to pandas for saving
            logger.info("Converting category mappings to pandas for file operations")
            pandas_categories = combined_categories.toPandas()

            # Save as parquet using pandas
            cat_parquet_path = self.config.PROCESSED_DATA_PATH / "category_mappings.parquet"
            logger.info(f"Saving category_mappings.parquet to {cat_parquet_path}")
            try:
                pandas_categories.to_parquet(cat_parquet_path, index=False, engine='pyarrow')
                logger.info(f"Successfully saved {len(pandas_categories)} category records to {cat_parquet_path}")
            except ImportError:
                logger.warning("PyArrow not available for parquet, using CSV only")

            # Save as CSV using pandas
            cat_csv_path = self.config.PROCESSED_DATA_PATH / "category_mappings.csv"
            logger.info(f"Saving category_mappings.csv to {cat_csv_path}")
            pandas_categories.to_csv(cat_csv_path, index=False)
            logger.info(f"Successfully saved {len(pandas_categories)} category records to {cat_csv_path}")

            logger.info(f"Saved category mappings for {len(all_categories)} countries in both formats using pandas")
        else:
            logger.warning("No category mappings could be processed")

    def run_full_pipeline(self, countries: list = None, save_output: bool = True) -> DataFrame:
        """
        Run the complete data processing pipeline

        Args:
            countries: List of countries to process. If None, processes all countries
            save_output: Whether to save the processed data

        Returns:
            Processed DataFrame
        """
        logger.info("Starting full YouTube Analytics pipeline")

        if countries is None:
            # Process all countries
            df = self.process_all_countries_data(apply_cleaning=True)
        else:
            # Process specific countries
            country_dfs = []
            for country in countries:
                try:
                    country_df = self.process_single_country_data(country, apply_cleaning=False)
                    country_dfs.append(country_df)
                except Exception as e:
                    logger.warning(f"Failed to process {country}: {e}")
                    continue

            if not country_dfs:
                raise ValueError("No countries could be processed")

            # Combine and clean
            df = country_dfs[0]
            for country_df in country_dfs[1:]:
                df = df.union(country_df)

            df = self._apply_cleaning(df)

        # Save output if requested
        if save_output:
            self.save_final_outputs(df, countries)

        logger.info("Full pipeline completed successfully")
        return df

    def save_final_outputs(self, df: DataFrame, countries: list = None) -> None:
        """
        Save final outputs in the required formats using pandas:
        - youtube_trending_videos in parquet and CSV
        - category_mappings by country in parquet and CSV

        Args:
            df: Processed DataFrame
            countries: List of countries processed
        """
        logger.info("Saving final outputs in required formats using pandas")

        # Convert Spark DataFrame to Pandas for saving
        logger.info("Converting Spark DataFrame to Pandas for file operations")
        pandas_df = df.toPandas()

        # 1. Save youtube_trending_videos in both formats
        video_parquet_path = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos.parquet"
        video_csv_path = self.config.PROCESSED_DATA_PATH / "youtube_trending_videos.csv"

        # Save as parquet using pandas
        logger.info(f"Saving youtube_trending_videos.parquet to {video_parquet_path}")
        try:
            pandas_df.to_parquet(video_parquet_path, index=False, engine='pyarrow')
            logger.info(f"Successfully saved {len(pandas_df)} records to {video_parquet_path}")
        except ImportError:
            logger.warning("PyArrow not available for parquet, using CSV only")

        # Save as CSV using pandas
        logger.info(f"Saving youtube_trending_videos.csv to {video_csv_path}")
        pandas_df.to_csv(video_csv_path, index=False)
        logger.info(f"Successfully saved {len(pandas_df)} records to {video_csv_path}")

        # 2. Save category mappings by country using pandas
        self.save_category_mappings_with_pandas(countries)

        logger.info("All final outputs saved successfully using pandas")
        logger.info("Generated files:")
        logger.info("  - youtube_trending_videos.parquet")
        logger.info("  - youtube_trending_videos.csv")
        logger.info("  - category_mappings.parquet")
        logger.info("  - category_mappings.csv")
