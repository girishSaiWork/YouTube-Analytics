#!/usr/bin/env python3
"""
Main pipeline script for YouTube Analytics
"""

import sys
import logging
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Change working directory to project root
import os
os.chdir(str(project_root))

from config.settings import Config
from src.utils.spark_utils import SparkUtils
from src.data_processing.pipeline import YouTubeDataPipeline

def setup_logging():
    """Setup logging configuration"""
    try:
        # Try to create log file handler
        log_file = Path(Config.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Config.LOG_FILE),
                logging.StreamHandler(sys.stdout)
            ]
        )
    except Exception:
        # Fallback to console only if file logging fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

def main():
    """Main pipeline execution - Process raw data and save to parquet"""
    logger = logging.getLogger(__name__)
    logger.info("Starting YouTube Analytics Data Processing Pipeline")
    logger.info("=" * 60)
    logger.info("Purpose: Transform raw CSV/JSON data to processed format")

    try:
        # Get Spark session
        spark = SparkUtils.get_spark_session()
        logger.info("Spark session created successfully")

        # Initialize pipeline
        pipeline = YouTubeDataPipeline(spark)

        # Define countries to process (you can modify this list)
        countries_to_process = ['US']  # Start with US only for testing
        logger.info(f"Processing countries: {countries_to_process}")

        # Run data processing pipeline
        logger.info("Starting data transformation pipeline...")
        logger.info("   1. Reading raw CSV files")
        logger.info("   2. Applying transformations (tags cleaning, type conversion)")
        logger.info("   3. Adding category names from JSON files")
        logger.info("   4. Applying data cleaning and validation")
        logger.info("   5. Saving processed data")

        # Process data with Pandas I/O support (Windows-friendly approach)
        logger.info("Processing data with Windows-compatible I/O...")
        processed_df = pipeline.run_full_pipeline(
            countries=countries_to_process,
            save_output=True  # This will now generate both parquet and CSV outputs
        )

        logger.info("Data processed and saved using Pandas I/O (Windows-compatible)")

        # Pipeline completion summary
        total_records = processed_df.count()
        logger.info("=" * 60)
        logger.info("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Total processed records: {total_records:,}")
        logger.info(f"Countries processed: {len(countries_to_process)}")
        logger.info("Generated outputs:")
        logger.info("   - youtube_trending_videos.parquet")
        logger.info("   - youtube_trending_videos.csv")
        logger.info("   - category_mappings.parquet")
        logger.info("   - category_mappings.csv")

        # Show sample of processed data
        logger.info("\nSample of processed data:")
        processed_df.select(
            "video_id", "title", "channel_title", "views", "likes",
            "category_name", "country"
        ).show(5, truncate=False)

        logger.info("=" * 60)
        logger.info("Pipeline completed. Data is ready for analysis!")
        logger.info("Next step: Run 'python scripts/analyze_processed_data.py' for analytics")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error("Check the error above and ensure:")
        logger.error("   - Raw data files exist in data/raw/")
        logger.error("   - Spark is properly configured")
        logger.error("   - Sufficient disk space available")
        raise
    finally:
        SparkUtils.stop_spark_session()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    setup_logging()
    main()
