#!/usr/bin/env python3
"""
Analytics script for YouTube Analytics project
This script reads processed parquet data and performs comprehensive analysis
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
from src.analytics.trending_analysis import TrendingAnalyzer
from src.data_ingestion.processed_data_loader import ProcessedDataLoader

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main analytics execution - Load parquet data and perform analysis"""
    logger = logging.getLogger(__name__)
    logger.info("📊 Starting YouTube Analytics - Data Analysis")
    logger.info("=" * 60)
    logger.info("Purpose: Load processed parquet data and perform analytics")
    
    try:
        # Get Spark session
        spark = SparkUtils.get_spark_session()
        logger.info("✅ Spark session created successfully")
        
        # Initialize components
        loader = ProcessedDataLoader(spark)
        analyzer = TrendingAnalyzer()
        
        # Check if processed data exists
        logger.info("🔍 Checking for processed data...")
        validation_results = loader.validate_processed_data()
        
        if not validation_results["videos_data_exists"]:
            logger.error("❌ No processed video data found!")
            logger.error("💡 Please run 'python scripts/run_pipeline.py' first to process the raw data")
            return 1
        
        logger.info("✅ Processed data found!")
        
        # Load processed data
        logger.info("📂 Loading processed video data from parquet...")
        videos_df = loader.load_processed_videos()
        
        logger.info("📂 Loading category mappings...")
        try:
            categories_df = loader.load_category_mappings()
            logger.info(f"✅ Loaded {categories_df.count():,} category mappings")
        except FileNotFoundError:
            logger.warning("⚠️  Category mappings not found, continuing without them")
            categories_df = None
        
        # Data summary
        logger.info("=" * 60)
        logger.info("📋 DATA SUMMARY")
        logger.info("=" * 60)
        
        summary = loader.get_data_summary()
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:,}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Analytics Section 1: Overall Statistics
        logger.info("\n" + "=" * 60)
        logger.info("📈 OVERALL ANALYTICS")
        logger.info("=" * 60)
        
        analytics_summary = analyzer.generate_trending_summary_report(videos_df)
        logger.info("📊 Trending Summary:")
        for key, value in analytics_summary.items():
            if isinstance(value, (int, float)):
                logger.info(f"   {key}: {value:,}")
            else:
                logger.info(f"   {key}: {value}")
        
        # Analytics Section 2: Top Videos
        logger.info("\n🏆 TOP 10 VIDEOS BY VIEWS:")
        top_videos = analyzer.top_videos_by_views(videos_df, limit=10)
        top_videos.select(
            "title", "channel_title", "views", "country", "category_name"
        ).show(10, truncate=False)
        
        # Analytics Section 3: Top Channels
        logger.info("\n🎬 TOP 10 CHANNELS BY TRENDING COUNT:")
        top_channels = analyzer.top_channels_by_trending_count(videos_df, limit=10)
        top_channels.show(10, truncate=False)
        
        # Analytics Section 4: Category Performance
        logger.info("\n📊 CATEGORY PERFORMANCE ANALYSIS:")
        category_performance = analyzer.category_performance_analysis(videos_df)
        category_performance.show(15, truncate=False)
        
        # Analytics Section 5: Country Comparison
        logger.info("\n🌍 COUNTRY COMPARISON ANALYSIS:")
        country_comparison = analyzer.country_comparison_analysis(videos_df)
        country_comparison.show(truncate=False)
        
        # Analytics Section 6: Engagement Analysis
        logger.info("\n💬 TOP 10 VIDEOS BY ENGAGEMENT RATE:")
        engagement_analysis = analyzer.engagement_rate_analysis(videos_df)
        engagement_analysis.select(
            "title", "channel_title", "views", "engagement_rate", 
            "like_rate", "comment_rate", "category_name"
        ).show(10, truncate=False)
        
        # Analytics Section 7: Trending Duration
        logger.info("\n⏱️  VIDEOS WITH MOST TRENDING DAYS:")
        trending_duration = analyzer.trending_duration_analysis(videos_df)
        trending_duration.show(10, truncate=False)
        
        # Data Quality Report
        logger.info("\n" + "=" * 60)
        logger.info("🔍 DATA QUALITY REPORT")
        logger.info("=" * 60)
        
        quality_metrics = validation_results.get("data_quality", {})
        for metric, value in quality_metrics.items():
            logger.info(f"   {metric}: {value}")
        
        # Completion
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ANALYTICS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("📊 All analytics have been generated and displayed above")
        logger.info("💡 You can now use this data for further insights or reporting")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Analytics failed: {e}")
        logger.error("💡 Possible solutions:")
        logger.error("   - Ensure processed data exists (run pipeline first)")
        logger.error("   - Check Spark configuration")
        logger.error("   - Verify data file permissions")
        raise
    finally:
        SparkUtils.stop_spark_session()
        logger.info("🔌 Spark session stopped")

if __name__ == "__main__":
    setup_logging()
    exit_code = main()
    sys.exit(exit_code)
