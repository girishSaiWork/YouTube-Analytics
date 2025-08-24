#!/usr/bin/env python3
"""
Feature Engineering Demo Script for YouTube Analytics
Demonstrates the use of advanced PySpark functions for feature engineering
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
from src.data_ingestion.processed_data_loader import ProcessedDataLoader
from src.data_processing.feature_engineering import YouTubeFeatureEngineer

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    """Main feature engineering demonstration"""
    logger = logging.getLogger(__name__)
    logger.info("Starting YouTube Analytics Feature Engineering Demo")
    logger.info("=" * 60)

    try:
        # Get Spark session
        spark = SparkUtils.get_spark_session()
        logger.info("Spark session created successfully")

        # Initialize data loader
        loader = ProcessedDataLoader(spark)
        config = Config()

        # Load processed data
        logger.info("Loading processed YouTube trending videos data...")
        df = loader.load_processed_videos()

        if df is None:
            raise ValueError("Could not load processed data. Please run the pipeline first.")

        logger.info(f"Loaded {df.count():,} records")

        # Initialize feature engineer
        feature_engineer = YouTubeFeatureEngineer()

        # Task 1: Calculate engagement score
        logger.info("\n" + "="*50)
        logger.info("TASK 1: Calculating Engagement Score")
        logger.info("="*50)
        
        df_with_engagement = feature_engineer.calculate_engagement_score(df)
        
        # Show sample results
        logger.info("Sample engagement scores:")
        df_with_engagement.select(
            "video_id", "title", "views", "likes", "dislikes", "comment_count", "engagement_score"
        ).orderBy("engagement_score", ascending=False).show(5, truncate=False)

        # Task 2: Calculate days to trend
        logger.info("\n" + "="*50)
        logger.info("TASK 2: Calculating Days to Trend")
        logger.info("="*50)
        
        df_with_days = feature_engineer.calculate_days_to_trend(df_with_engagement)
        
        # Show sample results
        logger.info("Sample days to trend calculations:")
        df_with_days.select(
            "video_id", "title", "trending_date", "publish_time", "days_to_trend"
        ).filter(df_with_days.days_to_trend.isNotNull()).show(5, truncate=False)

        # Task 3: Calculate trending rank
        logger.info("\n" + "="*50)
        logger.info("TASK 3: Calculating Trending Rank")
        logger.info("="*50)
        
        df_with_rank = feature_engineer.calculate_trending_rank(df_with_days)
        
        # Show sample results
        logger.info("Sample trending ranks (Top 3 per category per day):")
        df_with_rank.select(
            "trending_date", "category_name", "trending_rank", "title", "views"
        ).filter(df_with_rank.trending_rank <= 3).orderBy(
            "trending_date", "category_name", "trending_rank"
        ).show(10, truncate=False)

        # Get comprehensive statistics
        logger.info("\n" + "="*50)
        logger.info("FEATURE STATISTICS")
        logger.info("="*50)
        
        stats = feature_engineer.get_feature_statistics(df_with_rank)
        
        for feature_name, feature_stats in stats.items():
            logger.info(f"\n{feature_name.upper()} Statistics:")
            for stat_name, stat_value in feature_stats.items():
                if isinstance(stat_value, float):
                    logger.info(f"  {stat_name}: {stat_value:.6f}")
                else:
                    logger.info(f"  {stat_name}: {stat_value}")

        # Analyze correlations
        logger.info("\n" + "="*50)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*50)
        
        correlations = feature_engineer.analyze_feature_correlations(df_with_rank)
        
        for correlation_name, correlation_df in correlations:
            logger.info(f"\n{correlation_name.upper()}:")
            correlation_df.show(10, truncate=False)

        # Identify top performers
        logger.info("\n" + "="*50)
        logger.info("TOP PERFORMERS ANALYSIS")
        logger.info("="*50)
        
        top_performers = feature_engineer.identify_top_performers(df_with_rank)
        
        if top_performers.count() > 0:
            logger.info(f"Found {top_performers.count()} top performing videos:")
            top_performers.select(
                "title", "channel_title", "category_name", "views",
                "engagement_score", "days_to_trend", "trending_rank"
            ).orderBy("engagement_score", ascending=False).show(10, truncate=False)
        else:
            logger.info("No videos meet the strict top performer criteria.")
            logger.info("Trying with relaxed criteria...")
            
            relaxed_performers = feature_engineer.identify_top_performers(
                df_with_rank, 
                engagement_threshold=0.005,
                days_threshold=5,
                rank_threshold=5
            )
            
            logger.info(f"Found {relaxed_performers.count()} videos with relaxed criteria:")
            relaxed_performers.select(
                "title", "channel_title", "category_name", "views",
                "engagement_score", "days_to_trend", "trending_rank"
            ).orderBy("engagement_score", ascending=False).show(10, truncate=False)

        # Save the engineered features
        logger.info("\n" + "="*50)
        logger.info("SAVING ENGINEERED FEATURES")
        logger.info("="*50)

        output_base_path = config.OUTPUT_DATA_PATH / "youtube_trending_videos_with_features"
        feature_engineer.save_engineered_features(
            df_with_rank,
            str(output_base_path),
            save_csv=True,
            save_parquet=True
        )

        # Business Insights Analysis using Pandas
        logger.info("\n" + "="*50)
        logger.info("BUSINESS INSIGHTS ANALYSIS (PANDAS)")
        logger.info("="*50)

        from src.analytics.business_insights import YouTubeBusinessInsights

        # Initialize business insights analyzer
        insights_analyzer = YouTubeBusinessInsights(str(output_base_path) + '.parquet')

        # Generate comprehensive report
        report = insights_analyzer.generate_comprehensive_report(
            save_path=str(config.OUTPUT_DATA_PATH / "business_insights_report.json")
        )

        # Display key insights
        logger.info("\nKEY BUSINESS INSIGHTS:")
        for insight in report['key_insights']:
            logger.info(f"• {insight}")

        # Final summary
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("Summary of engineered features:")
        logger.info("1. engagement_score: Weighted metric combining likes, dislikes, and comments relative to views")
        logger.info("2. days_to_trend: Number of days between publish_time and trending_date")
        logger.info("3. trending_rank: Rank of videos within each trending_date and category based on views")
        logger.info("\nGenerated outputs:")
        logger.info(f"• {output_base_path}.parquet - Engineered features (parquet)")
        logger.info(f"• {output_base_path}.csv - Engineered features (CSV)")
        logger.info(f"• {config.OUTPUT_DATA_PATH / 'business_insights_report.json'} - Business insights report")
        logger.info("\nAll features have been successfully calculated, analyzed, and saved!")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        logger.error("Check the error above and ensure:")
        logger.error("   - Processed data files exist in data/processed/")
        logger.error("   - Spark is properly configured")
        logger.error("   - Required columns are present in the data")
        raise
    finally:
        SparkUtils.stop_spark_session()
        logger.info("Spark session stopped")

if __name__ == "__main__":
    setup_logging()
    main()
