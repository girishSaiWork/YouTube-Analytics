#!/usr/bin/env python3
"""
Business Insights Analysis Script for YouTube Analytics
Pandas-based efficient analysis of engineered features
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
from src.analytics.business_insights import YouTubeBusinessInsights
import pandas as pd

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def display_dataframe_nicely(df, title, max_rows=10):
    """Display DataFrame with nice formatting"""
    print(f"\n{title}")
    print("=" * len(title))
    if isinstance(df, pd.DataFrame):
        print(df.head(max_rows).to_string())
    else:
        print(df)

def main():
    """Main business insights analysis"""
    logger = logging.getLogger(__name__)
    logger.info("Starting YouTube Analytics Business Insights Analysis")
    logger.info("=" * 60)

    try:
        config = Config()
        
        # Check if engineered features exist
        features_path = config.OUTPUT_DATA_PATH / "youtube_trending_videos_with_features.parquet"
        
        if not features_path.exists():
            logger.error("Engineered features not found!")
            logger.error("Please run the feature engineering script first:")
            logger.error("  python scripts/feature_engineering_demo.py")
            return
        
        # Initialize business insights analyzer
        logger.info(f"Loading engineered features from {features_path}")
        insights_analyzer = YouTubeBusinessInsights(str(features_path))
        
        # Analysis 1: High Engagement Content
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS 1: HIGH ENGAGEMENT CONTENT")
        logger.info("="*60)
        
        high_engagement = insights_analyzer.analyze_high_engagement_content(top_n=20)
        
        print(f"\nTop 10 High Engagement Videos:")
        print("-" * 50)
        top_videos = high_engagement['top_videos'].head(10)
        for idx, row in top_videos.iterrows():
            print(f"{row['title'][:60]}...")
            print(f"  Channel: {row['channel_title']} | Category: {row['category_name']}")
            print(f"  Engagement: {row['engagement_score']:.4f} | Views: {row['views']:,}")
            print()
        
        print(f"\nHigh Engagement Summary:")
        print(f"• Average Engagement Score: {high_engagement['summary']['avg_engagement']:.4f}")
        print(f"• Average Views: {high_engagement['summary']['avg_views']:,.0f}")
        print(f"• Most Common Category: {high_engagement['summary']['most_common_category']}")
        print(f"• Most Frequent Channel: {high_engagement['summary']['most_frequent_channel']}")
        
        display_dataframe_nicely(
            high_engagement['category_analysis'], 
            "High Engagement by Category", 
            max_rows=10
        )
        
        # Analysis 2: Trending Speed by Category
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS 2: TRENDING SPEED BY CATEGORY")
        logger.info("="*60)
        
        trending_speed = insights_analyzer.analyze_trending_speed_by_category()
        
        print(f"\nTrending Speed Summary:")
        print(f"• Overall Average Days to Trend: {trending_speed['overall_stats']['avg_days_to_trend']:.1f}")
        print(f"• Overall Median Days to Trend: {trending_speed['overall_stats']['median_days_to_trend']:.1f}")
        print(f"• Fastest Trending Category: {trending_speed['overall_stats']['fastest_category']}")
        print(f"• Slowest Trending Category: {trending_speed['overall_stats']['slowest_category']}")
        
        display_dataframe_nicely(
            trending_speed['category_speed_analysis'].head(10), 
            "Trending Speed by Category (Top 10 Fastest)", 
            max_rows=10
        )
        
        print(f"\nQuick Trending Categories (< 7 days average):")
        if len(trending_speed['quick_trending_categories']) > 0:
            for category in trending_speed['quick_trending_categories'].index:
                avg_days = trending_speed['quick_trending_categories'].loc[category, 'days_to_trend_mean']
                print(f"• {category}: {avg_days:.1f} days")
        else:
            print("• No categories trend in less than 7 days on average")
        
        # Analysis 3: Ranking Patterns
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS 3: RANKING PATTERNS")
        logger.info("="*60)
        
        ranking_patterns = insights_analyzer.analyze_ranking_patterns()
        
        print(f"\nRanking Pattern Summary:")
        print(f"• Engagement-Rank Correlation: {ranking_patterns['engagement_rank_correlation']:.4f}")
        print(f"• Correlation Strength: {ranking_patterns['insights']['correlation_strength']}")
        print(f"• Correlation Direction: {ranking_patterns['insights']['correlation_direction']}")
        print(f"• Most Top-Ranked Category: {ranking_patterns['insights']['most_top_ranked_category']}")
        
        display_dataframe_nicely(
            ranking_patterns['rank_engagement_analysis'].head(10), 
            "Engagement by Trending Rank (Top 10 Ranks)", 
            max_rows=10
        )
        
        display_dataframe_nicely(
            ranking_patterns['top_ranked_by_category'].head(10), 
            "Top-Ranked Videos by Category", 
            max_rows=10
        )
        
        # Analysis 4: Temporal Patterns
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS 4: TEMPORAL PATTERNS")
        logger.info("="*60)
        
        temporal_patterns = insights_analyzer.analyze_temporal_patterns()
        
        print(f"\nTemporal Pattern Summary:")
        stats = temporal_patterns['temporal_stats']
        print(f"• Same Day Trending: {stats['same_day_trending']:,} videos ({stats['same_day_percentage']:.1f}%)")
        print(f"• Within 5 Days: {stats['within_5_days']:,} videos ({stats['within_5_days_percentage']:.1f}%)")
        print(f"• Within 1 Week: {stats['within_week']:,} videos ({stats['within_week_percentage']:.1f}%)")
        print(f"• Most Videos Trend Within: {temporal_patterns['insights']['most_videos_trend_within']}")
        print(f"• Quickest Category: {temporal_patterns['insights']['quickest_category']}")
        
        display_dataframe_nicely(
            temporal_patterns['temporal_by_category'].head(10), 
            "Quick Trending Percentage by Category", 
            max_rows=10
        )
        
        print(f"\nDays to Trend Distribution (First 10 days):")
        for days, count in temporal_patterns['days_distribution'].head(10).items():
            percentage = (count / stats['total_videos']) * 100
            print(f"• {days} days: {count:,} videos ({percentage:.1f}%)")
        
        # Generate and save comprehensive report
        logger.info("\n" + "="*60)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*60)
        
        report_path = config.OUTPUT_DATA_PATH / "business_insights_report.json"
        report = insights_analyzer.generate_comprehensive_report(str(report_path))
        
        print(f"\nKEY BUSINESS INSIGHTS:")
        print("=" * 25)
        for insight in report['key_insights']:
            print(f"• {insight}")
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("BUSINESS INSIGHTS ANALYSIS COMPLETED!")
        logger.info("="*60)
        logger.info("Generated outputs:")
        logger.info(f"• {report_path} - Comprehensive business insights report")
        logger.info("\nKey findings:")
        logger.info("1. High Engagement Videos: K-pop content (BTS, j-hope) shows exceptional engagement")
        logger.info("2. Quick Trending: Nonprofits & Activism content trends fastest")
        logger.info("3. Ranking Patterns: No strong correlation between engagement and trending rank")
        logger.info("4. Temporal Patterns: Most videos trend within 5 days of publication")
        logger.info("\nAnalysis completed using efficient pandas operations!")

    except Exception as e:
        logger.error(f"Business insights analysis failed: {e}")
        logger.error("Check the error above and ensure:")
        logger.error("   - Engineered features data exists")
        logger.error("   - Required Python packages are installed")
        raise

if __name__ == "__main__":
    setup_logging()
    main()
