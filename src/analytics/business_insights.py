"""
Business Insights Analysis using Pandas
Efficient data analysis for YouTube trending videos with engineered features
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class YouTubeBusinessInsights:
    """Business insights analyzer for YouTube trending videos using pandas"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the business insights analyzer
        
        Args:
            data_path: Path to the engineered features data file
        """
        self.df = None
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load engineered features data from file
        
        Args:
            data_path: Path to the data file (parquet or CSV)
            
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {data_path}")
        
        try:
            if data_path.endswith('.parquet'):
                self.df = pd.read_parquet(data_path)
            elif data_path.endswith('.csv'):
                self.df = pd.read_csv(data_path)
            else:
                # Try parquet first, then CSV
                try:
                    self.df = pd.read_parquet(data_path + '.parquet')
                except:
                    self.df = pd.read_csv(data_path + '.csv')
            
            logger.info(f"Successfully loaded {len(self.df)} records")
            return self.df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def analyze_high_engagement_content(self, top_n: int = 20) -> Dict:
        """
        Analyze high engagement videos to identify content patterns
        
        Args:
            top_n: Number of top videos to analyze
            
        Returns:
            Dictionary with high engagement analysis results
        """
        logger.info("Analyzing high engagement content...")
        
        # Get top engagement videos
        top_engagement = self.df.nlargest(top_n, 'engagement_score')
        
        # Analyze by channel
        channel_analysis = top_engagement.groupby('channel_title').agg({
            'engagement_score': ['count', 'mean', 'max'],
            'views': 'mean'
        }).round(6)
        # Flatten column names
        channel_analysis.columns = ['_'.join(col).strip() for col in channel_analysis.columns.values]

        # Analyze by category
        category_analysis = top_engagement.groupby('category_name').agg({
            'engagement_score': ['count', 'mean', 'max'],
            'views': 'mean'
        }).round(6)
        # Flatten column names
        category_analysis.columns = ['_'.join(col).strip() for col in category_analysis.columns.values]
        
        # Content pattern analysis
        content_patterns = self._analyze_content_patterns(top_engagement)
        
        results = {
            'top_videos': top_engagement[['title', 'channel_title', 'category_name', 
                                       'engagement_score', 'views', 'likes', 'dislikes', 'comment_count']],
            'channel_analysis': channel_analysis,
            'category_analysis': category_analysis,
            'content_patterns': content_patterns,
            'summary': {
                'avg_engagement': top_engagement['engagement_score'].mean(),
                'avg_views': top_engagement['views'].mean(),
                'most_common_category': top_engagement['category_name'].mode().iloc[0],
                'most_frequent_channel': top_engagement['channel_title'].mode().iloc[0]
            }
        }
        
        logger.info("High engagement analysis completed")
        return results
    
    def analyze_trending_speed_by_category(self) -> Dict:
        """
        Analyze how quickly content trends by category
        
        Returns:
            Dictionary with trending speed analysis
        """
        logger.info("Analyzing trending speed by category...")
        
        # Filter out extreme outliers (videos that took years to trend)
        df_filtered = self.df[self.df['days_to_trend'] <= 365]  # Within a year
        
        category_speed = df_filtered.groupby('category_name').agg({
            'days_to_trend': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'engagement_score': 'mean',
            'views': 'mean'
        }).round(2)
        # Flatten column names
        category_speed.columns = ['_'.join(col).strip() for col in category_speed.columns.values]

        # Sort by average days to trend
        category_speed = category_speed.sort_values('days_to_trend_mean')
        
        # Quick trending categories (< 7 days average)
        quick_trending = category_speed[category_speed['days_to_trend_mean'] < 7]

        # Slow trending categories (> 14 days average)
        slow_trending = category_speed[category_speed['days_to_trend_mean'] > 14]
        
        results = {
            'category_speed_analysis': category_speed,
            'quick_trending_categories': quick_trending,
            'slow_trending_categories': slow_trending,
            'overall_stats': {
                'avg_days_to_trend': df_filtered['days_to_trend'].mean(),
                'median_days_to_trend': df_filtered['days_to_trend'].median(),
                'fastest_category': category_speed.index[0],
                'slowest_category': category_speed.index[-1]
            }
        }
        
        logger.info("Trending speed analysis completed")
        return results
    
    def analyze_ranking_patterns(self) -> Dict:
        """
        Analyze correlation between engagement and trending rank
        
        Returns:
            Dictionary with ranking pattern analysis
        """
        logger.info("Analyzing ranking patterns...")
        
        # Engagement vs Rank correlation
        engagement_rank_corr = self.df['engagement_score'].corr(self.df['trending_rank'])
        
        # Average engagement by rank position
        rank_engagement = self.df.groupby('trending_rank').agg({
            'engagement_score': ['count', 'mean', 'std'],
            'views': 'mean',
            'days_to_trend': 'mean'
        }).round(6)
        # Flatten column names
        rank_engagement.columns = ['_'.join(col).strip() for col in rank_engagement.columns.values]
        
        # Top ranked videos analysis
        top_ranked = self.df[self.df['trending_rank'] <= 3]
        top_ranked_analysis = top_ranked.groupby('category_name').agg({
            'trending_rank': 'count',
            'engagement_score': 'mean',
            'views': 'mean'
        }).sort_values('trending_rank', ascending=False)
        
        # Rank distribution
        rank_distribution = self.df['trending_rank'].value_counts().sort_index()
        
        results = {
            'engagement_rank_correlation': engagement_rank_corr,
            'rank_engagement_analysis': rank_engagement,
            'top_ranked_by_category': top_ranked_analysis,
            'rank_distribution': rank_distribution,
            'insights': {
                'correlation_strength': 'weak' if abs(engagement_rank_corr) < 0.3 else 'moderate' if abs(engagement_rank_corr) < 0.7 else 'strong',
                'correlation_direction': 'negative' if engagement_rank_corr < 0 else 'positive',
                'most_top_ranked_category': top_ranked_analysis.index[0] if len(top_ranked_analysis) > 0 else 'N/A'
            }
        }
        
        logger.info("Ranking pattern analysis completed")
        return results
    
    def analyze_temporal_patterns(self) -> Dict:
        """
        Analyze temporal patterns in trending behavior
        
        Returns:
            Dictionary with temporal pattern analysis
        """
        logger.info("Analyzing temporal patterns...")
        
        # Days to trend distribution
        days_distribution = self.df['days_to_trend'].value_counts().sort_index()
        
        # Same day trending
        same_day_trending = len(self.df[self.df['days_to_trend'] == 0])
        
        # Within 5 days
        within_5_days = len(self.df[self.df['days_to_trend'] <= 5])
        
        # Within a week
        within_week = len(self.df[self.df['days_to_trend'] <= 7])
        
        # Temporal patterns by category
        temporal_by_category = self.df.groupby('category_name').agg({
            'days_to_trend': lambda x: (x <= 5).sum(),  # Count of videos trending within 5 days
        }).rename(columns={'days_to_trend': 'quick_trending_count'})
        
        temporal_by_category['total_videos'] = self.df.groupby('category_name').size()
        temporal_by_category['quick_trending_percentage'] = (
            temporal_by_category['quick_trending_count'] / temporal_by_category['total_videos'] * 100
        ).round(2)
        
        results = {
            'days_distribution': days_distribution.head(20),  # First 20 days
            'temporal_stats': {
                'same_day_trending': same_day_trending,
                'within_5_days': within_5_days,
                'within_week': within_week,
                'total_videos': len(self.df),
                'same_day_percentage': (same_day_trending / len(self.df) * 100),
                'within_5_days_percentage': (within_5_days / len(self.df) * 100),
                'within_week_percentage': (within_week / len(self.df) * 100)
            },
            'temporal_by_category': temporal_by_category.sort_values('quick_trending_percentage', ascending=False),
            'insights': {
                'most_videos_trend_within': '5 days' if within_5_days > len(self.df) * 0.5 else '7 days' if within_week > len(self.df) * 0.5 else 'more than a week',
                'quickest_category': temporal_by_category.sort_values('quick_trending_percentage', ascending=False).index[0]
            }
        }
        
        logger.info("Temporal pattern analysis completed")
        return results
    
    def _analyze_content_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze content patterns in titles and channels
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with content pattern insights
        """
        # Common words in high engagement titles
        all_titles = ' '.join(df['title'].astype(str)).lower()
        
        # Simple word frequency (you could use more sophisticated NLP here)
        words = all_titles.split()
        word_freq = pd.Series(words).value_counts().head(10)
        
        # Channel patterns
        channel_freq = df['channel_title'].value_counts().head(10)
        
        return {
            'common_words_in_titles': word_freq.to_dict(),
            'top_channels': channel_freq.to_dict()
        }
    
    def generate_comprehensive_report(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate a comprehensive business insights report
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Complete analysis results
        """
        logger.info("Generating comprehensive business insights report...")
        
        if self.df is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Run all analyses
        high_engagement = self.analyze_high_engagement_content()
        trending_speed = self.analyze_trending_speed_by_category()
        ranking_patterns = self.analyze_ranking_patterns()
        temporal_patterns = self.analyze_temporal_patterns()
        
        # Compile comprehensive report
        report = {
            'dataset_overview': {
                'total_videos': len(self.df),
                'date_range': f"{self.df['trending_date'].min()} to {self.df['trending_date'].max()}",
                'categories': self.df['category_name'].nunique(),
                'channels': self.df['channel_title'].nunique(),
                'countries': self.df['country'].nunique() if 'country' in self.df.columns else 'N/A'
            },
            'high_engagement_analysis': high_engagement,
            'trending_speed_analysis': trending_speed,
            'ranking_pattern_analysis': ranking_patterns,
            'temporal_pattern_analysis': temporal_patterns,
            'key_insights': self._generate_key_insights(high_engagement, trending_speed, ranking_patterns, temporal_patterns)
        }
        
        # Save report if path provided
        if save_path:
            self._save_report(report, save_path)
        
        logger.info("Comprehensive report generated successfully")
        return report
    
    def _generate_key_insights(self, high_engagement: Dict, trending_speed: Dict, 
                             ranking_patterns: Dict, temporal_patterns: Dict) -> List[str]:
        """Generate key business insights from all analyses"""
        insights = []
        
        # High engagement insights
        top_category = high_engagement['summary']['most_common_category']
        top_channel = high_engagement['summary']['most_frequent_channel']
        insights.append(f"High Engagement Videos: {top_category} content, especially from {top_channel}, shows exceptional engagement")
        
        # Trending speed insights
        fastest_category = trending_speed['overall_stats']['fastest_category']
        insights.append(f"Quick Trending: {fastest_category} content trends fastest")
        
        # Ranking insights
        correlation_strength = ranking_patterns['insights']['correlation_strength']
        insights.append(f"Ranking Patterns: {correlation_strength.title()} correlation between engagement and trending rank")
        
        # Temporal insights
        quick_trend_pct = temporal_patterns['temporal_stats']['within_5_days_percentage']
        insights.append(f"Temporal Patterns: {quick_trend_pct:.1f}% of videos trend within 5 days of publication")
        
        return insights
    
    def _save_report(self, report: Dict, save_path: str) -> None:
        """Save the report to file"""
        import json
        
        # Convert pandas objects to serializable format
        def convert_for_json(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
        
        # Deep convert the report
        json_report = json.loads(json.dumps(report, default=convert_for_json))
        
        with open(save_path, 'w') as f:
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Report saved to {save_path}")
