# YouTube Analytics Project

A PySpark-based data engineering project for analyzing YouTube trending videos data.

## Project Structure

```
YouTube Analytics/
├── config/          # Configuration files
├── data/           # Data storage (raw, processed, output)
├── src/            # Source code
├── notebooks/      # Jupyter notebooks for exploration
├── tests/          # Unit tests
├── scripts/        # Execution scripts
└── docs/           # Documentation
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Configure your environment: Copy `.env.example` to `.env`
3. **Process raw data**: `python scripts/run_pipeline.py`
4. **Analyze processed data**: `python scripts/analyze_processed_data.py`

## Workflow

### Step 1: Data Processing
```bash
python scripts/run_pipeline.py
```
- Reads raw CSV and JSON files from `data/raw/`
- Applies transformations and cleaning
- Saves processed data to `data/processed/` in both parquet and CSV formats
- **Outputs:**
  - `youtube_trending_videos.parquet` - Main dataset in parquet format
  - `youtube_trending_videos.csv` - Main dataset in CSV format
  - `category_mappings.parquet` - Category mappings in parquet format
  - `category_mappings.csv` - Category mappings in CSV format

### Step 2: Feature Engineering (NEW!)
```bash
python scripts/feature_engineering_demo.py
# OR
run_feature_engineering.bat
```
- Creates advanced features using PySpark functions
- **engagement_score**: Weighted metric combining likes, dislikes, and comments relative to views
- **days_to_trend**: Number of days between publish_time and trending_date
- **trending_rank**: Rank videos within each trending_date and category based on views
- **Outputs:**
  - `youtube_trending_videos_with_features.parquet` - Enhanced dataset with engineered features
  - `youtube_trending_videos_with_features.csv` - Enhanced dataset in CSV format
  - `business_insights_report.json` - Comprehensive business insights report

### Step 2b: Business Insights Analysis (NEW!)

#### **Script-based Analysis:**
```bash
python scripts/business_insights_analysis.py
# OR
run_business_insights.bat
```

#### **Interactive Notebook Analysis:**
```bash
jupyter notebook notebooks/04_business_insights_analysis.ipynb
# OR
launch_business_insights_notebook.bat
```

- **Pandas-based efficient analysis** of engineered features
- **Interactive visualizations** and statistical analysis
- Comprehensive business insights generation
- **Key Findings:**
  - **High Engagement Videos**: K-pop content (BTS, j-hope) shows exceptional engagement
  - **Quick Trending**: Nonprofits & Activism content trends fastest
  - **Ranking Patterns**: No strong correlation between engagement and trending rank
  - **Temporal Patterns**: Most videos trend within 5 days of publication

### Step 3: Data Analysis
```bash
python scripts/analyze_processed_data.py
```
- Loads processed parquet data
- Performs comprehensive analytics
- Generates insights and reports



## Data Sources

This project analyzes YouTube trending videos data from multiple countries:
- Video data: CSV files with trending video information
- Category data: JSON files with category ID mappings

## Jupyter Notebooks

Explore the data interactively using our comprehensive notebooks:

### `notebooks/01_data_exploration.ipynb`
- Initial data exploration and analysis
- Data quality assessment
- Basic statistics and visualizations

### `notebooks/02_production_pipeline_demo.ipynb`
- Production pipeline demonstration
- End-to-end workflow examples
- Analytics and insights generation

### `notebooks/03_feature_engineering.ipynb` (NEW!)
- Advanced feature engineering with PySpark
- Complex transformations and metrics
- Window functions and advanced analytics
- Interactive analysis of engineered features

### `notebooks/04_business_insights_analysis.ipynb` (NEW!)
- **Pandas-based business insights analysis**
- Interactive visualizations and statistical analysis
- Comprehensive business intelligence reporting
- Actionable recommendations and insights

## Features

- Multi-country data processing
- Advanced feature engineering with PySpark
- **Pandas-based business insights analysis**
- Trending pattern analysis
- Category-based insights
- Engagement metrics calculation
- Window functions for ranking and analysis
- Comprehensive business intelligence reporting
- Dual format outputs (Parquet + CSV)

## Project Structure

```
YouTube Analytics/
├── data/
│   ├── raw/                    # Raw CSV and JSON files
│   ├── processed/              # Processed parquet and CSV files
│   └── output/                 # Feature engineered data and analysis results
├── src/
│   ├── data_ingestion/         # Data loading modules
│   ├── data_processing/        # ETL pipeline and feature engineering
│   ├── analytics/              # Business insights analysis (NEW!)
│   └── utils/                  # Utility functions
├── scripts/                    # Execution scripts
├── notebooks/                  # Jupyter notebooks (including feature engineering)
├── tests/                      # Unit tests
└── config/                     # Configuration files
```

## Generated Output Files

### **Processed Data** (`data/processed/`)
- `youtube_trending_videos.parquet` - Main dataset in parquet format
- `youtube_trending_videos.csv` - Main dataset in CSV format
- `category_mappings.parquet` - Category mappings in parquet format
- `category_mappings.csv` - Category mappings in CSV format

### **Feature Engineered Data** (`data/output/`)
- `youtube_trending_videos_with_features.parquet` - Enhanced dataset with engineered features (32.7 MB)
- `youtube_trending_videos_with_features.csv` - Enhanced dataset in CSV format (64.4 MB)
- `business_insights_report.json` - Comprehensive business insights report (30.5 KB)

### **Engineered Features**
1. **engagement_score**: Weighted metric `((likes * 0.5) + (dislikes * 0.2) + (comment_count * 0.3)) / views`
2. **days_to_trend**: Number of days between publish_time and trending_date
3. **trending_rank**: Rank of videos within each trending_date and category based on views

## Key Business Insights (From Pandas Analysis)

### 📊 **High Engagement Videos**
- **Music content dominates** with 85% of top engagement videos
- **K-pop content (BTS, j-hope)** shows exceptional engagement rates (12-16%)
- **Top channels**: Shawn Mendes, ibighit, Bruno Mars

### ⚡ **Quick Trending Categories**
- **News & Politics**: 5.1 days average (fastest)
- **Nonprofits & Activism**: 5.4 days average
- **Sports**: 5.8 days average
- **Comedy**: 6.3 days average

### 🏆 **Ranking Patterns**
- **Weak correlation** (0.015) between engagement and trending rank
- **Comedy** has the most top-ranked videos
- **Higher ranks don't guarantee higher engagement**

### ⏰ **Temporal Patterns**
- **51.7%** of videos trend within 5 days of publication
- **69.2%** trend within 1 week
- **Peak trending**: Days 3-5 after publication
- **Same-day trending**: Only 0.3% of videos
