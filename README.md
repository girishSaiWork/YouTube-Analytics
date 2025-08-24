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
- Includes comprehensive analysis and correlation studies

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

## Features

- Multi-country data processing
- Advanced feature engineering with PySpark
- Trending pattern analysis
- Category-based insights
- Engagement metrics calculation
- Window functions for ranking and analysis
