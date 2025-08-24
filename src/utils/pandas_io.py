"""
Pandas-based I/O utilities for Windows compatibility
Converts between PySpark DataFrames and Pandas for file operations
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

logger = logging.getLogger(__name__)

class PandasIOHandler:
    """Handle file I/O using Pandas to bypass Windows Hadoop issues"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def save_spark_df_as_parquet(self, df: SparkDataFrame, output_path: str) -> bool:
        """
        Save Spark DataFrame as parquet using Pandas

        Args:
            df: Spark DataFrame to save
            output_path: Path to save the parquet file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Converting Spark DataFrame to Pandas for saving to {output_path}")

            # Convert Spark DataFrame to Pandas
            pandas_df = df.toPandas()

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Save as parquet using Pandas
            if not str(output_path).endswith('.parquet'):
                output_path = f"{output_path}.parquet"

            try:
                pandas_df.to_parquet(output_path, index=False, engine='pyarrow')
                logger.info(f"Successfully saved {len(pandas_df)} records to {output_path}")
                return True
            except ImportError:
                logger.warning("PyArrow not available, falling back to CSV format")
                # Fallback to CSV if pyarrow is not available
                csv_path = output_path.replace('.parquet', '.csv')
                return self.save_spark_df_as_csv(df, csv_path)

        except Exception as e:
            logger.error(f"Failed to save DataFrame as parquet: {e}")
            return False
    
    def save_spark_df_as_csv(self, df: SparkDataFrame, output_path: str) -> bool:
        """
        Save Spark DataFrame as CSV using Pandas
        
        Args:
            df: Spark DataFrame to save
            output_path: Path to save the CSV file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Converting Spark DataFrame to Pandas for saving to {output_path}")
            
            # Convert Spark DataFrame to Pandas
            pandas_df = df.toPandas()
            
            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV using Pandas
            if not str(output_path).endswith('.csv'):
                output_path = f"{output_path}.csv"
            
            pandas_df.to_csv(output_path, index=False)
            
            logger.info(f"Successfully saved {len(pandas_df)} records to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save DataFrame as CSV: {e}")
            return False
    
    def load_parquet_as_spark_df(self, input_path: str) -> Optional[SparkDataFrame]:
        """
        Load parquet file as Spark DataFrame using Pandas
        
        Args:
            input_path: Path to the parquet file
            
        Returns:
            SparkDataFrame or None if failed
        """
        try:
            logger.info(f"Loading parquet file from {input_path} using Pandas")
            
            # Load using Pandas
            pandas_df = pd.read_parquet(input_path)
            
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df)
            
            logger.info(f"Successfully loaded {len(pandas_df)} records from {input_path}")
            return spark_df
            
        except Exception as e:
            logger.error(f"Failed to load parquet file: {e}")
            return None
    
    def load_csv_as_spark_df(self, input_path: str) -> Optional[SparkDataFrame]:
        """
        Load CSV file as Spark DataFrame using Pandas
        
        Args:
            input_path: Path to the CSV file
            
        Returns:
            SparkDataFrame or None if failed
        """
        try:
            logger.info(f"Loading CSV file from {input_path} using Pandas")
            
            # Load using Pandas
            pandas_df = pd.read_csv(input_path)
            
            # Convert to Spark DataFrame
            spark_df = self.spark.createDataFrame(pandas_df)
            
            logger.info(f"Successfully loaded {len(pandas_df)} records from {input_path}")
            return spark_df
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            return None
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a saved file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with file information
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"exists": False}
            
            # Try to load and get basic info
            if path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path)
            elif path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                return {"exists": True, "error": "Unsupported file format"}
            
            return {
                "exists": True,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "dtypes": df.dtypes.to_dict()
            }
            
        except Exception as e:
            return {"exists": True, "error": str(e)}

def detect_environment() -> str:
    """
    Detect the current environment (Windows, Linux, Docker)
    
    Returns:
        str: Environment type
    """
    import platform
    import os
    
    system = platform.system()
    
    # Check if running in Docker
    if os.path.exists('/.dockerenv'):
        return "docker"
    
    # Check if running in WSL
    if system == "Linux" and "microsoft" in platform.uname().release.lower():
        return "wsl"
    
    return system.lower()

def should_use_pandas_io() -> bool:
    """
    Determine if Pandas I/O should be used based on environment
    
    Returns:
        bool: True if Pandas I/O should be used
    """
    env = detect_environment()
    
    # Use Pandas I/O for Windows to avoid Hadoop native library issues
    return env == "windows"
