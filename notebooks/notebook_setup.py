"""
Notebook setup helper for YouTube Analytics project
Run this at the beginning of any notebook to set up imports correctly
"""

import sys
import os
from pathlib import Path

def setup_notebook_environment():
    """Setup the notebook environment for imports"""
    
    # Get the project root (parent of notebooks directory)
    notebook_dir = Path.cwd()
    if notebook_dir.name == "notebooks":
        project_root = notebook_dir.parent
    else:
        # If running from project root
        project_root = notebook_dir
        if not (project_root / "notebooks").exists():
            raise ValueError("Cannot find project root. Please run from notebooks directory or project root.")
    
    # Add paths to sys.path
    paths_to_add = [
        str(project_root),
        str(project_root / "src"),
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set environment variables
    os.environ["PYTHONPATH"] = ":".join(paths_to_add)
    
    print(f"✅ Project root: {project_root}")
    print(f"✅ Added to Python path:")
    for path in paths_to_add:
        print(f"   - {path}")
    
    return project_root

def test_imports():
    """Test if all imports work correctly"""
    try:
        from config.settings import Config
        print("✅ Config import successful")
        
        from src.utils.spark_utils import SparkUtils
        print("✅ SparkUtils import successful")
        
        from src.data_ingestion.readers import YouTubeDataReader
        print("✅ YouTubeDataReader import successful")
        
        print("✅ All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    # When run as script
    project_root = setup_notebook_environment()
    test_imports()
