"""
YouTube Analytics Project

A PySpark-based data engineering project for analyzing YouTube trending videos data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Make imports easier
import sys
from pathlib import Path

# Add src to Python path
_project_root = Path(__file__).parent
_src_path = _project_root / "src"

if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
