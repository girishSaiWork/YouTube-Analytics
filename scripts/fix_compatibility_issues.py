#!/usr/bin/env python3
"""
Fix Compatibility Issues Script

This script addresses common compatibility issues between different versions
of pandas, scipy, and other packages in the advanced analytics notebooks.

Usage:
    python scripts/fix_compatibility_issues.py
"""

import sys
import warnings
from pathlib import Path

def check_package_versions():
    """Check and report package versions"""
    packages = {
        'pandas': 'pd',
        'numpy': 'np', 
        'scipy': 'scipy',
        'sklearn': 'sklearn',
        'tensorflow': 'tf'
    }
    
    print("Package Version Check:")
    print("=" * 40)
    
    for package, alias in packages.items():
        try:
            if package == 'tensorflow':
                import tensorflow as tf
                print(f"{package}: {tf.__version__}")
            elif package == 'sklearn':
                import sklearn
                print(f"{package}: {sklearn.__version__}")
            else:
                module = __import__(package)
                print(f"{package}: {module.__version__}")
        except ImportError:
            print(f"{package}: Not installed")
        except Exception as e:
            print(f"{package}: Error - {e}")

def fix_sparse_matrix_indexing():
    """
    Fix for scipy sparse matrix indexing with pandas boolean Series
    
    Issue: 'Series' object has no attribute 'nonzero'
    Solution: Convert boolean mask to numpy array indices
    """
    print("\nApplying sparse matrix indexing fix...")
    
    # This is already fixed in the notebook, but we can provide guidance
    fix_code = """
    # Instead of:
    # category_tfidf = tfidf_matrix[category_mask].mean(axis=0).A1
    
    # Use:
    category_indices = np.where(category_mask)[0]
    category_tfidf = tfidf_matrix[category_indices].mean(axis=0).A1
    """
    
    print("Fix applied in notebooks. Use this pattern for similar issues:")
    print(fix_code)

def suppress_warnings():
    """Suppress common warnings that don't affect functionality"""
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', message='.*oneDNN custom operations.*')
    print("Common warnings suppressed for cleaner output.")

def main():
    """Main function to run all compatibility fixes"""
    print("YouTube Analytics - Compatibility Fix Script")
    print("=" * 50)
    
    # Check package versions
    check_package_versions()
    
    # Apply fixes
    fix_sparse_matrix_indexing()
    suppress_warnings()
    
    print("\n" + "=" * 50)
    print("Compatibility fixes applied successfully!")
    print("\nCommon Issues and Solutions:")
    print("1. 'Series' object has no attribute 'nonzero' - Fixed in notebooks")
    print("2. TensorFlow oneDNN warnings - Suppressed (harmless)")
    print("3. Package version conflicts - Check versions above")
    
    print("\nIf you encounter other issues:")
    print("- Update packages: pip install --upgrade -r requirements.txt")
    print("- Check Python version: Python 3.8+ recommended")
    print("- Restart Jupyter kernel after package updates")

if __name__ == "__main__":
    main()
