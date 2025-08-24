@echo off
echo YouTube Analytics - Compatibility Fix
echo =====================================
echo.
echo This script will check package versions and apply compatibility fixes
echo for common issues in the advanced analytics notebooks.
echo.
echo Common issues addressed:
echo - Sparse matrix indexing with pandas Series
echo - TensorFlow warning messages
echo - Package version conflicts
echo.
pause
echo.
echo Running compatibility fixes...
python scripts/fix_compatibility_issues.py
echo.
echo Compatibility check complete!
pause
