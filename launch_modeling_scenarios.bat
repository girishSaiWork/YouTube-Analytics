@echo off
echo Launching Machine Learning Modeling Scenarios Notebook...
echo.
echo This notebook provides comprehensive ML modeling including:
echo - Viral video prediction (Classification)
echo - View count forecasting (Regression)
echo - Time series forecasting (ARIMA, Prophet)
echo - Content-based recommendation systems
echo - Deep learning with neural networks
echo - Business intelligence and ROI analysis
echo.
echo Make sure you have run the feature engineering script first!
echo.
echo Required libraries: scikit-learn, xgboost, lightgbm
echo Optional: tensorflow (for deep learning), prophet (for time series)
echo.
pause
jupyter notebook notebooks/06_modeling_scenarios.ipynb
