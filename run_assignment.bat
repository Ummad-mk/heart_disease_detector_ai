@echo off
echo ============================================================
echo  ML Assignment 2 - Run All Scripts
echo ============================================================
echo.

echo [Step 1] Running Task 1 - Titanic Survival Prediction...
python Task1_Titanic_Survival_Prediction.py
if errorlevel 1 (
    echo ERROR: Task 1 failed!
    pause
    exit /b 1
)
echo Task 1 complete!
echo.

echo [Step 2] Running Task 2 - Heart Disease Prediction...
python Task2_Heart_Disease_Prediction.py
if errorlevel 1 (
    echo ERROR: Task 2 failed!
    pause
    exit /b 1
)
echo Task 2 complete!
echo.

echo [Step 3] Executing Notebooks (with outputs)...
jupyter nbconvert --to notebook --execute --inplace Task1_Titanic_Survival_Prediction.ipynb
jupyter nbconvert --to notebook --execute --inplace Task2_Heart_Disease_Prediction.ipynb
echo Notebooks executed!
echo.

echo [Step 4] Launching Heart Disease Streamlit App...
echo Open browser at: http://localhost:8501
streamlit run heart_disease_app.py

pause
