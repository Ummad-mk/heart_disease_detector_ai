@echo off
:: =============================================================
:: run_assignment.bat  –  ML Assignment 2 (Windows helper)
:: =============================================================
:: Equivalent of the Makefile for native Windows (no WSL needed).
:: =============================================================
setlocal

set ROOT=%~dp0

echo.
echo  ====================================================
echo   ML Assignment 2  –  Runner
echo  ====================================================
echo.
echo   1. Train BOTH models
echo   2. Train Titanic only
echo   3. Train Heart Disease only
echo   4. Launch Titanic Streamlit app
echo   5. Launch Heart Disease Streamlit app
echo   6. Run tests
echo   7. Execute Notebooks
echo   8. Generate PDF Reports from Notebooks
echo.
set /p choice="  Choose an option [1-8]: "

if "%choice%"=="1" goto train_all
if "%choice%"=="2" goto train_titanic
if "%choice%"=="3" goto train_heart
if "%choice%"=="4" goto app_titanic
if "%choice%"=="5" goto app_heart
if "%choice%"=="6" goto run_tests
if "%choice%"=="7" goto notebooks
if "%choice%"=="8" goto pdf_reports
echo Invalid option. Exiting.
goto end

:train_all
echo.
echo [1/2] Training Titanic model ...
python scripts\train_titanic.py || goto error
echo [2/2] Training Heart Disease model ...
python scripts\train_heart_disease.py || goto error
echo.
echo  Both models trained successfully.
goto end

:train_titanic
echo.
echo Training Titanic model ...
python scripts\train_titanic.py || goto error
goto end

:train_heart
echo.
echo Training Heart Disease model ...
python scripts\train_heart_disease.py || goto error
goto end

:app_titanic
echo.
echo Launching Titanic app at http://localhost:8501 ...
streamlit run apps\titanic_app.py
goto end

:app_heart
echo.
echo Launching Heart Disease app at http://localhost:8501 ...
streamlit run apps\heart_disease_app.py
goto end

:run_tests
echo.
echo Running tests ...
pytest tests\ -v || goto error
goto end

:notebooks
echo.
echo Executing notebooks ...
jupyter nbconvert --to notebook --execute --inplace notebooks\01_titanic_survival_prediction.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks\02_heart_disease_prediction.ipynb
echo Notebooks executed.
goto end

:pdf_reports
echo.
echo Generating PDF reports from notebooks ...
python scripts\generate_pdf_reports.py || goto error
echo.
echo PDF reports generated successfully in outputs\reports\
goto end

:error
echo.
echo  ERROR: Command failed (exit code %errorlevel%).
pause
exit /b 1

:end
echo.
pause
