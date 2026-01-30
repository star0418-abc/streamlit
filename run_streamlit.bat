@echo off
REM GPE Lab Streamlit Launcher for Windows + Conda
REM ================================================
REM This script launches the GPE Lab Streamlit application.
REM It attempts to use conda to activate the environment and run streamlit.
REM
REM Usage:
REM   1. Edit CONDA_ENV below to match your environment name
REM   2. Double-click run_streamlit.bat to launch

setlocal

REM ============ CONFIGURATION ============
REM Set your conda environment name here:
set CONDA_ENV=base

REM =======================================

cd /d "%~dp0"

echo.
echo ========================================
echo   GPE Lab Streamlit Launcher
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda not found in PATH.
    echo.
    echo Please ensure Anaconda/Miniconda is installed and added to PATH.
    echo Or run manually:
    echo   1. Open Anaconda Prompt
    echo   2. conda activate %CONDA_ENV%
    echo   3. pip install -r requirements.txt
    echo   4. streamlit run app.py
    echo.
    pause
    exit /b 1
)

echo [INFO] Using conda environment: %CONDA_ENV%
echo.

REM Try to activate conda and run streamlit
call conda activate %CONDA_ENV% 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate conda environment: %CONDA_ENV%
    echo.
    echo Please check that the environment exists:
    echo   conda env list
    echo.
    echo Or create it:
    echo   conda create -n gpe_lab python=3.10
    echo   conda activate gpe_lab
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] streamlit not installed in environment.
    echo.
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo [INFO] Starting Streamlit...
echo [INFO] App will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Start the app and open browser
start http://localhost:8501
python -m streamlit run app.py

pause
