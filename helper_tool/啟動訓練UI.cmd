@echo off
setlocal
set "PYTHONUTF8=1"
set "TRAINING_PYTHON=%~dp0reports\training-runtime\Scripts\python.exe"
if not exist "%TRAINING_PYTHON%" set "TRAINING_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%TRAINING_PYTHON%" set "TRAINING_PYTHON=python"
pushd "%~dp0.."
"%TRAINING_PYTHON%" helper_tool\ui_server.py
popd
pause
