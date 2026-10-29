@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto missing_setup
if not exist ".windows_models_ready" goto missing_setup
set USE_TF=0
set USE_FLAX=0
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TOKENIZERS_PARALLELISM=false
set HF_HUB_DISABLE_TELEMETRY=1
echo Open http://127.0.0.1:8501 in your browser.
echo Keep this window open. Press Ctrl+C to stop.
".venv\Scripts\python.exe" scripts\local_runtime.py --streamlit --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false
pause
exit /b
:missing_setup
echo Run setup_windows.bat successfully before starting the app.
pause
exit /b 1
