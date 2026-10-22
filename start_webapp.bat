@echo off
REM BERT-LS Web App Launcher
REM Starts the Flask web server on http://localhost:5000

echo ============================================================
echo   BERT-LS Lexical Simplification Web App
echo ============================================================
echo.
echo Starting in LITE mode (no FastText, ~2GB RAM)...
echo For FULL mode, edit this script and remove --skip-fasttext
echo Open http://localhost:5000 in your browser after loading.
echo.

py -3 app.py --skip-fasttext

pause
