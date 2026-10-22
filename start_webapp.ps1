# BERT-LS Web App Launcher (PowerShell)
# Starts the Flask web server on http://localhost:5000

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  BERT-LS Lexical Simplification Web App" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Choose mode: remove --skip-fasttext for full mode (needs ~8GB RAM)
Write-Host "Starting in LITE mode (no FastText, ~2GB RAM)..."
Write-Host "For FULL mode, edit this script and remove --skip-fasttext"
Write-Host "Open http://localhost:5000 in your browser after loading."
Write-Host ""

py -3 app.py --skip-fasttext
