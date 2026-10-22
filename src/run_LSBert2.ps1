# BERT-LS Windows Runner (PowerShell)
# Usage: .\run_LSBert2.ps1
# Uses Python 3.12 for optimal PyTorch performance

$BERT_DIR = $PSScriptRoot
$Result_DIR = Join-Path $BERT_DIR "results"

# Create results directory if it doesn't exist
if (-not (Test-Path $Result_DIR)) {
    New-Item -ItemType Directory -Path $Result_DIR | Out-Null
}

# Dataset options: lex.mturk.txt, NNSeval.txt, BenchLS.txt
$dataset = "lex.mturk.txt"

py -3.12 LSBert2.py `
  --do_eval `
  --do_lower_case `
  --no_cuda `
  --low_mem `
  --num_selections 10 `
  --prob_mask 0.5 `
  --eval_dir "$BERT_DIR\datasets\$dataset" `
  --bert_model bert-base-uncased `
  --max_seq_length 128 `
  --word_embeddings "$BERT_DIR\datasets\Fasttext\crawl-300d-2M-subword.vec" `
  --word_frequency "$BERT_DIR\SUBTLEX_frequency.xlsx" `
  --ppdb "$BERT_DIR\datasets\ppdb-2.0-tldr" `
  --output_SR_file "$Result_DIR\features.txt"
