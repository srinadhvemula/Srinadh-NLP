@echo off
REM BERT-LS Windows Runner (Batch File)
REM Usage: run_LSBert2.bat
REM Uses Python 3.12 for optimal PyTorch performance

set BERT_DIR=%~dp0
set Result_DIR=%BERT_DIR%results

REM Create results directory if it doesn't exist
if not exist "%Result_DIR%" mkdir "%Result_DIR%"

REM Dataset options: lex.mturk.txt, NNSeval.txt, BenchLS.txt
set DATASET=lex.mturk.txt

py -3.12 LSBert2.py ^
  --do_eval ^
  --do_lower_case ^
  --no_cuda ^
  --low_mem ^
  --num_selections 10 ^
  --prob_mask 0.5 ^
  --eval_dir "%BERT_DIR%datasets\%DATASET%" ^
  --bert_model bert-base-uncased ^
  --max_seq_length 128 ^
  --word_embeddings "%BERT_DIR%datasets\Fasttext\crawl-300d-2M-subword.vec" ^
  --word_frequency "%BERT_DIR%SUBTLEX_frequency.xlsx" ^
  --ppdb "%BERT_DIR%datasets\ppdb-2.0-tldr" ^
  --output_SR_file "%Result_DIR%\features.txt"

pause
