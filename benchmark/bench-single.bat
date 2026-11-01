@echo off
REM Benchmark: SINGLE GPU (local GPU only)
REM Kills all running llama-server.exe processes before and after the run.
REM
REM Edit the variables at the top before running.

set LLAMA_DIR=C:\path\to\llama.cpp
set MODEL=C:\path\to\model.gguf
set PY=python

taskkill /im llama-server.exe /f >nul 2>&1
timeout /t 3 >nul

echo Starting server (single GPU)...
start "bench-server" /min "%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL%" ^
  --fit on --fit-ctx 128000 --fit-target 256 ^
  -np 1 -fa on -t 8 ^
  -b 2048 -ub 2048 ^
  -ctk q8_0 -ctv q8_0 ^
  --jinja ^
  --host 127.0.0.1 --port 8033

REM Context sizes: 512, 1K, 2K, 4K, 8K, 16K, 24K, 32K, 48K, 64K, 75K
REM 5 runs per size for steadier numbers, 256 tokens generated each time.
"%PY%" "%~dp0bench.py" --label single --out "%~dp0results-single.csv" ^
  --sizes 512 1000 2000 4000 8000 16000 24000 32000 48000 64000 75000 ^
  --runs 5 --gen 256 --wait

echo Stopping server...
taskkill /im llama-server.exe /f >nul 2>&1
pause
