@echo off
REM Benchmark: SINGLE GPU (local GPU only)
REM Kills all running llama-server.exe processes before and after the run.

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

"%PY%" "%~dp0bench.py" --label single --out "%~dp0results-single.csv" --wait

echo Stopping server...
taskkill /im llama-server.exe /f >nul 2>&1
pause
