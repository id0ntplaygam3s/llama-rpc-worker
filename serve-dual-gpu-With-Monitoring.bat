@echo off
REM Dual-GPU distributed inference: local GPU + remote RPC worker
REM Launches: hardware sidecar (port 8083), llama-server (port 8033), dashboard in browser.
REM
REM Remote RPC worker must be running on port 50052 before launching.
REM
REM Edit the variables below (Python path, model paths, RPC address).

set PY=C:\path\to\python.exe
set LLAMA_DIR=C:\path\to\llama.cpp
set MODEL=C:\path\to\model.gguf
set MMPROJ=C:\path\to\mmproj.gguf
set RPC_HOST=<rpc_worker_ip>:50052

REM 1. Hardware sidecar (called with python.exe directly, bypassing the py launcher / shebang)
start "hw-metrics" /min "%PY%" "%LLAMA_DIR%\hw-metrics.py" --host 127.0.0.1 --port 8083

REM 2. Dashboard in default browser once servers have had a moment to start
start "dashboard-http" /min "%PY%" -m http.server 3000 --bind 127.0.0.1 --directory "%LLAMA_DIR%"
start "" /b powershell -NoProfile -Command "Start-Sleep 15; Start-Process 'http://localhost:3000/monitor.html?server=http://localhost:8033&hw=http://localhost:8083'"

REM 3. llama-server with dual-GPU: local CUDA + remote RPC worker
"%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL%" ^
  --mmproj "%MMPROJ%" ^
  --fit on --fit-ctx 128000 --fit-target 256 ^
  -np 1 -fa on -t 8 ^
  -b 2048 -ub 2048 ^
  -ctk q8_0 -ctv q8_0 ^
  --jinja ^
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 ^
  --presence-penalty 0.0 --repeat-penalty 1.0 ^
  --chat-template-kwargs "{\"preserve_thinking\": true}" ^
  --host 0.0.0.0 --port 8033 ^
  --metrics --slots ^
  --rpc %RPC_HOST%
pause
