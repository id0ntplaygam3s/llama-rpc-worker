@echo off
REM Dual-GPU distributed inference: RTX 5070 Ti (local) + RTX 3070 (remote RPC worker)
REM Launches: hardware sidecar (port 8083), llama-server (port 8033), dashboard in browser.
REM
REM Hardware:
REM   Local GPU:  RTX 5070 Ti 16 GB (Ryzen 7 3800X, 64 GB DDR4)
REM   Remote GPU: RTX 3070 8 GB (remote server, LAN)
REM
REM Remote RPC worker must be running on port 50052 before launching.

set PY=C:\path\to\python\python.exe

REM 1. Hardware sidecar (called with python.exe directly, bypassing the py launcher / shebang)
start "hw-metrics" /min "%PY%" C:\llama\hw-metrics.py --host 127.0.0.1 --port 8083

REM 2. Dashboard in default browser once servers have had a moment to start
start "dashboard-http" /min "%PY%" -m http.server 3000 --bind 127.0.0.1 --directory C:\llama
start "" /b powershell -NoProfile -Command "Start-Sleep 15; Start-Process 'http://localhost:3000/monitor.html?server=http://localhost:8033&hw=http://localhost:8083'"

REM 3. llama-server with dual-GPU: local CUDA + remote RPC worker
C:\llama\llama-server.exe ^
  -m C:\models\model.gguf ^
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
  --rpc 192.168.0.xxx:50052
pause
