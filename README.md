# llama.cpp RPC Server (CUDA)

Prebuilt Docker image of the llama.cpp RPC server with CUDA GPU acceleration, built from the same commit as the Windows `llama-server.exe` binary.

## Why this image

Prebuilt images from other users had version mismatches with the Windows `llama-server.exe` (build 11869), causing RPC connection crashes. This image is built from the exact same commit (`d327bf1f0`, tag `b11869`) to ensure compatibility. It also uses CUDA 12.4 — newer versions (12.6+) trigger forward compatibility errors on RTX 3070.

Enables distributed LLM inference across multiple GPUs on different machines. The RPC server exposes a TCP endpoint that the Windows `llama-server` coordinator connects to, offloading model inference to a secondary GPU.

**Typical setup:**
- **Windows desktop** (inference coordinator) — `llama-server.exe` with local GPU (RTX 5070 Ti 16 GB)
- **Remote server** (RPC worker) — this container on a secondary GPU (RTX 3070 8 GB)
- Models loaded on the RPC worker are accessed over LAN via TCP port `50052`

## Hardware

| Component | Details |
|-----------|---------|
| RPC Worker GPU | NVIDIA GeForce RTX 3070 (8 GB VRAM, Ampere/sm_86) |
| Coordinator GPU | NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) |
| CPU | AMD Ryzen 7 3800X, 64 GB DDR4 |
| CUDA | 12.4.0 runtime |
| llama.cpp | Build 11869 (commit `d327bf1f0`) |
| Network | LAN-only, TCP 50052 |

## Image

```
id0ntplaygam3s/rpc-b11869:latest
```

## Dependencies (Windows coordinator)

Download the following to `C:\llama\` on your Windows machine:

### llama.cpp binaries (build 11869)
Get from the official llama.cpp releases or build from source:
- `llama-server.exe` — main inference server
- `llama.dll`, `llama-common.dll`, `llama-server-impl.dll` — runtime libraries
- `ggml.dll`, `ggml-base.dll`, `ggml-cuda.dll` — GGML backend (CUDA)
- `ggml-rpc.dll`, `ggml-rpc-server.exe` — RPC client/server
- CPU backend DLLs (`ggml-cpu-*.dll`) — fallback CPU inference
- `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll` — NVIDIA CUDA libraries
- `libomp.dll` — OpenMP runtime

### Monitoring tools
- `hw-metrics.py` — hardware metrics sidecar (GPU, CPU, memory)
- `monitor.html` — browser dashboard
- `serve-dual-gpu-With-Monitoring.bat` — dual-GPU launch script (local + RPC)

### Python
- Python 3.13+ (or adjust the `PY` path in the bat files)

## Usage

### Docker Run

```bash
docker run -d \
  --name llama-rpc-worker \
  --gpus all \
  --restart unless-stopped \
  -p 50052:50052 \
  -v /path/to/models:/models:rw \
  id0ntplaygam3s/rpc-b11869:latest \
  -H 0.0.0.0 -p 50052
```

### Docker Compose (TrueNAS SCALE)

```yaml
services:
  llama-rpc:
    image: id0ntplaygam3s/rpc-b11869:latest
    container_name: llama-rpc-worker
    restart: unless-stopped
    ports:
      - "50052:50052"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    command: ["-H", "0.0.0.0", "-p", "50052"]
    volumes:
      - /mnt/Applications/LLM/models:/models:rw
```

### Windows Coordinator (example)

Connect to the RPC worker from your Windows `llama-server`:

```bash
llama-server.exe -m model.gguf -ctk meta -ctk cuda -ngl 99 \
  --rpc 192.168.0.xxx:50052
```

## Building from source

```bash
docker build -t rpc-b11869 .
```

Or trigger a rebuild via GitHub Actions (repository dispatch).

## Notes

- GPU memory is **not reserved** at startup — allocated only when a model is loaded, freed when unloaded
- LAN-only deployment — no internet exposure, no reverse proxy
- Compatible with Windows `llama-server.exe` build 11869 (commit `d327bf1f0`)
