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

Download the following on your Windows machine:

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
llama-server.exe -m model.gguf --fit on --fit-ctx 128000 --fit-target 256 -np 1 -fa on -ub 2048 -ctk q8_0 -ctv q8_0 --rpc 192.168.0.xxx:50052
```

Don't set `-ngl` together with `--fit on`: `--fit` chooses the layer/tensor split across the local GPU and the RPC device automatically.

### Performance: single vs dual GPU

Dual-GPU mainly speeds up generation at moderate context and slows prompt processing on large batches; see Performance below.

#### Benchmark setup

- **Coordinator:** Windows desktop, AMD Ryzen 7 3800X (8 cores / 16 threads), 64 GB DDR4, RTX 5070 Ti 16 GB, llama-server.exe build 11869 (commit `d327bf1f0`).
- **RPC worker:** TrueNAS SCALE, Intel Xeon E5-2697A v4 (16 cores / 32 threads), 64 GB DDR4, RTX 3070 8 GB, Docker image (CUDA 12.4, build 11869).
- **Network:** wired gigabit LAN, RPC on TCP 50052.
- **Model:** Qwen3.6-35B-A3B, UD-Q4_K_XL quant. It does not fit entirely in 16 GB, so `--fit` keeps the remainder in system RAM; with dual, `--fit` can place more of the model on the GPUs.
- **Server flags:** `--fit on --fit-ctx 128000 --fit-target 256 -np 1 -fa on -t 8 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0` (dual adds `--rpc <worker>:50052`).
- **Method:** fixed prompts of exact token counts (random filler text), 256 generated tokens per request (`ignore_eos`), prompt cache off, temperature 0 and fixed seed, 2 runs per context size, speeds taken from the server's own timings. Single and dual were run back to back.

#### Results

| Context | Prompt tok/s (single -> dual) | Gen tok/s (single -> dual) |
|---------|-------------------------------|----------------------------|
| 1K  | 723 -> 498 (-31%) | 47.2 -> 51.3 (+9%) |
| 16K | 521 -> 554 (+6%)  | 38.6 -> 49.0 (+27%) |
| 48K | 575 -> 404 (-30%) | 39.9 -> 40.7 (+2%) |

A real 75K-token opencode prompt measured about 165 tok/s on single and about 137 tok/s on dual.

#### When to use which

- **Dual GPU** for short or medium sessions (under roughly 30-35K tokens of context): replies are about 10-25% faster, saving roughly 5-6 s per turn.
- **Single GPU** for large contexts or long sessions: generation is about equal by 48K, and a cold 48K prompt load takes about 119 s on dual versus 84 s on single. Single also has no network or TrueNAS dependency.

#### Caveats

Only 2 runs per point; the filler-text prompts make absolute prompt speeds about 3× higher than real prompts, so trust the ratios more than the raw numbers. Results will differ on other hardware, models and networks.

### Benchmark your own setup

The `benchmark/` folder contains `bench.py` (Python 3, standard library only — no `pip` installs needed) which sends identical fixed-length prompts to a running `llama-server` and prints prompt and generation speed for each context size, writing a CSV.

#### Quick start

1. **Edit the variables** at the top of `benchmark/bench-single.bat` and `benchmark/bench-dual.bat` (llama.cpp folder, model path, Python path, RPC address).
2. **Close other `llama-server` instances** and any client using the GPU (the batch files run `taskkill` on `llama-server.exe`, which stops **all** running `llama-server` processes).
3. **Run `bench-single.bat`** — this measures single-GPU performance.
4. **Start the RPC worker** and run `bench-dual.bat` — this measures dual-GPU performance.
5. **Compare** with: `python benchmark/bench.py compare benchmark/results-single.csv benchmark/results-dual.csv`

#### bench.py options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://127.0.0.1:8033` | llama-server API endpoint |
| `--label` | `run` | Label written to the CSV |
| `--out` | `results.csv` | Output CSV path |
| `--sizes` | `1000 16000 48000` | Prompt sizes in tokens |
| `--runs` | `2` | Repeats per size |
| `--gen` | `256` | Tokens to generate per request |
| `--wait` | (off) | Wait for `/health` before starting |

It also works against any running `llama-server`, not just the batch files.

#### Compare output

The `compare` subcommand prints prompt and generation speeds per context size for each setup, the percentage change, and an estimated time for a turn of 2000 new prompt tokens plus 1000 generated tokens at the deepest context.

#### Tips

- Run both setups back to back with nothing else using the PC or network.
- Use `--runs 3` or more for steadier numbers.
- Expect the first run after startup to be a little slower (cold model load).

## Building from source

```bash
docker build -t rpc-b11869 .
```

Or trigger a rebuild via GitHub Actions (repository dispatch).

## Notes

- GPU memory is **not reserved** at startup — allocated only when a model is loaded, freed when unloaded
- LAN-only deployment — no internet exposure, no reverse proxy
- Compatible with Windows `llama-server.exe` build 11869 (commit `d327bf1f0`)
- With `--fit on`, `--tensor-split` is ignored or overridden. For manual placement use `--fit off` with `-ngl`, `--n-cpu-moe` or `-ot`.
- Dual-GPU needs the RPC worker reachable on port 50052 before `llama-server` starts.
