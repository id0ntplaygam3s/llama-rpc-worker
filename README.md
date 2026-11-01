# llama.cpp RPC Server (CUDA)

Prebuilt Docker image of the llama.cpp RPC server with CUDA GPU acceleration, built from the same commit as the Windows `llama-server.exe` binary.

## Table of contents

- [Why this image](#why-this-image)
- [Hardware requirements](#hardware-requirements)
- [Image](#image)
- [Usage](#usage)
  - [Docker Run](#docker-run)
  - [Docker Compose (TrueNAS SCALE)](#docker-compose-truenas-scale)
  - [Windows Coordinator (example)](#windows-coordinator-example)
- [Dependencies (Windows coordinator)](#dependencies-windows-coordinator)
- [Benchmark results](#benchmark-results)
  - [Setup](#setup)
  - [Performance: single vs dual GPU](#performance-single-vs-dual-gpu)
  - [Key findings](#key-findings)
  - [Caveats](#caveats)
- [Benchmark your own setup](#benchmark-your-own-setup)
  - [Quick start](#quick-start)
  - [bench.py options](#benchpy-options)
  - [Compare output](#compare-output)
  - [Tips](#tips)
- [Building from source](#building-from-source)
- [Notes](#notes)

## Why this image

Prebuilt images from other users had version mismatches with the Windows `llama-server.exe` (build 11869), causing RPC connection crashes. This image is built from the exact same commit (`d327bf1f0`, tag `b11869`) to ensure compatibility. It also uses CUDA 12.4 — newer versions (12.6+) trigger forward compatibility errors on RTX 3070.

Enables distributed LLM inference across multiple GPUs on different machines. The RPC server exposes a TCP endpoint that the Windows `llama-server` coordinator connects to, offloading model inference to a secondary GPU.

## Hardware requirements

This Docker image runs on **any NVIDIA GPU with CUDA support** (Ampere architecture — sm_86 — and newer recommended, e.g. RTX 30-series and up). The llama.cpp build must match your Windows coordinator's build for RPC compatibility.

## Image

```
id0ntplaygam3s/rpc-b11869:latest
```

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
```

### Windows Coordinator (example)

Connect to the RPC worker from your Windows `llama-server`:

```bash
llama-server.exe -m model.gguf --fit on --fit-ctx 128000 --fit-target 256 -np 1 -fa on -ub 2048 -ctk q8_0 -ctv q8_0 --rpc 192.168.0.xxx:50052
```

Don't set `-ngl` together with `--fit on`: `--fit` chooses the layer/tensor split across the local GPU and the RPC device automatically.

## Dependencies (Windows coordinator)

Download the following on your Windows machine:

### llama.cpp binaries (build 11869)
Get from the [official llama.cpp releases](https://github.com/ggerganov/llama.cpp/releases) — download the `llama-server-win-cuda.zip` for your build (build 11869 / commit `d327bf1f0`), or build from source:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git checkout d327bf1f0
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

- `llama-server.exe` — main inference server
- `llama.dll`, `llama-common.dll`, `llama-server-impl.dll` — runtime libraries
- `ggml.dll`, `ggml-base.dll`, `ggml-cuda.dll` — GGML backend (CUDA)
- `ggml-rpc.dll`, `ggml-rpc-server.exe` — RPC client/server
- CPU backend DLLs (`ggml-cpu-*.dll`) — fallback CPU inference
- `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll` — NVIDIA CUDA libraries (included with CUDA 12.x or in the llama.cpp release zip)
- `libomp.dll` — OpenMP runtime (included with CUDA 12.x or in the llama.cpp release zip)

### Monitoring tools
- `hw-metrics.py` — hardware metrics sidecar (GPU, CPU, memory)
- `monitor.html` — browser dashboard
- `serve-dual-gpu-With-Monitoring.bat` — dual-GPU launch script (local + RPC)

### Python
- [Python 3.13+](https://www.python.org/downloads/) (or adjust the `PY` path in the bat files)

## Benchmark results

### Setup

| Component | Details |
|-----------|---------|
| Coordinator GPU | NVIDIA GeForce RTX 5070 Ti (16 GB VRAM) |
| Coordinator CPU | AMD Ryzen 7 5700X3D (8 cores / 16 threads), 64 GB DDR4 3600 MHz |
| RPC Worker GPU | NVIDIA GeForce RTX 3070 (8 GB VRAM, Ampere/sm_86) |
| RPC Worker CPU | Intel Xeon E5-2697A v4 (16 cores / 32 threads), 64 GB DDR4 |
| Model | Qwen3.6-35B-A3B, UD-Q4_K_XL quant |
| llama.cpp | Build 11869 (commit `d327bf1f0`) |
| Network | Wired gigabit LAN, RPC on TCP 50052 |

### What was tested

- **Context sizes:** 512, 1K, 2K, 4K, 8K, 16K, 24K, 32K, 48K, 64K, 75K tokens.
- **Runs:** 5 per size, in ascending order, 256 generated tokens per request.
- **Method:** Fixed-length filler prompts (random seed 42), prompt cache off, temperature 0, `ignore_eos`, speeds from the server's own timings.
- **Single flags:** `--fit on --fit-ctx 128000 --fit-target 256 -np 1 -fa on -t 8 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0`
- **Dual flags:** same + `--rpc <rpc_worker_ip>:50052`

### Full results (mean of 5 runs)

| Context | Prompt tok/s single | dual | Change | Gen tok/s single | dual | Change |
|---|---|---|---|---|---|---|
| 512 | 859 | 554 | −35% | 51.7 | 50.3 | −3% |
| 1K | 579 | 540 | −7% | 50.8 | 54.2 | +7% |
| 2K | 665 | 579 | −13% | 51.0 | 54.1 | +6% |
| 4K | 724 | 592 | −18% | 51.8 | 55.4 | +7% |
| 8K | 763 | 477 | −37% | 51.7 | 45.4 | −12% |
| 16K | 750 | 449 | −40% | 48.4 | 45.0 | −7% |
| 24K | 627 | 395 | −37% | 46.0 | 46.0 | 0% |
| 32K | 609 | 281 | −54% | 45.0 | 40.2 | −11% |
| 48K | 498 | 194 | −61% | 42.0 | 13.7 | −67% |
| 64K | 358 | 171 | −52% | 39.2 | 8.2 | −79% |
| 75K | 376 | 153 | −59% | 39.6 | 7.4 | −81% |

### Visual comparison

**Prompt processing (tok/s)** — higher is better

![Prompt Processing](benchmark/prompt-chart.png)

**Generation speed (tok/s)** — higher is better

![Generation Speed](benchmark/gen-chart.png)

\* dual wins at 1–4K generation (by about 6–7%, ≈ 1 s per turn).

### Key findings

1. **Single GPU is faster at prompt processing at every context size.** The gap widens from −7% at 1K to −61% at 48K.
2. **Dual GPU only wins at 1–4K generation** (by about 6–7%), worth roughly 1 second per turn. From 8K up, dual is equal or slower.
3. **Dual cliff at 48K and above:** dual generation drops to 7–14 tok/s while single holds 39–42 tok/s all the way to 75K. The cause was not established and may be specific to this setup (e.g. power/thermal limits or memory pressure on the worker machine), so treat dual's 48K+ numbers with caution.
4. **Per-turn cost** (2,000 new prompt tokens + 1,000 generated) at largest common context:
   - At 16K: single ≈ 23 s, dual ≈ 27 s
   - At 32K: single ≈ 25.5 s, dual ≈ 32 s
   - At 4K: roughly equal
5. **Cold prompt loads** (mean time to process a prompt from scratch):

   | Context | Single | Dual |
   |---|---|---|
   | 16K | ≈ 21 s | ≈ 37 s |
   | 48K | ≈ 107 s | ≈ 249 s |
   | 75K | ≈ 200 s | ≈ 490 s |

6. **Real-world measurement:** a real 75K-token opencode prompt measured about 165 tok/s on single and about 137 tok/s on dual.

### Recommendation

**Default to single GPU** for this model on this hardware. Dual may still make sense when a model does not fit otherwise (a larger model or quant than one GPU plus RAM can hold); that case was not tested here.

### Caveats

- Filler-text prompts make absolute prompt speeds higher than real prompts — trust the ratios more than the raw numbers.
- Single-GPU prompt speed also varied at 48K+ (about 320–680 tok/s).
- Results will differ on other hardware, models, and networks.
- Dual GPU needs the RPC worker reachable on port 50052 before `llama-server` starts.

## Benchmark your own setup

The `benchmark/` folder contains `bench.py` (Python 3, standard library only — no `pip` installs needed) which sends identical fixed-length prompts to a running `llama-server` and prints prompt and generation speed for each context size, writing a CSV.

### Quick start

1. **Edit the variables** at the top of `benchmark/bench-single.bat` and `benchmark/bench-dual.bat` (llama.cpp folder, model path, Python path, RPC address).
2. **Close other `llama-server` instances** and any client using the GPU (the batch files run `taskkill` on `llama-server.exe`, which stops **all** running `llama-server` processes).
3. **Run `bench-single.bat`** — this measures single-GPU performance.
4. **Start the RPC worker** and run `bench-dual.bat` — this measures dual-GPU performance.
5. **Compare** with: `python benchmark/bench.py compare benchmark/results-single.csv benchmark/results-dual.csv`

### bench.py options

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

### Example full run (from these results)

```
python benchmark/bench.py --label single --out benchmark/results-single.csv ^
  --sizes 512 1000 2000 4000 8000 16000 24000 32000 48000 64000 75000 ^
  --runs 5 --gen 256 --wait
```

On this hardware, a single run takes roughly **one hour**. Dual takes up to **two hours**.

### Compare output

The `compare` subcommand prints prompt and generation speeds per context size for each setup, the percentage change, and an estimated time for a turn of 2,000 new prompt tokens + 1,000 generated tokens at the deepest common context.

### Tips

- Nothing else should use the PC or network during a run.
- Use `--runs 3` or more for steadier numbers.
- Expect the first run after startup to be slower (cold model load).
- Watch for speeds that decay run after run — this can indicate throttling or memory pressure.

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
