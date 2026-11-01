#!/usr/bin/env python3
"""Benchmark a running llama-server with identical, fixed prompts.

Run a benchmark:
  python bench.py --label single --out results-single.csv --wait
Compare two result files:
  python bench.py compare results-single.csv results-dual.csv

Each request uses an exact token count, a fixed seed, no prompt cache and
ignore_eos, so every run does the same amount of work on both setups.
Speeds come from the server's own timings, so they match the real launch flags.
"""
import argparse, csv, json, random, statistics, sys, time, urllib.request, urllib.error


def call(url, payload=None, timeout=7200):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_ready(base, limit_s):
    start = time.time()
    while time.time() - start < limit_s:
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(3)
    sys.exit("Server did not become ready in time.")


def make_tokens(base, n):
    rnd = random.Random(42)  # same text every time, on both setups
    vocab = ["delta", "river", "matrix", "orbit", "violet", "cache", "lantern", "vector",
             "harbor", "signal", "copper", "meadow", "kernel", "falcon", "garden", "socket",
             "ribbon", "thunder", "pixel", "anchor", "marble", "engine", "window", "forest"]
    text = " ".join(rnd.choice(vocab) + str(rnd.randint(0, 999)) for _ in range(n))
    toks = call(base + "/tokenize", {"content": text, "add_special": False})["tokens"]
    if len(toks) < n:
        sys.exit(f"Could only build {len(toks)} tokens, wanted {n}.")
    return toks[:n]


def run_bench(a):
    base = a.url.rstrip("/")
    if a.wait:
        wait_ready(base, 1800)
    print(f"[{a.label}] warm-up...")
    call(base + "/completion", {"prompt": make_tokens(base, 256), "n_predict": 32,
                                "cache_prompt": False, "temperature": 0, "seed": 1})
    rows = []
    for size in a.sizes:
        toks = make_tokens(base, size)
        for i in range(1, a.runs + 1):
            r = call(base + "/completion", {
                "prompt": toks, "n_predict": a.gen, "ignore_eos": True,
                "cache_prompt": False, "temperature": 0, "seed": 1, "stream": False})
            t = r["timings"]
            row = {"label": a.label, "prompt_tokens": t["prompt_n"], "run": i,
                   "pp_tps": round(t["prompt_per_second"], 2),
                   "tg_tps": round(t["predicted_per_second"], 2),
                   "pp_s": round(t["prompt_ms"] / 1000, 2),
                   "tg_s": round(t["predicted_ms"] / 1000, 2),
                   "gen_tokens": t["predicted_n"]}
            rows.append(row)
            print(f"  ctx {size:>6} run {i}: prompt {row['pp_tps']:>7} tok/s "
                  f"({row['pp_s']}s) | gen {row['tg_tps']:>6} tok/s ({row['gen_tokens']} tok)")
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {a.out}")


def load(path):
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["prompt_tokens"]), []).append(r)
    return out


def mean(rows, key):
    return statistics.mean(float(r[key]) for r in rows)


def compare(a_path, b_path):
    a, b = load(a_path), load(b_path)
    print(f"A = {a_path}   B = {b_path}\n")
    print(f"{'ctx':>7} | {'prompt A':>9} {'prompt B':>9} {'B vs A':>8} | "
          f"{'gen A':>7} {'gen B':>7} {'B vs A':>8}")
    for size in sorted(set(a) & set(b)):
        pa, pb = mean(a[size], "pp_tps"), mean(b[size], "pp_tps")
        ga, gb = mean(a[size], "tg_tps"), mean(b[size], "tg_tps")
        print(f"{size:>7} | {pa:>9.1f} {pb:>9.1f} {100*(pb/pa-1):>+7.1f}% | "
              f"{ga:>7.1f} {gb:>7.1f} {100*(gb/ga-1):>+7.1f}%")
    print("\n(tok/s, higher is better; B vs A is the change when using B)")
    # Example turn: 2000 new prompt tokens + 1000 generated, at the largest common depth
    size = max(set(a) & set(b))
    for name, d in (("A", a), ("B", b)):
        t = 2000 / mean(d[size], "pp_tps") + 1000 / mean(d[size], "tg_tps")
        print(f"Estimated time for a turn with 2000 new prompt + 1000 generated tokens "
              f"at {size} ctx, {name}: {t:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
        sys.exit()
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8033")
    p.add_argument("--label", default="run")
    p.add_argument("--out", default="results.csv")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 16000, 48000],
                   help="prompt sizes in tokens")
    p.add_argument("--runs", type=int, default=2, help="repeats per size")
    p.add_argument("--gen", type=int, default=256, help="tokens to generate per request")
    p.add_argument("--wait", action="store_true", help="wait for /health before starting")
    run_bench(p.parse_args())
