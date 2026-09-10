#!/usr/bin/env python3
"""Draw line charts of TinyLLaMA benchmark results.

Reads benchmark records from this directory (docs/bench) — files matching
*.txt that are not *.gen.txt — and plots per-model lines over time for the
main metrics:
    - Generate throughput (tok/s)
    - TTFT (ms)
    - Peak memory (MB)

Usage:
    chart.py [OUTPUT.png]        # default output: chart.png next to the script
"""

import argparse
import datetime
import glob
import os
import re
import sys

import matplotlib

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("error: matplotlib is required (pip install matplotlib)")

TS_RE = re.compile(r"_(\d{8})_(\d{6})\.txt$")
MODEL_RE = re.compile(r"^Model:\s+(\S+)", re.MULTILINE)
PERF_RE = re.compile(
    r"TTFT:\s+([\d.]+) ms \| Prefill:\s+([\d.]+) ms \| "
    r"Generate:\s+([\d.]+) ms \(\s*(\d+) token(?:\w)?, ([\d.]+) tok/s\)"
)
AVG_TTFT_RE = re.compile(r"^Avg TTFT:\s+([\d.]+) ms", re.MULTILINE)
AVG_GEN_RE = re.compile(
    r"^Avg Generate:\s+([\d.]+) ms \(\s*([\d.]+) tok/s\)", re.MULTILINE
)
PEAK_MEM_RE = re.compile(r"^Peak Memory:\s+([\d.]+) MB", re.MULTILINE)
MATMUL_RE = re.compile(r"^\s+matmul\s+([\d.]+)s\s+([\d.]+)%", re.MULTILINE)

BENCH_RE = re.compile(r"\[\d+/\d+\]\s+TTFT:")

HERE = os.path.dirname(os.path.abspath(__file__))


def benchmark_files(directory):
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        name = os.path.basename(path)
        if name.endswith(".gen.txt"):
            continue
        yield path


def parse_run(path):
    with open(path, "r", errors="replace") as fh:
        text = fh.read()

    name = os.path.basename(path)
    ts_match = TS_RE.search(name)
    if ts_match:
        ts = datetime.datetime.strptime(
            "{} {}".format(*ts_match.groups()), "%Y%m%d %H%M%S"
        )
    else:
        m = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)
        ts = (
            datetime.datetime.strptime(m.group(0), "%Y-%m-%d %H:%M:%S")
            if m
            else datetime.datetime.min
        )

    model = None
    m = MODEL_RE.search(text)
    if m:
        model = os.path.splitext(os.path.basename(m.group(1)))[0]
    if not model:
        model = name[: name.rfind("_")]

    perf = None
    m = PERF_RE.search(text)
    if m:
        ttft, prefill, gen_ms, gen_toks, gen_tps = m.groups()
        perf = {
            "ttft_ms": float(ttft),
            "prefill_ms": float(prefill),
            "gen_ms": float(gen_ms),
            "gen_tokens": int(gen_toks),
            "gen_tokps": float(gen_tps),
        }
    else:
        m = AVG_GEN_RE.search(text)
        if m:
            perf = {
                "gen_ms": float(m.group(1)),
                "gen_tokps": float(m.group(2)),
                "ttft_ms": None,
                "ttft_tokps": None,
            }
            m2 = AVG_TTFT_RE.search(text)
            perf["ttft_ms"] = float(m2.group(1)) if m2 else None

    mem = None
    m = PEAK_MEM_RE.search(text)
    if m:
        mem = float(m.group(1))

    matmul_pct = None
    m = MATMUL_RE.search(text)
    if m:
        matmul_pct = float(m.group(2))

    return name, {
        "ts": ts,
        "model": model,
        "perf": perf,
        "mem_mb": mem,
        "matmul_pct": matmul_pct,
    }


def collect(directory):
    runs = [parse_run(p) for p in benchmark_files(directory)]
    runs = [r for _, r in runs if r["perf"]]
    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)
    for runs_ in by_model.values():
        runs_.sort(key=lambda r: r["ts"])
    return by_model


def metric_key(metric):
    return {
        "generate": "gen_tokps",
        "ttft": "ttft_ms",
        "memory": "mem_mb",
        "matmul": "matmul_pct",
    }[metric]


def plot(by_model, out, metrics):
    metrics = metrics or ["generate", "ttft", "memory"]
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    ylabels = {
        "generate": "Generate throughput (tok/s)",
        "ttft": "TTFT (ms)",
        "memory": "Peak memory (MB)",
        "matmul": "MatMul share of runtime (%)",
    }

    for ax, metric in zip(axes, metrics):
        key = metric_key(metric)
        for model, runs in sorted(by_model.items()):
            xs = [r["ts"] for r in runs]
            ys = []
            for r in runs:
                if metric == "memory":
                    ys.append(r["mem_mb"])
                elif metric == "matmul":
                    ys.append(r["matmul_pct"])
                else:
                    perf = r["perf"]
                    ys.append(perf[key] if perf else None)
            if all(v is not None for v in ys):
                ax.plot(xs, ys, marker="o", label=model)
        ax.set_ylabel(ylabels[metric])
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="best",
            title="Run date: {}".format(datetime.date.today().isoformat()),
            ncol=1,
            fontsize="small",
        )
        ax.tick_params(axis="x", rotation=20)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("saved {}".format(out))
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()
    else:
        print("no interactive backend; chart written to {}".format(out))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", nargs="?", default=os.path.join(HERE, "chart.png"))
    ap.add_argument(
        "-d",
        "--dir",
        default=HERE,
        help="directory containing benchmark .txt files (default: script dir)",
    )
    ap.add_argument(
        "-m",
        "--metrics",
        default="generate,ttft,memory",
        help="comma-separated metrics to plot: generate,ttft,memory,matmul",
    )
    args = ap.parse_args(argv)

    by_model = collect(args.dir)
    if not by_model:
        sys.exit("no benchmark records found in {}".format(args.dir))
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    plot(by_model, args.out, metrics)


if __name__ == "__main__":
    main(sys.argv[1:])