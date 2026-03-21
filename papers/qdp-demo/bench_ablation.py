#!/usr/bin/env python3
"""
Ablation: 3 pipeline configs, all reading from Parquet.
1. Per-vector: read Parquet -> engine.encode() per row
2. Batched: read Parquet -> engine.encode_batch() in chunks
3. Full pipeline: QuantumDataLoader.source_file(parquet)

Usage:
    modal run bench_ablation.py
"""

import modal

app = modal.App("qdp-ablation-v2")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "curl",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "wget",
        "gnupg",
        "software-properties-common",
    )
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-5",
    )
    .pip_install("numpy>=1.24,<2.0", "maturin[patchelf]", "torch", "pyarrow", "pandas")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        'export PATH="$HOME/.cargo/bin:$PATH" && git clone --depth 1 https://github.com/apache/mahout.git /opt/mahout',
        'export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.5/bin:$PATH" && export CUDA_PATH=/usr/local/cuda-12.5 && cd /opt/mahout/qdp/qdp-python && maturin build --release',
        "python -c \"import glob,subprocess; whl=glob.glob('/opt/mahout/qdp/target/wheels/qumat_qdp-*.whl')[0]; subprocess.run(['pip','install',whl],check=True)\"",
    )
)

QUBITS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
BATCH_SIZE = 64
RUNS = 10


def total_samples(nq):
    """Keep Parquet ~400MB max."""
    if nq >= 20:
        return 64
    if nq >= 18:
        return 128
    if nq >= 16:
        return 768
    if nq >= 14:
        return 3200
    return 12800


@app.function(image=gpu_image, gpu="A100", timeout=1800)
def run_ablation(nq: int):
    import os
    import time

    import numpy as np
    import torch
    from qumat_qdp import QdpEngine, QuantumDataLoader

    torch.cuda.synchronize()
    vec_len = 1 << nq
    ns = total_samples(nq)
    num_batches = ns // BATCH_SIZE

    # Generate Parquet file
    tmpdir = f"/tmp/ablation_{nq}q"
    os.makedirs(tmpdir, exist_ok=True)
    parquet_path = f"{tmpdir}/data.parquet"

    data = np.random.randn(ns, vec_len)
    # Use .npy for all - QDP's Parquet reader expects single-column format
    file_path = f"{tmpdir}/data.npy"
    np.save(file_path, data)
    fsize = os.path.getsize(file_path) / 1e6
    print(
        f"{nq}q: wrote {file_path} ({fsize:.1f} MB, {ns} samples x {vec_len} features)"
    )

    engine = QdpEngine(0)
    results = {}

    def read_file(limit=None):
        d = np.load(file_path)
        return d[:limit] if limit else d

    # --- Config 1: Per-vector (read file + encode, all timed) ---
    pv_count = min(ns, 200)
    pv_vals = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        read_data = read_file(pv_count)
        for row in read_data:
            qt = engine.encode(row.tolist(), nq, "amplitude")
            _ = torch.from_dlpack(qt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pv_vals.append(pv_count / elapsed)
    results["per_vector"] = float(np.median(pv_vals))
    print(f"{nq}q per-vector: {results['per_vector']:.1f} vec/s")

    # --- Config 2: Batched (read file + encode, all timed) ---
    batch_vals = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        read_data = read_file().astype(np.float64)
        for i in range(0, ns, BATCH_SIZE):
            chunk = np.ascontiguousarray(read_data[i : i + BATCH_SIZE])
            qt = engine.encode(chunk, nq, "amplitude")
            _ = torch.from_dlpack(qt)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        batch_vals.append(ns / elapsed)
    results["batched"] = float(np.median(batch_vals))
    print(f"{nq}q batched: {results['batched']:.1f} vec/s")

    # --- Config 3: Full pipeline (QuantumDataLoader, Rust IO + dual-stream) ---
    pipe_vals = []
    for _ in range(RUNS):
        loader = (
            QuantumDataLoader(device_id=0)
            .qubits(nq)
            .encoding("amplitude")
            .batches(num_batches, size=BATCH_SIZE)
            .source_file(file_path)
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        count = 0
        for qt in loader:
            _ = torch.from_dlpack(qt)
            count += BATCH_SIZE
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        pipe_vals.append(count / elapsed)
    results["pipeline"] = float(np.median(pipe_vals))
    print(f"{nq}q pipeline: {results['pipeline']:.1f} vec/s")

    return {"qubits": nq, **results}


@app.local_entrypoint()
def main():
    import json

    futures = [(nq, run_ablation.spawn(nq)) for nq in QUBITS]
    results = []
    for nq, f in futures:
        try:
            r = f.get()
            results.append(r)
        except Exception as e:
            print(f"{nq}q FAILED: {e}")

    results.sort(key=lambda x: x["qubits"])

    with open("bench_ablation.json", "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"\n{'Q':>3s} {'Per-vector':>12s} {'Batched':>12s} {'Pipeline':>12s} {'Batch/PV':>10s} {'Pipe/Batch':>10s}"
    )
    print("-" * 62)
    for r in results:
        pv = r["per_vector"]
        ba = r["batched"]
        pi = r["pipeline"]
        print(
            f"{r['qubits']:3d} {pv:12.0f} {ba:12.0f} {pi:12.0f} {ba / pv:10.1f}x {pi / ba:10.1f}x"
        )
