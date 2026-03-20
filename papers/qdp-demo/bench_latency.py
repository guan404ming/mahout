#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run all benchmarks on Modal A100.
3 encodings x 4 qubits x 3 frameworks + IQP Mahout-only.

Usage:
    modal run bench_repo.py
"""

import modal

app = modal.App("qdp-bench-v2")

# Use debian_slim + install CUDA toolkit via pip (lighter than nvidia/cuda image)
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
        # Install CUDA 12.5 toolkit (nvcc + cudart)
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-5",
    )
    .pip_install(
        "numpy>=1.24,<2.0",
        "maturin[patchelf]",
        "torch",
        "pennylane>=0.35",
        "qiskit>=1.0",
        "qiskit-aer>=0.17.2",
    )
    .run_commands(
        # Install Rust
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        # Clone repo
        'export PATH="$HOME/.cargo/bin:$PATH" && git clone --depth 1 https://github.com/apache/mahout.git /opt/mahout',
        # Build wheel with CUDA
        'export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.5/bin:$PATH" && export CUDA_PATH=/usr/local/cuda-12.5 && cd /opt/mahout/qdp/qdp-python && maturin build --release',
        # Install wheel
        "python -c \"import glob,subprocess; whl=glob.glob('/opt/mahout/qdp/target/wheels/qumat_qdp-*.whl')[0]; subprocess.run(['pip','install',whl],check=True)\"",
    )
)

QUBITS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
BATCHES = 50
BATCH_SIZE = 64


def num_vectors(nq):
    """Fewer vectors at high qubit counts to avoid timeout."""
    if nq >= 20:
        return 5
    if nq >= 18:
        return 10
    if nq >= 16:
        return 20
    if nq >= 14:
        return 50
    return 200


@app.function(image=gpu_image, gpu="A100", timeout=1200)
def run_benchmarks():
    import os
    import sys
    import time

    import numpy as np
    import torch

    # Set LD_LIBRARY_PATH to find CUDA libs from torch
    cuda_lib_dirs = []
    for p in sys.path:
        for name in [
            "nvidia/cuda_runtime/lib",
            "nvidia/cublas/lib",
            "nvidia/cudnn/lib",
        ]:
            candidate = os.path.join(p, name)
            if os.path.isdir(candidate):
                cuda_lib_dirs.append(candidate)
    if cuda_lib_dirs:
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(cuda_lib_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # Try importing QDP
    try:
        from qumat_qdp import QdpBenchmark

        has_qdp = True
        print("QDP loaded successfully")
    except Exception as e:
        has_qdp = False
        print(f"QDP import failed: {e}")

    import pennylane as qml
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    # =====================================================
    # AMPLITUDE
    # =====================================================
    for nq in QUBITS:
        nv = num_vectors(nq)
        vec_len = 1 << nq
        data = normalize(np.random.randn(nv, vec_len))

        # Mahout
        if has_qdp:
            try:
                r = (
                    QdpBenchmark(device_id=0)
                    .qubits(nq)
                    .encoding("amplitude")
                    .batches(BATCHES, size=BATCH_SIZE)
                    .warmup(5)
                    .run_latency()
                )
                results.append(
                    {
                        "framework": "Mahout",
                        "encoding": "amplitude",
                        "qubits": nq,
                        "ms_per_vec": round(r.latency_ms_per_vector, 4),
                    }
                )
                print(f"Mahout amplitude {nq}q: {r.latency_ms_per_vector:.4f}")
            except Exception as e:
                print(f"Mahout amplitude {nq}q FAILED: {e}")

        # PennyLane
        dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(dev, interface="torch")
        def pl_amp(x):
            qml.AmplitudeEmbedding(
                features=x, wires=range(nq), normalize=True, pad_with=0.0
            )
            return qml.state()

        sync()
        t0 = time.perf_counter()
        for row in data:
            s = pl_amp(torch.tensor(row, dtype=torch.float64))
            _ = s.to("cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "PennyLane",
                "encoding": "amplitude",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"PennyLane amplitude {nq}q: {ms:.4f}")

        # Qiskit (initialize - circuit-based state preparation)
        sync()
        t0 = time.perf_counter()
        for row in data:
            qc = QuantumCircuit(nq)
            qc.initialize(row.tolist(), range(nq))
            sv = Statevector.from_instruction(qc)
            _ = torch.tensor(sv.data, device="cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "Qiskit",
                "encoding": "amplitude",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"Qiskit amplitude {nq}q: {ms:.4f}")

    # =====================================================
    # ANGLE
    # =====================================================
    for nq in QUBITS:
        nv = num_vectors(nq)
        angle_data = np.random.randn(nv, nq)

        if has_qdp:
            try:
                r = (
                    QdpBenchmark(device_id=0)
                    .qubits(nq)
                    .encoding("angle")
                    .batches(BATCHES, size=BATCH_SIZE)
                    .warmup(5)
                    .run_latency()
                )
                results.append(
                    {
                        "framework": "Mahout",
                        "encoding": "angle",
                        "qubits": nq,
                        "ms_per_vec": round(r.latency_ms_per_vector, 4),
                    }
                )
                print(f"Mahout angle {nq}q: {r.latency_ms_per_vector:.4f}")
            except Exception as e:
                print(f"Mahout angle {nq}q FAILED: {e}")

        dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(dev, interface="torch")
        def pl_angle(x):
            qml.AngleEmbedding(features=x, wires=range(nq))
            return qml.state()

        sync()
        t0 = time.perf_counter()
        for row in angle_data:
            s = pl_angle(torch.tensor(row, dtype=torch.float64))
            _ = s.to("cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "PennyLane",
                "encoding": "angle",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"PennyLane angle {nq}q: {ms:.4f}")

        sync()
        t0 = time.perf_counter()
        for row in angle_data:
            qc = QuantumCircuit(nq)
            for i, a in enumerate(row):
                qc.ry(float(a), i)
            sv = Statevector.from_instruction(qc)
            _ = torch.tensor(sv.data, device="cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "Qiskit",
                "encoding": "angle",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"Qiskit angle {nq}q: {ms:.4f}")

    # =====================================================
    # BASIS
    # =====================================================
    for nq in QUBITS:
        nv = num_vectors(nq)

        if has_qdp:
            try:
                r = (
                    QdpBenchmark(device_id=0)
                    .qubits(nq)
                    .encoding("basis")
                    .batches(BATCHES, size=BATCH_SIZE)
                    .warmup(5)
                    .run_latency()
                )
                results.append(
                    {
                        "framework": "Mahout",
                        "encoding": "basis",
                        "qubits": nq,
                        "ms_per_vec": round(r.latency_ms_per_vector, 4),
                    }
                )
                print(f"Mahout basis {nq}q: {r.latency_ms_per_vector:.4f}")
            except Exception as e:
                print(f"Mahout basis {nq}q FAILED: {e}")

        dev = qml.device("default.qubit", wires=nq)

        @qml.qnode(dev, interface="torch")
        def pl_basis(x):
            qml.BasisEmbedding(features=x, wires=range(nq))
            return qml.state()

        sync()
        t0 = time.perf_counter()
        for _ in range(nv):
            bits = np.random.randint(0, 2, size=nq)
            s = pl_basis(bits)
            _ = s.to("cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "PennyLane",
                "encoding": "basis",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"PennyLane basis {nq}q: {ms:.4f}")

        sync()
        t0 = time.perf_counter()
        for _ in range(nv):
            k = np.random.randint(0, 1 << nq)
            qc = QuantumCircuit(nq)
            for i in range(nq):
                if (k >> i) & 1:
                    qc.x(i)
            sv = Statevector.from_instruction(qc)
            _ = torch.tensor(sv.data, device="cuda", dtype=torch.complex64)
        sync()
        ms = ((time.perf_counter() - t0) / nv) * 1000
        results.append(
            {
                "framework": "Qiskit",
                "encoding": "basis",
                "qubits": nq,
                "ms_per_vec": round(ms, 4),
            }
        )
        print(f"Qiskit basis {nq}q: {ms:.4f}")

    # =====================================================
    # IQP: Mahout only (use QdpEngine.encode with correct input size)
    # IQP expects n + n*(n-1)/2 parameters per vector
    # =====================================================
    IQP_QUBITS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    if has_qdp:
        from qumat_qdp import QdpEngine

        engine = QdpEngine(0)
        for nq in IQP_QUBITS:
            iqp_dim = nq + nq * (nq - 1) // 2
            nv = 200
            data = np.random.randn(nv, iqp_dim)
            try:
                sync()
                t0 = time.perf_counter()
                for row in data:
                    qt = engine.encode(row.tolist(), nq, "iqp")
                    _ = torch.from_dlpack(qt)
                sync()
                ms = ((time.perf_counter() - t0) / nv) * 1000
                results.append(
                    {
                        "framework": "Mahout",
                        "encoding": "iqp",
                        "qubits": nq,
                        "ms_per_vec": round(ms, 4),
                    }
                )
                print(f"Mahout iqp {nq}q: {ms:.4f}")
            except Exception as e:
                print(f"Mahout iqp {nq}q FAILED: {e}")

    return results


@app.local_entrypoint()
def main():
    import json

    results = run_benchmarks.remote()

    with open("bench_all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results")
    print(f"\n{'Framework':12s} {'Encoding':10s} {'Qubits':>6s} {'ms/vec':>10s}")
    print("-" * 42)
    for r in results:
        if isinstance(r, dict) and "ms_per_vec" in r:
            print(
                f"{r['framework']:12s} {r['encoding']:10s} {r['qubits']:6d} {r['ms_per_vec']:10.4f}"
            )
