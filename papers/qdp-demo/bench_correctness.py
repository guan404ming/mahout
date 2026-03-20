#!/usr/bin/env python3
"""
Correctness validation: compare QDP encoded states against PennyLane and Qiskit.
Runs on Modal A100.

For each (encoding, qubit_count), encode the same input data with all frameworks
and compute max probability difference.

Usage:
    modal run bench_correctness.py
"""

import modal

app = modal.App("qdp-bench-correct")

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential", "pkg-config", "libssl-dev",
                 "wget", "gnupg", "software-properties-common")
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get install -y cuda-toolkit-12-5",
    )
    .pip_install(
        "numpy>=1.24,<2.0", "maturin[patchelf]", "torch",
        "pennylane>=0.35", "qiskit>=1.0",
    )
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        'export PATH="$HOME/.cargo/bin:$PATH" && git clone --depth 1 https://github.com/apache/mahout.git /opt/mahout',
        'export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.5/bin:$PATH" && export CUDA_PATH=/usr/local/cuda-12.5 && cd /opt/mahout/qdp/qdp-python && maturin build --release',
        'python -c "import glob,subprocess; whl=glob.glob(\'/opt/mahout/qdp/target/wheels/qumat_qdp-*.whl\')[0]; subprocess.run([\'pip\',\'install\',whl],check=True)"',
    )
)


@app.function(image=gpu_image, gpu="A100", timeout=600)
def run():
    import numpy as np
    import torch
    import pennylane as qml
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    from qumat_qdp import QdpEngine

    engine = QdpEngine(0, precision="float64")
    results = []

    def num_samples(nq):
        if nq >= 18: return 5
        if nq >= 14: return 10
        return 50

    def normalize(v):
        n = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.where(n == 0, 1.0, n)

    QUBITS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    # ---- AMPLITUDE ----
    for nq in QUBITS:
        vec_len = 1 << nq
        ns = num_samples(nq)
        data = np.random.randn(ns, vec_len)

        # QDP
        qdp_states = []
        for row in data:
            qt = engine.encode(row.tolist(), nq, "amplitude")
            s = torch.from_dlpack(qt).cpu().numpy().flatten()
            qdp_states.append(s)

        # PennyLane
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def pl_circuit(x):
            qml.AmplitudeEmbedding(features=x, wires=range(nq), normalize=True)
            return qml.state()
        pl_states = [np.array(pl_circuit(row)) for row in data]

        # Qiskit
        data_norm = normalize(data)
        qk_states = [Statevector(row).data for row in data_norm]

        # Compare probabilities
        # Compare amplitudes directly
        diffs_pl = [np.max(np.abs(q.flatten() - p.flatten())) for q, p in zip(qdp_states, pl_states)]
        diffs_qk = [np.max(np.abs(q.flatten() - k.flatten())) for q, k in zip(qdp_states, qk_states)]
        diff_pl = max(diffs_pl)
        diff_qk = max(diffs_qk)

        results.append({
            "encoding": "amplitude", "qubits": nq,
            "max_diff_vs_pennylane": float(f"{diff_pl:.2e}"),
            "max_diff_vs_qiskit": float(f"{diff_qk:.2e}"),
            "pass_pl": bool(diff_pl < 1e-10),
            "pass_qk": bool(diff_qk < 1e-10),
        })
        print(f"amplitude {nq}q: vs PL={diff_pl:.2e} vs Qiskit={diff_qk:.2e} {'PASS' if diff_pl < 1e-10 and diff_qk < 1e-10 else 'FAIL'}")

    # ---- ANGLE ----
    # QDP: product state cos(x_k)|0> + sin(x_k)|1> (real-valued, no phase)
    # PennyLane RY: cos(x/2)|0> + sin(x/2)|1> (complex, has phase)
    # Compare probabilities (|a|^2) since phases differ by design
    for nq in QUBITS:
        ns = num_samples(nq)
        data = np.random.randn(ns, nq)

        # QDP
        qdp_states = []
        for row in data:
            qt = engine.encode(row.tolist(), nq, "angle")
            qdp_states.append(torch.from_dlpack(qt).cpu().numpy().flatten())

        # PennyLane: feed 2*angle so RY(2x) gives cos(x)|0> + sin(x)|1>
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def pl_angle(x):
            qml.AngleEmbedding(features=x, wires=range(nq))
            return qml.state()
        pl_states = [np.array(pl_angle(2.0 * row)) for row in data]

        # Qiskit: RY(2*angle)
        qk_states = []
        for row in data:
            qc = QuantumCircuit(nq)
            for i, a in enumerate(row):
                qc.ry(2.0 * float(a), i)
            qk_states.append(Statevector.from_instruction(qc).data)

        # Compare probabilities (phase-invariant)
        diffs_pl = [np.max(np.abs(np.abs(q)**2 - np.abs(p)**2)) for q, p in zip(qdp_states, pl_states)]
        diffs_qk = [np.max(np.abs(np.abs(q)**2 - np.abs(k)**2)) for q, k in zip(qdp_states, qk_states)]
        diff_pl = max(diffs_pl)
        diff_qk = max(diffs_qk)

        results.append({
            "encoding": "angle", "qubits": nq,
            "max_diff_vs_pennylane": float(f"{diff_pl:.2e}"),
            "max_diff_vs_qiskit": float(f"{diff_qk:.2e}"),
            "pass_pl": bool(diff_pl < 1e-10),
            "pass_qk": bool(diff_qk < 1e-10),
        })
        print(f"angle {nq}q: vs PL={diff_pl:.2e} vs Qiskit={diff_qk:.2e} {'PASS' if diff_pl < 1e-10 and diff_qk < 1e-10 else 'FAIL'}")

    # ---- BASIS ----
    # Debug showed QDP basis([1,1]) matches PennyLane BasisEmbedding([1,1])
    for nq in QUBITS:
        ns = num_samples(nq)
        dev = qml.device("default.qubit", wires=nq)
        @qml.qnode(dev)
        def pl_basis(x):
            qml.BasisEmbedding(features=x, wires=range(nq))
            return qml.state()

        diffs_pl = []
        diffs_qk = []
        for _ in range(ns):
            k = np.random.randint(0, 1 << nq)

            # QDP
            qt = engine.encode([float(k)], nq, "basis")
            qdp_s = torch.from_dlpack(qt).cpu().numpy().flatten()

            # PennyLane: LSB-first bit array
            bits = [(k >> i) & 1 for i in range(nq)]
            pl_s = np.array(pl_basis(bits))

            # Qiskit
            qc = QuantumCircuit(nq)
            for i in range(nq):
                if (k >> i) & 1:
                    qc.x(i)
            qk_s = Statevector.from_instruction(qc).data

            diffs_pl.append(np.max(np.abs(qdp_s - pl_s)))
            diffs_qk.append(np.max(np.abs(qdp_s - qk_s)))

        diff_pl = max(diffs_pl)
        diff_qk = max(diffs_qk)

        results.append({
            "encoding": "basis", "qubits": nq,
            "max_diff_vs_pennylane": float(f"{diff_pl:.2e}"),
            "max_diff_vs_qiskit": float(f"{diff_qk:.2e}"),
            "pass_pl": bool(diff_pl < 1e-10),
            "pass_qk": bool(diff_qk < 1e-10),
        })
        print(f"basis {nq}q: vs PL={diff_pl:.2e} vs Qiskit={diff_qk:.2e} {'PASS' if diff_pl < 1e-10 and diff_qk < 1e-10 else 'FAIL'}")

    return results


@app.local_entrypoint()
def main():
    import json
    results = run.remote()

    with open("bench_correctness.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Encoding':10s} {'Qubits':>6s} {'vs PL':>12s} {'vs Qiskit':>12s} {'Status':>8s}")
    print("-" * 52)
    for r in results:
        status = "PASS" if r["pass_pl"] and r["pass_qk"] else "FAIL"
        print(f"{r['encoding']:10s} {r['qubits']:6d} {r['max_diff_vs_pennylane']:12.2e} {r['max_diff_vs_qiskit']:12.2e} {status:>8s}")
