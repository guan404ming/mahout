#!/usr/bin/env python3
"""Minimal debug: compare QDP vs numpy for a tiny case."""

import modal

app = modal.App("qdp-debug")

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
    .pip_install("numpy>=1.24,<2.0", "maturin[patchelf]", "torch", "pennylane>=0.35")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        'export PATH="$HOME/.cargo/bin:$PATH" && git clone --depth 1 https://github.com/apache/mahout.git /opt/mahout',
        'export PATH="$HOME/.cargo/bin:/usr/local/cuda-12.5/bin:$PATH" && export CUDA_PATH=/usr/local/cuda-12.5 && cd /opt/mahout/qdp/qdp-python && maturin build --release',
        'python -c "import glob,subprocess; whl=glob.glob(\'/opt/mahout/qdp/target/wheels/qumat_qdp-*.whl\')[0]; subprocess.run([\'pip\',\'install\',whl],check=True)"',
    )
)


@app.function(image=gpu_image, gpu="A100", timeout=300)
def run():
    import numpy as np
    import torch
    from qumat_qdp import QdpEngine
    import pennylane as qml

    # 2-qubit amplitude: input [1, 2, 3, 4]
    data = [1.0, 2.0, 3.0, 4.0]
    nq = 2

    # QDP float64
    engine64 = QdpEngine(0, precision="float64")
    qt64 = engine64.encode(data, nq, "amplitude")
    qdp_out64 = torch.from_dlpack(qt64).cpu().numpy()
    print(f"QDP f64 output: {qdp_out64}")
    print(f"QDP f64 shape: {qdp_out64.shape}, dtype: {qdp_out64.dtype}")
    print(f"QDP f64 probs: {np.abs(qdp_out64.flatten())**2}")

    # QDP float32
    engine32 = QdpEngine(0, precision="float32")
    qt32 = engine32.encode(data, nq, "amplitude")
    qdp_out32 = torch.from_dlpack(qt32).cpu().numpy()
    print(f"\nQDP f32 output: {qdp_out32}")
    print(f"QDP f32 shape: {qdp_out32.shape}, dtype: {qdp_out32.dtype}")

    # Expected: normalize [1,2,3,4]
    d = np.array(data)
    norm = np.linalg.norm(d)
    expected = d / norm
    print(f"\nExpected (numpy): {expected}")
    print(f"Expected probs: {expected**2}")

    # PennyLane
    dev = qml.device("default.qubit", wires=nq)
    @qml.qnode(dev)
    def circuit(x):
        qml.AmplitudeEmbedding(features=x, wires=range(nq), normalize=True)
        return qml.state()
    pl_out = np.array(circuit(data))
    print(f"\nPennyLane output: {pl_out}")
    print(f"PennyLane probs: {np.abs(pl_out)**2}")

    # Diff
    qdp_flat = qdp_out64.flatten()
    print(f"\nDiff QDP vs Expected: {np.max(np.abs(np.abs(qdp_flat)**2 - expected**2))}")
    print(f"Diff QDP vs PennyLane: {np.max(np.abs(np.abs(qdp_flat)**2 - np.abs(pl_out)**2))}")
    print(f"Diff Expected vs PennyLane: {np.max(np.abs(expected**2 - np.abs(pl_out)**2))}")

    # ANGLE: 2 qubit, angles [0.5, 1.0]
    print("\n" + "="*50)
    print("ANGLE DEBUG")
    angles = [0.5, 1.0]
    qt_angle = engine64.encode(angles, 2, "angle")
    qdp_angle = torch.from_dlpack(qt_angle).cpu().numpy().flatten()
    print(f"QDP angle output: {qdp_angle}")

    # QDP kernel: amplitude = prod( cos or sin )
    # idx=0 (00): cos(0.5)*cos(1.0)
    # idx=1 (01): sin(0.5)*cos(1.0)  (bit0=1)
    # idx=2 (10): cos(0.5)*sin(1.0)  (bit1=1)
    # idx=3 (11): sin(0.5)*sin(1.0)
    manual = np.array([
        np.cos(0.5)*np.cos(1.0),
        np.sin(0.5)*np.cos(1.0),
        np.cos(0.5)*np.sin(1.0),
        np.sin(0.5)*np.sin(1.0),
    ])
    print(f"Manual (cos/sin): {manual}")
    print(f"Diff QDP vs manual: {np.max(np.abs(qdp_angle.real - manual))}")

    # PennyLane AngleEmbedding
    dev2 = qml.device("default.qubit", wires=2)
    @qml.qnode(dev2)
    def angle_circuit(x):
        qml.AngleEmbedding(features=x, wires=range(2))
        return qml.state()
    pl_angle = np.array(angle_circuit(angles))
    print(f"PennyLane angle output: {pl_angle}")

    # PennyLane with 2*angle
    pl_angle2 = np.array(angle_circuit([2*a for a in angles]))
    print(f"PennyLane angle(2x) output: {pl_angle2}")
    print(f"Diff QDP vs PL(2x): {np.max(np.abs(np.abs(qdp_angle)**2 - np.abs(pl_angle2)**2))}")

    # BASIS: 2 qubit, index=3
    print("\n" + "="*50)
    print("BASIS DEBUG")
    qt_basis = engine64.encode([3.0], 2, "basis")
    qdp_basis = torch.from_dlpack(qt_basis).cpu().numpy().flatten()
    print(f"QDP basis(3) output: {qdp_basis}")
    print(f"Expected: [0,0,0,1] at index 3")

    @qml.qnode(dev2)
    def basis_circuit(x):
        qml.BasisEmbedding(features=x, wires=range(2))
        return qml.state()
    # PennyLane: bits for index 3 = [1,1] (both qubits)
    pl_basis_11 = np.array(basis_circuit([1, 1]))
    print(f"PennyLane basis([1,1]): {pl_basis_11}")
    # Try reversed
    pl_basis_11r = np.array(basis_circuit([1, 1]))
    print(f"Diff QDP vs PL([1,1]): {np.max(np.abs(np.abs(qdp_basis)**2 - np.abs(pl_basis_11)**2))}")

    return "done"


@app.local_entrypoint()
def main():
    run.remote()
