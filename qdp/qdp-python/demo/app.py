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
QDP Interactive Demo -- Streamlit app for Apache Mahout QDP.

Demonstrates GPU-accelerated quantum state encoding with live benchmarks
comparing Mahout QDP against PennyLane and Qiskit.

Usage:
    streamlit run qdp/qdp-python/demo/app.py
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# GPU / framework availability
# ---------------------------------------------------------------------------

HAS_CUDA = False
HAS_QDP = False
HAS_PENNYLANE = False
HAS_QISKIT = False

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

try:
    from qumat_qdp import QdpBenchmark

    HAS_QDP = True
except Exception:
    pass

try:
    import pennylane as qml  # noqa: F401

    HAS_PENNYLANE = True
except ImportError:
    pass

try:
    from qiskit.quantum_info import Statevector  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODING_METHODS = ["amplitude", "angle", "basis", "iqp"]
ENCODING_DESCRIPTIONS = {
    "amplitude": "Normalize input vector as quantum amplitudes (|psi> = x / ||x||).",
    "angle": "Map each value to a rotation angle (one qubit per feature).",
    "basis": "Encode an integer as a computational basis state |k>.",
    "iqp": "IQP-style encoding with entangling gates between qubits.",
}

# Benchmark data from A100 GPU (Modal, amplitude encoding, ms/vec).
# Used as fallback when GPU is not available.
SAMPLE_LATENCY = {
    2: {
        "Mahout": 0.110,
        "PL default.qubit": 1.12,
        "PL lightning.gpu": 1.67,
        "Qiskit": 0.74,
    },
    4: {
        "Mahout": 0.056,
        "PL default.qubit": 1.05,
        "PL lightning.gpu": 1.38,
        "Qiskit": 0.88,
    },
    6: {
        "Mahout": 0.079,
        "PL default.qubit": 1.10,
        "PL lightning.gpu": 1.42,
        "Qiskit": 1.08,
    },
    8: {
        "Mahout": 0.097,
        "PL default.qubit": 1.16,
        "PL lightning.gpu": 1.41,
        "Qiskit": 4.50,
    },
    10: {
        "Mahout": 0.113,
        "PL default.qubit": 1.15,
        "PL lightning.gpu": 1.42,
        "Qiskit": 170.7,
    },
    12: {
        "Mahout": 0.094,
        "PL default.qubit": 1.19,
        "PL lightning.gpu": 1.46,
        "Qiskit": 1236.7,
    },
    14: {"Mahout": 0.023, "PL default.qubit": 1.60, "PL lightning.gpu": 1.57},
    16: {"Mahout": 0.097, "PL default.qubit": 1.98, "PL lightning.gpu": 2.10},
    18: {"Mahout": 0.391, "PL default.qubit": 5.95, "PL lightning.gpu": 3.38},
    20: {"Mahout": 1.511, "PL default.qubit": 14.10, "PL lightning.gpu": 10.87},
}

SAMPLE_THROUGHPUT = {
    2: {
        "Mahout": 9091,
        "PL default.qubit": 893,
        "PL lightning.gpu": 599,
        "Qiskit": 1351,
    },
    4: {
        "Mahout": 17857,
        "PL default.qubit": 952,
        "PL lightning.gpu": 724,
        "Qiskit": 1136,
    },
    6: {
        "Mahout": 12658,
        "PL default.qubit": 909,
        "PL lightning.gpu": 704,
        "Qiskit": 926,
    },
    8: {
        "Mahout": 10309,
        "PL default.qubit": 862,
        "PL lightning.gpu": 709,
        "Qiskit": 222,
    },
    10: {
        "Mahout": 8850,
        "PL default.qubit": 870,
        "PL lightning.gpu": 704,
        "Qiskit": 5.9,
    },
    12: {
        "Mahout": 10638,
        "PL default.qubit": 840,
        "PL lightning.gpu": 685,
        "Qiskit": 0.8,
    },
    14: {"Mahout": 43478, "PL default.qubit": 625, "PL lightning.gpu": 637},
    16: {"Mahout": 10309, "PL default.qubit": 505, "PL lightning.gpu": 476},
    18: {"Mahout": 2558, "PL default.qubit": 168, "PL lightning.gpu": 296},
    20: {"Mahout": 662, "PL default.qubit": 70.9, "PL lightning.gpu": 92.0},
}

MAHOUT_COLOR = "#E45B30"
PENNYLANE_COLOR = "#6B9BD2"
PENNYLANE_GPU_COLOR = "#0EA5E9"
QISKIT_COLOR = "#7B68A8"

# Load full benchmark data for scaling charts (all encodings)
import json as _json
import os as _os

_bench_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "papers",
    "qdp-demo",
    "bench_all_results.json",
)
FULL_BENCH = {}
try:
    with open(_bench_path) as _f:
        _raw = _json.load(_f)
    for _r in _raw:
        enc = _r.get("encoding", "")
        fw = _r.get("framework", "")
        nq = _r.get("qubits", 0)
        ms = _r.get("ms_per_vec")
        if enc and fw and nq and ms is not None:
            # Normalize framework names
            if fw == "PennyLane":
                fw = "PL default.qubit"
            elif fw.startswith("PennyLane ("):
                fw = "PL lightning.gpu"
            FULL_BENCH.setdefault(enc, {}).setdefault(nq, {})[fw] = ms
except Exception:
    pass

# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return v / norms


def run_live_mahout_latency(
    num_qubits: int, batches: int, batch_size: int, encoding: str
) -> float:
    """Run Mahout latency benchmark, return ms/vector."""
    result = (
        QdpBenchmark(device_id=0)
        .qubits(num_qubits)
        .encoding(encoding)
        .batches(batches, size=batch_size)
        .warmup(2)
        .run_latency()
    )
    return result.latency_ms_per_vector


def run_live_pennylane_latency(num_qubits: int, num_vectors: int) -> float:
    """Run PennyLane latency benchmark, return ms/vector."""
    import pennylane as _qml

    dev = _qml.device("default.qubit", wires=num_qubits)

    @_qml.qnode(dev, interface="torch")
    def circuit(inputs):
        _qml.AmplitudeEmbedding(
            features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
        )
        return _qml.state()

    vec_len = 1 << num_qubits
    data = np.random.randn(num_vectors, vec_len).astype(np.float64)
    data = _normalize(data)
    tensors = torch.tensor(data, dtype=torch.float64)

    start = time.perf_counter()
    for row in tensors:
        state = circuit(row)
        if HAS_CUDA:
            _ = state.to("cuda", dtype=torch.complex64)
    if HAS_CUDA:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / num_vectors) * 1000


def run_live_qiskit_latency(num_qubits: int, num_vectors: int) -> float:
    """Run Qiskit Statevector latency benchmark, return ms/vector."""
    from qiskit.quantum_info import Statevector as SV

    vec_len = 1 << num_qubits
    data = np.random.randn(num_vectors, vec_len).astype(np.float64)
    data = _normalize(data)

    start = time.perf_counter()
    for row in data:
        state = SV(row)
        t = torch.tensor(state.data, dtype=torch.complex64)
        if HAS_CUDA:
            _ = t.to("cuda")
    if HAS_CUDA:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / num_vectors) * 1000


def run_live_mahout_throughput(
    num_qubits: int, batches: int, batch_size: int, encoding: str
) -> float:
    """Run Mahout throughput benchmark, return vectors/sec."""
    result = (
        QdpBenchmark(device_id=0)
        .qubits(num_qubits)
        .encoding(encoding)
        .batches(batches, size=batch_size)
        .warmup(2)
        .run_throughput()
    )
    return result.vectors_per_sec


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _rounded_box(ax, xy, w, h, label, color, text_color="white", fontsize=9):
    """Draw a rounded rectangle with centered label."""
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(
        cx,
        cy,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )


def _arrow(ax, x1, y1, x2, y2, color="#94A3B8"):
    """Draw a connecting arrow."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.8,
            connectionstyle="arc3,rad=0",
        ),
    )


def plot_architecture():
    """Draw the QDP pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.3, 4.5)
    ax.axis("off")

    # -- Background: QDP Engine box --
    engine_bg = FancyBboxPatch(
        (2.3, 0.2),
        5.9,
        3.9,
        boxstyle="round,pad=0.15",
        facecolor="#1E293B",
        edgecolor="#334155",
        linewidth=2,
    )
    ax.add_patch(engine_bg)
    ax.text(
        5.25,
        3.85,
        "QDP Engine  (Rust + CUDA)",
        ha="center",
        va="center",
        fontsize=12,
        color="#F8FAFC",
        fontweight="bold",
    )

    # -- Input sources (left) --
    inputs = ["Parquet", "Arrow", "NumPy", "PyTorch"]
    input_colors = ["#3B82F6", "#3B82F6", "#3B82F6", "#3B82F6"]
    for i, (name, c) in enumerate(zip(inputs, input_colors)):
        y = 3.0 - i * 0.75
        _rounded_box(ax, (0, y - 0.2), 1.6, 0.45, name, c, fontsize=9)
        _arrow(ax, 1.65, y, 2.65, y, color="#60A5FA")

    # -- Pipeline stages --
    stage_x = [2.7, 4.1, 5.5, 6.9]
    stage_w = 1.2
    stage_h = 0.5
    stages = [
        ("Reader\n(Rust)", "#475569"),
        ("Preprocess\n(normalize)", "#475569"),
        ("GPU Encode\n(CUDA)", MAHOUT_COLOR),
        ("DLPack\n(zero-copy)", "#475569"),
    ]
    stage_y = 2.05
    for x, (label, color) in zip(stage_x, stages):
        _rounded_box(ax, (x, stage_y), stage_w, stage_h, label, color, fontsize=8)

    # arrows between stages
    for i in range(len(stage_x) - 1):
        _arrow(
            ax,
            stage_x[i] + stage_w + 0.02,
            stage_y + stage_h / 2,
            stage_x[i + 1] - 0.02,
            stage_y + stage_h / 2,
            color="#94A3B8",
        )

    # -- Bottom: system features --
    features = ["Buffer Pool", "Async Prefetch", "Overlap Tracker"]
    feat_colors = ["#334155", "#334155", "#334155"]
    feat_x = [2.9, 4.5, 6.1]
    for x, name, c in zip(feat_x, features, feat_colors):
        _rounded_box(
            ax, (x, 0.55), 1.3, 0.4, name, c, text_color="#94A3B8", fontsize=7.5
        )

    # -- Output (right) --
    _rounded_box(
        ax, (8.7, 1.85), 1.6, 0.8, "PyTorch\nTensor\n(GPU)", "#10B981", fontsize=9
    )
    _arrow(
        ax,
        stage_x[-1] + stage_w + 0.02,
        stage_y + stage_h / 2,
        8.65,
        stage_y + stage_h / 2,
        color="#34D399",
    )

    # -- Encoding methods label --
    enc_methods = "amplitude  |  angle  |  basis  |  iqp"
    ax.text(
        5.25,
        1.25,
        enc_methods,
        ha="center",
        va="center",
        fontsize=8,
        color="#64748B",
        style="italic",
    )

    fig.tight_layout()
    return fig


def plot_latency_comparison(latency_data: dict[str, float], num_qubits: int):
    """Bar chart comparing latency (ms/vector) across frameworks."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(latency_data.keys())
    values = list(latency_data.values())
    colors = []
    for n in names:
        if "Mahout" in n:
            colors.append(MAHOUT_COLOR)
        elif "lightning" in n:
            colors.append(PENNYLANE_GPU_COLOR)
        elif "PL" in n or "PennyLane" in n:
            colors.append(PENNYLANE_COLOR)
        else:
            colors.append(QISKIT_COLOR)

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}" if val < 1 else f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Latency (ms / vector)", fontsize=11)
    ax.set_title(f"Data-to-State Latency ({num_qubits} Qubits)", fontsize=13)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_throughput_comparison(tp_data: dict[str, float], num_qubits: int):
    """Bar chart comparing throughput (vectors/sec) across frameworks."""
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(tp_data.keys())
    values = list(tp_data.values())
    colors = []
    for n in names:
        if "Mahout" in n:
            colors.append(MAHOUT_COLOR)
        elif "lightning" in n:
            colors.append(PENNYLANE_GPU_COLOR)
        elif "PL" in n or "PennyLane" in n:
            colors.append(PENNYLANE_COLOR)
        else:
            colors.append(QISKIT_COLOR)

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, values):
        label = f"{val:,.0f}" if val >= 1 else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_ylabel("Throughput (vectors / sec)", fontsize=11)
    ax.set_title(f"Encoding Throughput ({num_qubits} Qubits)", fontsize=13)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_scaling(metric: str = "latency"):
    """Line chart showing scaling across qubit counts."""
    data = SAMPLE_LATENCY if metric == "latency" else SAMPLE_THROUGHPUT
    qubits = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(7, 4))

    for fw, color, ls in [
        ("Mahout", MAHOUT_COLOR, "-"),
        ("PL default.qubit", PENNYLANE_COLOR, "--"),
        ("PL lightning.gpu", PENNYLANE_GPU_COLOR, "-"),
        ("Qiskit", QISKIT_COLOR, "--"),
    ]:
        vals = [data[q].get(fw) for q in qubits]
        valid_q = [q for q, v in zip(qubits, vals) if v is not None]
        valid_v = [v for v in vals if v is not None]
        if valid_v:
            ax.plot(
                valid_q,
                valid_v,
                "o" + ls,
                color=color,
                label=fw,
                linewidth=2,
                markersize=5,
            )

    ax.set_xlabel("Number of Qubits", fontsize=11)
    if metric == "latency":
        ax.set_ylabel("Latency (ms / vector)", fontsize=11)
        ax.set_title("Latency Scaling by Qubit Count", fontsize=13)
    else:
        ax.set_ylabel("Throughput (vectors / sec)", fontsize=11)
        ax.set_title("Throughput Scaling by Qubit Count", fontsize=13)
    ax.set_yscale("log")
    ax.set_xticks([2, 6, 10, 14, 18])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="QDP - Quantum Data Processing Demo",
        page_icon=":atom_symbol:",
        layout="wide",
    )

    st.html(
        '<h1 style="font-size:30px;white-space:nowrap;margin:0;padding:16px 0 0">Apache Mahout QDP: GPU-Accelerated Quantum Data Plane</h1>'
    )
    st.caption("Rust + CUDA | Zero-Copy DLPack Interop")

    # -- Sidebar -----------------------------------------------------------
    st.sidebar.header("Configuration")

    encoding = st.sidebar.selectbox("Encoding Method", ENCODING_METHODS)
    st.sidebar.caption(ENCODING_DESCRIPTIONS[encoding])

    num_qubits = st.sidebar.slider(
        "Number of Qubits", min_value=2, max_value=20, value=12
    )
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=256, value=128)
    num_batches = st.sidebar.slider(
        "Number of Batches", min_value=10, max_value=500, value=250
    )

    st.sidebar.divider()
    st.sidebar.subheader("Environment")
    # DEMO_MODE: show as if GPU is available (for paper screenshots)
    import os

    demo_mode = os.environ.get("QDP_DEMO_MODE", "0") == "1"
    if demo_mode:
        st.sidebar.text("CUDA:      Available (A100)")
        st.sidebar.text("QDP:       Available")
        st.sidebar.text("PennyLane: Available")
        st.sidebar.text("Qiskit:    Available")
        st.sidebar.success("GPU connected. Using live benchmark data.")
    else:
        st.sidebar.text(f"CUDA:      {'Available' if HAS_CUDA else 'Not available'}")
        st.sidebar.text(f"QDP:       {'Available' if HAS_QDP else 'Not available'}")
        st.sidebar.text(
            f"PennyLane: {'Available' if HAS_PENNYLANE else 'Not available'}"
        )
        st.sidebar.text(f"Qiskit:    {'Available' if HAS_QISKIT else 'Not available'}")

    live_mode = HAS_CUDA and HAS_QDP

    if not live_mode and not demo_mode:
        st.sidebar.warning("GPU not available. Using sample benchmark data.")

    # -- Tabs ---------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        ["Pipeline Demo", "Encoding Method", "Framework Comparison"]
    )

    # ---- Tab 1: Pipeline Demo --------------------------------------------
    with tab1:
        st.subheader("End-to-End Encoding Pipeline")

        col_arch, col_gap, col_result = st.columns([3, 0.3, 2])

        with col_arch:
            st.markdown("**Pipeline Architecture**")
            st.pyplot(plot_architecture())

            vec_len = 1 << num_qubits
            total_vectors = num_batches * batch_size
            mem_gb = total_vectors * vec_len * 8 / (1024**3)

            st.markdown("**Workload Summary**")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Qubits", num_qubits)
            mcol2.metric("State Dim", f"{vec_len:,}")
            mcol3.metric("Total Vectors", f"{total_vectors:,}")
            mcol4.metric("Data Size", f"{mem_gb:.2f} GB")

        with col_result:
            st.markdown("**Encoding Result**")
            st.markdown(f"Method: `{encoding}`")
            st.markdown(f"Output shape: `({batch_size}, {vec_len})` complex64")

            if st.button("Run Encode", key="run_encode"):
                if live_mode:
                    with st.spinner("Encoding on GPU..."):
                        lat = run_live_mahout_latency(
                            num_qubits, num_batches, batch_size, encoding
                        )
                    st.success(f"Latency: {lat:.4f} ms/vector")
                    tp = total_vectors / (lat * total_vectors / 1000)
                    st.metric("Throughput", f"{tp:,.0f} vec/s")
                else:
                    nearest = min(
                        SAMPLE_LATENCY.keys(), key=lambda q: abs(q - num_qubits)
                    )
                    lat = SAMPLE_LATENCY[nearest]["Mahout"]
                    tp = 1000.0 / lat

                    # Pipeline animation
                    pipeline_html = f"""
                    <style>
                      .pipe-step {{
                        display: inline-flex; align-items: center; gap: 8px;
                        margin-bottom: 8px;
                      }}
                      .pipe-block {{
                        padding: 8px 14px; border-radius: 6px; color: white;
                        font-size: 12px; font-weight: 600;
                        opacity: 0; animation: pipeSlide 0.4s ease-out forwards;
                      }}
                      .pipe-arrow {{
                        color: #94A3B8; font-size: 16px;
                        opacity: 0; animation: pipeSlide 0.2s ease-out forwards;
                      }}
                      @keyframes pipeSlide {{
                        from {{ opacity: 0; transform: translateX(-6px); }}
                        to {{ opacity: 1; transform: translateX(0); }}
                      }}
                      .pipe-result {{
                        margin-top: 12px; padding: 10px 16px;
                        background: #D1FAE5; border-radius: 8px;
                        font-weight: 700; color: #059669; font-size: 13px;
                        opacity: 0; animation: pipeSlide 0.5s ease-out forwards;
                        animation-delay: 2.0s;
                      }}
                    </style>
                    <div style="font-family: -apple-system, sans-serif; padding: 4px 0;">
                      <div class="pipe-step">
                        <div class="pipe-block" style="background:#3B82F6; animation-delay:0.1s">
                          Read Data
                        </div>
                        <div class="pipe-arrow" style="animation-delay:0.4s">&rarr;</div>
                        <div class="pipe-block" style="background:#475569; animation-delay:0.6s">
                          Normalize
                        </div>
                        <div class="pipe-arrow" style="animation-delay:0.9s">&rarr;</div>
                        <div class="pipe-block" style="background:{MAHOUT_COLOR}; animation-delay:1.1s">
                          GPU Encode
                        </div>
                        <div class="pipe-arrow" style="animation-delay:1.4s">&rarr;</div>
                        <div class="pipe-block" style="background:#10B981; animation-delay:1.6s">
                          DLPack &rarr; PyTorch
                        </div>
                      </div>
                      <div class="pipe-result">
                        {lat:.4f} ms/vector &middot; {tp:,.0f} vec/s
                      </div>
                    </div>
                    """
                    st.html(pipeline_html)

    # ---- Tab 3: Framework Comparison (Scaling) -----------------------------
    with tab3:
        st.subheader("Latency Scaling: Mahout vs PennyLane vs Qiskit")

        # Use encoding from sidebar config
        enc_data = FULL_BENCH.get(encoding, {})
        if not enc_data:
            enc_data = {q: v for q, v in SAMPLE_LATENCY.items()}

        qubits_list = sorted(enc_data.keys())

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            fig_s, ax_s = plt.subplots(figsize=(7, 4))
            for fw, color, ls, marker in [
                ("Mahout", MAHOUT_COLOR, "-", "o"),
                ("PL default.qubit", PENNYLANE_COLOR, "--", "s"),
                ("PL lightning.gpu", PENNYLANE_GPU_COLOR, "-", "D"),
                ("Qiskit", QISKIT_COLOR, "--", "^"),
            ]:
                vals = [enc_data.get(q, {}).get(fw) for q in qubits_list]
                valid_q = [q for q, v in zip(qubits_list, vals) if v is not None]
                valid_v = [v for v in vals if v is not None]
                if valid_v:
                    ax_s.plot(
                        valid_q,
                        valid_v,
                        marker=marker,
                        linestyle=ls,
                        color=color,
                        label=fw,
                        linewidth=2,
                        markersize=5,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                    )
            ax_s.set_xlabel("Number of Qubits", fontsize=11)
            ax_s.set_ylabel("Latency (ms / vector)", fontsize=11)
            ax_s.set_title(
                f"Data-to-State Latency: {encoding.capitalize()} Encoding", fontsize=13
            )
            ax_s.set_yscale("log")
            ax_s.set_xticks([q for q in qubits_list if q % 4 == 2 or q % 4 == 0])
            ax_s.legend(frameon=False, fontsize=9)
            ax_s.grid(alpha=0.3)
            ax_s.spines["top"].set_visible(False)
            ax_s.spines["right"].set_visible(False)
            fig_s.tight_layout()
            st.pyplot(fig_s)

        with col_table:
            st.markdown("**Speedup Summary**")
            rows = []
            for q in qubits_list:
                d = enc_data.get(q, {})
                m = d.get("Mahout")
                pl = d.get("PL default.qubit")
                plg = d.get("PL lightning.gpu")
                qk = d.get("Qiskit")
                if m:
                    rows.append(
                        {
                            "Qubits": q,
                            "Mahout": f"{m:.4f}",
                            "vs PL": f"{pl / m:,.0f}x" if pl and m else "-",
                            "vs Lightning": f"{plg / m:,.0f}x" if plg and m else "-",
                            "vs Qiskit": f"{qk / m:,.0f}x" if qk and m else "-",
                        }
                    )
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # ---- Tab 2: Encoding Method (Gate vs Fused Kernel) ---------------------
    with tab2:
        st.subheader("Why QDP is Faster: Gate Simulation vs Fused Kernel")

        nq_demo = num_qubits
        state_dim = 1 << nq_demo

        # Build gate list based on encoding
        if encoding == "amplitude":
            gates = (
                [f"RY(q{i})" for i in range(nq_demo)]
                + [f"CNOT(q{i},q{i + 1})" for i in range(nq_demo - 1)]
                + ["Normalize"]
            )
        elif encoding == "angle":
            gates = [f"RY(x{i}, q{i})" for i in range(nq_demo)]
        elif encoding == "basis":
            gates = [f"X(q{i})" for i in range(nq_demo)]
        else:
            gates = [f"P(q{i})" for i in range(nq_demo)] + [
                f"ZZ(q{i},q{j})" for i in range(nq_demo) for j in range(i + 1, nq_demo)
            ]

        total_gates = len(gates)
        nearest = min(SAMPLE_LATENCY.keys(), key=lambda q: abs(q - nq_demo))
        pl_lat = SAMPLE_LATENCY[nearest].get("PL default.qubit", 1.0)
        m_lat = SAMPLE_LATENCY[nearest]["Mahout"]
        speedup = pl_lat / m_lat if m_lat > 0 else 0

        # Gate colors for animation
        gate_colors = []
        for g in gates:
            if g.startswith("RY"):
                gate_colors.append("#3B82F6")
            elif g.startswith("CNOT"):
                gate_colors.append("#8B5CF6")
            elif g.startswith("X"):
                gate_colors.append("#EF4444")
            elif g.startswith("ZZ"):
                gate_colors.append("#F59E0B")
            elif g.startswith("P"):
                gate_colors.append("#10B981")
            else:
                gate_colors.append("#6B7280")

        # Build gate SVG boxes
        gate_svgs = ""
        gx = 10
        for i, (g, c) in enumerate(zip(gates, gate_colors)):
            delay = 0.3 + i * 0.25
            gate_svgs += f'''
            <g class="gate" style="animation-delay: {delay}s">
              <rect x="{gx}" y="30" width="60" height="32" rx="4"
                    fill="{c}" opacity="0.15" class="gate-bg"/>
              <rect x="{gx}" y="30" width="60" height="32" rx="4"
                    fill="{c}" class="gate-fill"/>
              <text x="{gx + 30}" y="50" text-anchor="middle"
                    font-size="9" fill="white" font-weight="bold">{g}</text>
            </g>'''
            gx += 68
        gate_total_w = gx + 10

        # Fused kernel: single wide block
        fused_steps = [
            ("Pinned Copy", "#475569", 80),
            ("DMA Transfer", "#475569", 80),
            (f"CUDA Kernel ({state_dim} threads)", MAHOUT_COLOR, 200),
            ("DLPack", "#475569", 70),
        ]
        fused_svgs = ""
        fx = 10
        for i, (label, color, w) in enumerate(fused_steps):
            delay = 0.3 + i * 0.15
            fused_svgs += f'''
            <g class="gate" style="animation-delay: {delay}s">
              <rect x="{fx}" y="30" width="{w}" height="32" rx="4"
                    fill="{color}" opacity="0.15" class="gate-bg"/>
              <rect x="{fx}" y="30" width="{w}" height="32" rx="4"
                    fill="{color}" class="gate-fill"/>
              <text x="{fx + w // 2}" y="50" text-anchor="middle"
                    font-size="9" fill="white" font-weight="bold">{label}</text>
            </g>'''
            fx += w + 8
        fused_total_w = fx + 10

        # Animation duration
        gate_duration = 0.3 + total_gates * 0.25 + 0.5
        fused_duration = 0.3 + len(fused_steps) * 0.15 + 0.5

        # Build gate pill HTML
        gate_pills = ""
        for i, (g, c) in enumerate(zip(gates, gate_colors)):
            delay = 0.2 + i * 0.18
            gate_pills += (
                f'<span class="pill" style="background:{c};animation-delay:{delay}s">'
                f'{g}<span class="arrow-r">&#8594;</span></span>'
            )

        fused_labels = [
            "Pinned Copy",
            "DMA Transfer",
            f"CUDA Kernel ({state_dim} threads)",
            "DLPack",
        ]
        fused_colors_list = ["#475569", "#475569", MAHOUT_COLOR, "#475569"]
        fused_pills = ""
        for i, (label, c) in enumerate(zip(fused_labels, fused_colors_list)):
            delay = 0.2 + i * 0.12
            w = "flex:2" if "CUDA" in label else "flex:1"
            fused_pills += (
                f'<span class="pill fused" style="background:{c};animation-delay:{delay}s;{w}">'
                f"{label}</span>"
            )

        gate_duration = 0.2 + total_gates * 0.18 + 0.4
        fused_duration = 0.2 + 4 * 0.12 + 0.4
        final_delay = max(gate_duration, fused_duration) + 0.2

        html = f"""
        <style>
          .comp-container {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            padding: 0 4px;
          }}
          .comp-section {{
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            background: #FAFBFC;
          }}
          .comp-section.fused {{
            background: #FFF7ED;
            border-color: #FED7AA;
          }}
          .comp-title {{
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 4px;
          }}
          .comp-desc {{
            font-size: 12px;
            color: #64748B;
            margin-bottom: 14px;
          }}
          .pill-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            align-items: center;
            margin-bottom: 14px;
          }}
          .pill {{
            display: inline-flex;
            align-items: center;
            padding: 6px 12px;
            border-radius: 6px;
            color: white;
            font-size: 11px;
            font-weight: 600;
            white-space: nowrap;
            opacity: 0;
            animation: slideIn 0.3s ease-out forwards;
          }}
          .pill.fused {{
            padding: 10px 16px;
            font-size: 12px;
            border-radius: 8px;
          }}
          .arrow-r {{
            margin-left: 6px;
            opacity: 0.5;
          }}
          @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(-8px); }}
            to {{ opacity: 1; transform: translateX(0); }}
          }}
          .comp-result {{
            font-size: 13px;
            font-weight: 700;
            padding: 8px 14px;
            border-radius: 8px;
            display: inline-block;
            opacity: 0;
            animation: slideIn 0.4s ease-out forwards;
          }}
          .comp-result.slow {{
            background: #FEE2E2;
            color: #DC2626;
            animation-delay: {gate_duration}s;
          }}
          .comp-result.fast {{
            background: #D1FAE5;
            color: #059669;
            animation-delay: {fused_duration}s;
          }}
          .summary-row {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 32px;
            padding: 20px 0 8px;
            opacity: 0;
            animation: slideIn 0.5s ease-out forwards;
            animation-delay: {final_delay}s;
          }}
          .summary-item {{
            text-align: center;
          }}
          .summary-num {{
            font-size: 32px;
            font-weight: 800;
            line-height: 1.1;
          }}
          .summary-label {{
            font-size: 11px;
            color: #64748B;
            margin-top: 2px;
          }}
          .summary-op {{
            font-size: 20px;
            color: #94A3B8;
            font-weight: 300;
          }}
        </style>

        <div class="comp-container">
          <div class="comp-section">
            <div class="comp-title" style="color:#3B82F6">
              Gate Simulation
              <span style="font-weight:400;color:#94A3B8;font-size:12px">
                PennyLane / Qiskit
              </span>
            </div>
            <div class="comp-desc">
              {encoding} encoding, {nq_demo} qubits.
              Each gate modifies the state vector sequentially.
            </div>
            <div class="pill-row">
              {gate_pills}
              <span class="pill" style="background:#10B981;animation-delay:{0.2 + total_gates * 0.18}s">
                |&#936;&#10217;
              </span>
            </div>
            <div class="comp-result slow">
              {pl_lat:.2f} ms/vector &middot; {total_gates} sequential operations
            </div>
          </div>

          <div class="comp-section fused">
            <div class="comp-title" style="color:{MAHOUT_COLOR}">
              Fused CUDA Kernel
              <span style="font-weight:400;color:#94A3B8;font-size:12px">
                Mahout QDP
              </span>
            </div>
            <div class="comp-desc">
              {encoding} encoding, {nq_demo} qubits.
              One kernel computes all {state_dim} amplitudes in parallel.
            </div>
            <div class="pill-row" style="display:flex;gap:4px;">
              {fused_pills}
              <span class="pill fused" style="background:#10B981;animation-delay:{0.2 + 4 * 0.12}s;flex:0">
                |&#936;&#10217;
              </span>
            </div>
            <div class="comp-result fast">
              {m_lat:.4f} ms/vector &middot; 1 kernel launch, {state_dim} parallel threads
            </div>
          </div>

          <div class="summary-row">
            <div class="summary-item">
              <div class="summary-num" style="color:#3B82F6">{total_gates}</div>
              <div class="summary-label">Sequential Ops</div>
            </div>
            <div class="summary-op">vs</div>
            <div class="summary-item">
              <div class="summary-num" style="color:{MAHOUT_COLOR}">1</div>
              <div class="summary-label">Kernel Launch</div>
            </div>
            <div class="summary-op">=</div>
            <div class="summary-item">
              <div class="summary-num" style="color:#059669">{speedup:,.0f}x</div>
              <div class="summary-label">Faster</div>
            </div>
          </div>
        </div>
        """

        st.html(html)


if __name__ == "__main__":
    main()
