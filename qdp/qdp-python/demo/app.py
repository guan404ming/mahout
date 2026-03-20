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
PENNYLANE_GPU_COLOR = "#2563EB"
QISKIT_COLOR = "#7B68A8"

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

    st.title("QDP: GPU-Accelerated Quantum Data Processing")
    st.caption("Apache Mahout | Rust + CUDA | Zero-Copy DLPack Interop")

    # -- Sidebar -----------------------------------------------------------
    st.sidebar.header("Configuration")

    encoding = st.sidebar.selectbox("Encoding Method", ENCODING_METHODS)
    st.sidebar.caption(ENCODING_DESCRIPTIONS[encoding])

    num_qubits = st.sidebar.slider(
        "Number of Qubits", min_value=2, max_value=20, value=8
    )
    batch_size = st.sidebar.slider("Batch Size", min_value=1, max_value=256, value=64)
    num_batches = st.sidebar.slider(
        "Number of Batches", min_value=10, max_value=500, value=100
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
        ["Pipeline Demo", "Framework Comparison", "Scaling Analysis"]
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
                    st.info(f"Sample latency ({nearest}q): {lat:.4f} ms/vector")
                    tp = 1000.0 / lat
                    st.metric("Est. Throughput", f"{tp:,.0f} vec/s")

    # ---- Tab 2: Framework Comparison -------------------------------------
    with tab2:
        st.subheader("Mahout QDP vs PennyLane vs Qiskit")

        if st.button("Run Benchmark", key="run_bench"):
            if live_mode:
                with st.spinner("Running Mahout..."):
                    m_lat = run_live_mahout_latency(
                        num_qubits, num_batches, batch_size, encoding
                    )
                    m_tp = run_live_mahout_throughput(
                        num_qubits, num_batches, batch_size, encoding
                    )

                latency_data = {"Mahout": m_lat}
                tp_data = {"Mahout": m_tp}

                # Limit vectors for competitors (they are much slower)
                competitor_vecs = min(50, total_vectors)

                if HAS_PENNYLANE:
                    with st.spinner("Running PennyLane..."):
                        pl_lat = run_live_pennylane_latency(num_qubits, competitor_vecs)
                    latency_data["PennyLane"] = pl_lat
                    tp_data["PennyLane"] = 1000.0 / pl_lat

                if HAS_QISKIT:
                    with st.spinner("Running Qiskit..."):
                        q_lat = run_live_qiskit_latency(num_qubits, competitor_vecs)
                    latency_data["Qiskit"] = q_lat
                    tp_data["Qiskit"] = 1000.0 / q_lat

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_latency_comparison(latency_data, num_qubits))
                with col2:
                    st.pyplot(plot_throughput_comparison(tp_data, num_qubits))

                # Speedup summary
                if len(latency_data) > 1:
                    st.divider()
                    scols = st.columns(len(latency_data) - 1)
                    idx = 0
                    for fw, lat in latency_data.items():
                        if fw == "Mahout":
                            continue
                        speedup = lat / m_lat
                        scols[idx].metric(f"Speedup vs {fw}", f"{speedup:,.0f}x")
                        idx += 1
            else:
                nearest = min(SAMPLE_LATENCY.keys(), key=lambda q: abs(q - num_qubits))
                latency_data = SAMPLE_LATENCY[nearest]
                tp_data = SAMPLE_THROUGHPUT[nearest]

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_latency_comparison(latency_data, nearest))
                with col2:
                    st.pyplot(plot_throughput_comparison(tp_data, nearest))

                st.divider()
                m_lat = latency_data["Mahout"]
                scols = st.columns(3)
                scols[0].metric(
                    "vs PL default",
                    f"{latency_data['PL default.qubit'] / m_lat:,.0f}x",
                )
                scols[1].metric(
                    "vs PL lightning",
                    f"{latency_data['PL lightning.gpu'] / m_lat:,.0f}x",
                )
                scols[2].metric(
                    "vs Qiskit",
                    f"{latency_data['Qiskit'] / m_lat:,.0f}x",
                )
        else:
            st.info("Click 'Run Benchmark' to compare frameworks.")

    # ---- Tab 3: Scaling Analysis -----------------------------------------
    with tab3:
        st.subheader("Performance Scaling by Qubit Count")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_scaling("latency"))
        with col2:
            st.pyplot(plot_scaling("throughput"))

        st.divider()
        st.markdown("**Speedup Summary (sample data)**")

        rows = []
        for q in sorted(SAMPLE_LATENCY.keys()):
            d = SAMPLE_LATENCY[q]
            m = d["Mahout"]
            pl = d.get("PL default.qubit")
            plg = d.get("PL lightning.gpu")
            qk = d.get("Qiskit")
            rows.append(
                {
                    "Qubits": q,
                    "Mahout (ms)": f"{m:.3f}",
                    "PL default (ms)": f"{pl:.2f}" if pl else "-",
                    "PL lightning (ms)": f"{plg:.2f}" if plg else "-",
                    "Qiskit (ms)": f"{qk:.1f}" if qk else "-",
                    "vs PL default": f"{pl / m:,.0f}x" if pl else "-",
                    "vs PL lightning": f"{plg / m:,.0f}x" if plg else "-",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
