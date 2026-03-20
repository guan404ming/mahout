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

"""Generate evaluation figures for the QDP demo paper from benchmark data."""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42

import json
import os

import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTDIR, exist_ok=True)

MAHOUT_COLOR = "#E45B30"
PENNYLANE_COLOR = "#6B9BD2"
PENNYLANE_GPU_COLOR = "#2563EB"
QISKIT_COLOR = "#7B68A8"

# Load all data from single file
with open(os.path.join(os.path.dirname(__file__), "bench_all_results.json")) as f:
    ALL_DATA = json.load(f)

# Split by framework for convenience
RAW = [r for r in ALL_DATA if not r.get("framework", "").startswith("PennyLane (")]
LGPU = [r for r in ALL_DATA if r.get("framework", "").startswith("PennyLane (")]


def get(framework, encoding, qubits, source=RAW):
    for r in source:
        if (
            r.get("framework", "").startswith(framework)
            and r.get("encoding") == encoding
            and r.get("qubits") == qubits
        ):
            return r.get("ms_per_vec")
    return None


def gen_scaling_4line():
    """3-panel line chart with 4 frameworks: Mahout, PL default, PL lightning.gpu, Qiskit."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), sharey=False)

    encodings = ["amplitude", "angle", "basis"]
    titles = ["(a) Amplitude Encoding", "(b) Angle Encoding", "(c) Basis Encoding"]
    qubits = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    frameworks = [
        ("Mahout", RAW, MAHOUT_COLOR, "o", "-"),
        ("PennyLane", RAW, PENNYLANE_COLOR, "s", "--"),
        ("PennyLane", LGPU, PENNYLANE_GPU_COLOR, "D", "-"),
        ("Qiskit", RAW, QISKIT_COLOR, "^", "--"),
    ]
    labels = ["Mahout QDP", "PL default.qubit", "PL lightning.gpu", "Qiskit initialize"]

    for ax, enc, title in zip(axes, encodings, titles):
        for (fw, src, color, marker, ls), label in zip(frameworks, labels):
            vals = [get(fw, enc, q, src) for q in qubits]
            valid_q = [q for q, v in zip(qubits, vals) if v is not None]
            valid_v = [v for v in vals if v is not None]
            if valid_v:
                ax.plot(
                    valid_q,
                    valid_v,
                    marker=marker,
                    linestyle=ls,
                    color=color,
                    label=label,
                    linewidth=1.5,
                    markersize=3.5,
                    markeredgecolor="white",
                    markeredgewidth=0.3,
                )

        ax.set_xlabel("Qubits", fontsize=8)
        ax.set_title(title, fontsize=9, pad=6)
        ax.set_yscale("log")
        ax.set_xticks([2, 6, 10, 14, 18])
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("Latency (ms / vector)", fontsize=8)
    axes[0].legend(fontsize=5.5, frameon=False, loc="upper left")

    fig.tight_layout(pad=0.8)
    fig.savefig(os.path.join(OUTDIR, "eval_scaling.pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(OUTDIR, "eval_scaling.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("  -> eval_scaling.pdf")


if __name__ == "__main__":
    print("Generating eval figures...")
    gen_scaling_4line()
    print("Done.")
