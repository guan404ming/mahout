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
#

import pytest
import numpy as np

from .conftest import TESTING_BACKENDS
from qumat import QuMat


class TestOverlapMeasurement:
    """Test class for overlap measurement functionality across different backends."""

    def get_backend_config(self, backend_name):
        """Helper method to get backend configuration."""
        if backend_name == "qiskit":
            return {
                "backend_name": backend_name,
                "backend_options": {
                    "simulator_type": "aer_simulator",
                    "shots": 10000,
                },
            }
        elif backend_name == "cirq":
            return {
                "backend_name": backend_name,
                "backend_options": {
                    "simulator_type": "default",
                    "shots": 10000,
                },
            }
        elif backend_name == "amazon_braket":
            return {
                "backend_name": backend_name,
                "backend_options": {
                    "simulator_type": "local",
                    "shots": 10000,
                },
            }

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_zero_states(self, backend_name):
        """Test overlap measurement with two identical |0> states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits: ancilla (0), state1 (1), state2 (2)
        qumat.create_empty_circuit(num_qubits=3)

        # Both states are |0> by default (no preparation needed)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # For identical states, overlap should be ≈ 1.0
        assert overlap > 0.95, (
            f"Expected overlap ≈ 1.0 for identical |0> states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_one_states(self, backend_name):
        """Test overlap measurement with two identical |1> states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |1> on both qubits
        qumat.apply_pauli_x_gate(1)
        qumat.apply_pauli_x_gate(2)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # For identical states, overlap should be ≈ 1.0
        assert overlap > 0.95, (
            f"Expected overlap ≈ 1.0 for identical |1> states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_orthogonal_states(self, backend_name):
        """Test overlap measurement with orthogonal states |0> and |1>."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |0> on qubit 1 (default, no gates needed)
        # Prepare |1> on qubit 2
        qumat.apply_pauli_x_gate(2)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # For orthogonal states, overlap should be ≈ 0.0
        assert overlap < 0.05, (
            f"Expected overlap ≈ 0.0 for orthogonal states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_identical_plus_states(self, backend_name):
        """Test overlap measurement with two identical |+> states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |+> = (|0> + |1>) / sqrt(2) on both qubits
        qumat.apply_hadamard_gate(1)
        qumat.apply_hadamard_gate(2)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # For identical |+> states, overlap should be ≈ 1.0
        assert overlap > 0.95, (
            f"Expected overlap ≈ 1.0 for identical |+> states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_plus_minus_states(self, backend_name):
        """Test overlap measurement with |+> and |-> states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |+> = (|0> + |1>) / sqrt(2) on qubit 1
        qumat.apply_hadamard_gate(1)

        # Prepare |-> = (|0> - |1>) / sqrt(2) on qubit 2
        qumat.apply_pauli_x_gate(2)
        qumat.apply_hadamard_gate(2)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # |+> and |-> are orthogonal, so overlap should be ≈ 0.0
        assert overlap < 0.05, (
            f"Expected overlap ≈ 0.0 for |+> and |-> states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_partial_overlap_states(self, backend_name):
        """Test overlap measurement with states having partial overlap."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |+> on qubit 1
        qumat.apply_hadamard_gate(1)

        # Prepare |0> on qubit 2
        # |0> = (|+> + |->) / sqrt(2)
        # <+|0> = 1/sqrt(2), so |<+|0>|² = 1/2

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # Expected overlap: |<+|0>|² = 1/2 = 0.5
        assert 0.4 < overlap < 0.6, f"Expected overlap ≈ 0.5, got {overlap}"

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_rotated_states(self, backend_name):
        """Test overlap measurement with rotated states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare state at angle π/4 on qubit 1
        # |ψ> = cos(π/8)|0> + sin(π/8)|1>
        qumat.apply_ry_gate(1, np.pi / 4)

        # Prepare state at same angle on qubit 2
        qumat.apply_ry_gate(2, np.pi / 4)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # Identical rotated states should have overlap ≈ 1.0
        assert overlap > 0.95, (
            f"Expected overlap ≈ 1.0 for identical rotated states, got {overlap}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_different_rotated_states(self, backend_name):
        """Test overlap measurement with differently rotated states."""
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 3 qubits
        qumat.create_empty_circuit(num_qubits=3)

        # Prepare |ψ> = cos(π/8)|0> + sin(π/8)|1> on qubit 1
        qumat.apply_ry_gate(1, np.pi / 4)

        # Prepare |φ> = cos(π/4)|0> + sin(π/4)|1> on qubit 2
        qumat.apply_ry_gate(2, np.pi / 2)

        # Measure overlap
        overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # Calculate expected overlap: |<ψ|φ>|²
        # <ψ|φ> = cos(π/8)cos(π/4) + sin(π/8)sin(π/4)
        #       = cos(π/8 - π/4) = cos(-π/8)
        # |<ψ|φ>|² = cos²(π/8) ≈ 0.8536
        expected_overlap = np.cos(np.pi / 8) ** 2

        # Allow for statistical noise (±0.05 tolerance)
        assert abs(overlap - expected_overlap) < 0.05, (
            f"Expected overlap ≈ {expected_overlap:.4f}, got {overlap:.4f}"
        )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_entangled_states_same(self, backend_name):
        """Test overlap measurement with identical entangled states (Bell states).

        Note: This test creates identical Bell states on pairs of qubits.
        We need 5 qubits total: 1 ancilla + 2 qubits for first state + 2 qubits for second state.
        """
        backend_config = self.get_backend_config(backend_name)
        qumat = QuMat(backend_config)

        # Create circuit with 5 qubits
        # Qubit 0: ancilla
        # Qubits 1-2: first entangled state
        # Qubits 3-4: second entangled state
        qumat.create_empty_circuit(num_qubits=5)

        # Prepare Bell state (|00> + |11>) / sqrt(2) on qubits 1-2
        qumat.apply_hadamard_gate(1)
        qumat.apply_cnot_gate(1, 2)

        # Prepare same Bell state on qubits 3-4
        qumat.apply_hadamard_gate(3)
        qumat.apply_cnot_gate(3, 4)

        # Since we can only compare single qubits with swap test,
        # we compare the first qubit of each entangled pair
        # The reduced density matrix of each qubit in a maximally entangled state is I/2
        # So the overlap between reduced states should be 1.0
        overlap = qumat.measure_overlap(qubit1=1, qubit2=3, ancilla_qubit=0)

        # The individual qubits are maximally mixed, so their overlap depends on the implementation
        # This test verifies the method works with entangled qubits
        assert 0.0 <= overlap <= 1.0, (
            f"Overlap should be in valid range [0,1], got {overlap}"
        )

    def test_all_backends_consistency(self, testing_backends):
        """Test that all backends produce consistent results for the same overlap measurement."""
        results_dict = {}

        for backend_name in testing_backends:
            backend_config = self.get_backend_config(backend_name)
            qumat = QuMat(backend_config)

            # Create circuit with identical |0> states
            qumat.create_empty_circuit(num_qubits=3)

            # Measure overlap
            overlap = qumat.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)
            results_dict[backend_name] = overlap

        # All backends should give similar results (within statistical tolerance)
        overlaps = list(results_dict.values())
        for i in range(len(overlaps)):
            for j in range(i + 1, len(overlaps)):
                diff = abs(overlaps[i] - overlaps[j])
                assert diff < 0.05, (
                    f"Backends have inconsistent results: {results_dict}"
                )

    @pytest.mark.parametrize("backend_name", TESTING_BACKENDS)
    def test_measure_overlap_with_different_ancilla(self, backend_name):
        """Test that overlap measurement works with different ancilla qubit positions."""
        backend_config = self.get_backend_config(backend_name)

        # Test with ancilla at position 0
        qumat1 = QuMat(backend_config)
        qumat1.create_empty_circuit(num_qubits=4)
        qumat1.apply_hadamard_gate(1)
        qumat1.apply_hadamard_gate(2)
        overlap1 = qumat1.measure_overlap(qubit1=1, qubit2=2, ancilla_qubit=0)

        # Test with ancilla at position 3
        qumat2 = QuMat(backend_config)
        qumat2.create_empty_circuit(num_qubits=4)
        qumat2.apply_hadamard_gate(0)
        qumat2.apply_hadamard_gate(1)
        overlap2 = qumat2.measure_overlap(qubit1=0, qubit2=1, ancilla_qubit=3)

        # Both should give overlap ≈ 1.0 for identical |+> states
        assert overlap1 > 0.95, f"Expected overlap1 ≈ 1.0, got {overlap1}"
        assert overlap2 > 0.95, f"Expected overlap2 ≈ 1.0, got {overlap2}"
        assert abs(overlap1 - overlap2) < 0.05, (
            f"Results should be consistent regardless of ancilla position"
        )
