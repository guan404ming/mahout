//
// Licensed to the Apache Software Foundation (ASF) under one or more
// contributor license agreements.  See the NOTICE file distributed with
// this work for additional information regarding copyright ownership.
// The ASF licenses this file to You under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with
// the License.  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Amplitude Encoding CUDA Kernel

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector_types.h>

/// Paged amplitude encoding kernel
/// Uses page table for state vector access to support arbitrarily large allocations
///
/// @param input        Input data pointer (may be offset for chunked processing)
/// @param page_table   Device array of page pointers
/// @param input_len    Number of input elements to process in this chunk
/// @param state_len    Total state vector size (2^num_qubits)
/// @param state_offset Starting index in state vector for this chunk
/// @param page_size    Elements per page (power of 2)
/// @param page_shift   log2(page_size) for bit-shift division
/// @param inv_norm     1.0 / L2_norm for normalization
__global__ void amplitude_encode_paged_kernel(
    const double* __restrict__ input,
    cuDoubleComplex** __restrict__ page_table,
    size_t input_len,
    size_t state_len,
    size_t state_offset,    // Starting index in state vector for chunked processing
    size_t page_size,       // Elements per page (power of 2)
    unsigned int page_shift, // log2(page_size) for bit-shift division
    double inv_norm
) {
    // We process 2 elements per thread to maximize memory bandwidth via double2
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread handles two state amplitudes (indices 2*idx and 2*idx + 1)
    size_t local_idx = idx * 2;

    if (local_idx >= input_len) return;

    double v1 = 0.0;
    double v2 = 0.0;

    // Vectorized Load Optimization:
    // If we are well within bounds, treat input as double2 to issue a single 128-bit load instruction.
    if (local_idx + 1 < input_len) {
        const double2* input_vec = reinterpret_cast<const double2*>(input);
        double2 loaded = input_vec[idx];
        v1 = loaded.x;
        v2 = loaded.y;
    }
    else if (local_idx < input_len) {
        v1 = input[local_idx];
    }

    // Calculate global state indices (local index + chunk offset)
    size_t global_idx_1 = state_offset + local_idx;
    size_t global_idx_2 = state_offset + local_idx + 1;

    // Page table lookup using bit operations (fast for power-of-2 page sizes)
    size_t page_mask = page_size - 1;

    // First element
    if (global_idx_1 < state_len) {
        size_t page_id_1 = global_idx_1 >> page_shift;
        size_t offset_1 = global_idx_1 & page_mask;
        cuDoubleComplex* page_1 = page_table[page_id_1];
        page_1[offset_1] = make_cuDoubleComplex(v1 * inv_norm, 0.0);
    }

    // Second element (may be on same or next page)
    if (global_idx_2 < state_len) {
        size_t page_id_2 = global_idx_2 >> page_shift;
        size_t offset_2 = global_idx_2 & page_mask;
        cuDoubleComplex* page_2 = page_table[page_id_2];
        page_2[offset_2] = make_cuDoubleComplex(v2 * inv_norm, 0.0);
    }
}

extern "C" {

/// Launch paged amplitude encoding kernel
///
/// # Arguments
/// * input_d - Device pointer to input data
/// * page_table_d - Device pointer to array of page pointers
/// * input_len - Number of input elements to process
/// * state_len - Target state vector size (2^num_qubits)
/// * state_offset - Starting index in state vector (for chunked processing)
/// * page_size - Elements per page (must be power of 2)
/// * page_shift - log2(page_size) for fast division
/// * norm - L2 norm computed by host
/// * stream - CUDA stream for async execution (nullptr = default stream)
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_amplitude_encode(
    const double* input_d,
    void** page_table_d,
    size_t input_len,
    size_t state_len,
    size_t state_offset,
    size_t page_size,
    unsigned int page_shift,
    double norm,
    cudaStream_t stream
) {
    if (norm <= 0.0) {
        return cudaErrorInvalidValue;
    }

    double inv_norm = 1.0 / norm;

    cuDoubleComplex** page_table = reinterpret_cast<cuDoubleComplex**>(page_table_d);

    const int blockSize = 256;
    // Grid size based on input_len (chunk size), not state_len
    const int gridSize = (input_len / 2 + blockSize - 1) / blockSize;

    amplitude_encode_paged_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_d,
        page_table,
        input_len,
        state_len,
        state_offset,
        page_size,
        page_shift,
        inv_norm
    );

    return (int)cudaGetLastError();
}

/// Paged batch amplitude encoding kernel
///
/// Memory Layout (row-major):
/// - input_batch: [sample0_data | sample1_data | ... | sampleN_data]
/// - state pages: distributed across page table
///
/// Optimizations:
/// 1. Vectorized double2 loads for 128-bit memory transactions
/// 2. Grid-stride loop for arbitrary batch sizes
/// 3. Coalesced memory access within warps
/// 4. Bit-shift page lookup for power-of-2 page sizes
__global__ void amplitude_encode_batch_paged_kernel(
    const double* __restrict__ input_batch,
    cuDoubleComplex** __restrict__ page_table,
    const double* __restrict__ inv_norms,
    size_t num_samples,
    size_t input_len,
    size_t state_len,
    size_t page_size,
    unsigned int page_shift
) {
    // Grid-stride loop pattern for flexibility
    const size_t elements_per_sample = state_len / 2;  // Each thread handles 2 elements
    const size_t total_work = num_samples * elements_per_sample;
    const size_t stride = gridDim.x * blockDim.x;
    const size_t page_mask = page_size - 1;
    const size_t total_state_elements = num_samples * state_len;

    size_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process elements in grid-stride fashion
    for (size_t idx = global_idx; idx < total_work; idx += stride) {
        // Decompose linear index into (sample, element_pair)
        const size_t sample_idx = idx / elements_per_sample;
        const size_t elem_pair = idx % elements_per_sample;

        // Calculate base addresses (strength-reduced)
        const size_t input_base = sample_idx * input_len;
        const size_t state_base = sample_idx * state_len;
        const size_t elem_offset = elem_pair * 2;

        // Load inverse norm (cached by L1)
        const double inv_norm = inv_norms[sample_idx];

        // Vectorized load: read 2 doubles as double2 for 128-bit transaction
        double v1, v2;
        if (elem_offset + 1 < input_len) {
            // Aligned vectorized load
            const double2 vec_data = __ldg(reinterpret_cast<const double2*>(input_batch + input_base) + elem_pair);
            v1 = vec_data.x;
            v2 = vec_data.y;
        } else if (elem_offset < input_len) {
            // Edge case: single element load
            v1 = __ldg(input_batch + input_base + elem_offset);
            v2 = 0.0;
        } else {
            // Padding region
            v1 = v2 = 0.0;
        }

        // Normalize
        const cuDoubleComplex c1 = make_cuDoubleComplex(v1 * inv_norm, 0.0);
        const cuDoubleComplex c2 = make_cuDoubleComplex(v2 * inv_norm, 0.0);

        // Page table lookup for first element
        size_t global_idx_1 = state_base + elem_offset;
        size_t page_id_1 = global_idx_1 >> page_shift;
        size_t offset_1 = global_idx_1 & page_mask;
        page_table[page_id_1][offset_1] = c1;

        // Page table lookup for second element
        size_t global_idx_2 = state_base + elem_offset + 1;
        if (global_idx_2 < total_state_elements) {
            size_t page_id_2 = global_idx_2 >> page_shift;
            size_t offset_2 = global_idx_2 & page_mask;
            page_table[page_id_2][offset_2] = c2;
        }
    }
}

/// Launch paged batch amplitude encoding kernel
///
/// # Arguments
/// * input_batch_d - Device pointer to batch input data
/// * page_table_d - Device pointer to array of page pointers
/// * inv_norms_d - Device pointer to inverse norms array
/// * num_samples - Number of samples in batch
/// * input_len - Elements per sample
/// * state_len - State vector size per sample (2^num_qubits)
/// * page_size - Elements per page (must be power of 2)
/// * page_shift - log2(page_size) for fast division
/// * stream - CUDA stream for async execution
///
/// # Returns
/// CUDA error code (0 = cudaSuccess)
int launch_amplitude_encode_batch(
    const double* input_batch_d,
    void** page_table_d,
    const double* inv_norms_d,
    size_t num_samples,
    size_t input_len,
    size_t state_len,
    size_t page_size,
    unsigned int page_shift,
    cudaStream_t stream
) {
    if (num_samples == 0 || state_len == 0) {
        return cudaErrorInvalidValue;
    }

    cuDoubleComplex** page_table = reinterpret_cast<cuDoubleComplex**>(page_table_d);

    // Optimal configuration for modern GPUs (SM 7.0+)
    // - Block size: 256 threads (8 warps, good occupancy)
    // - Grid size: Enough blocks to saturate GPU, but not excessive
    const int blockSize = 256;
    const size_t total_work = num_samples * (state_len / 2);

    // Calculate grid size: aim for high occupancy without too many blocks
    // Limit to reasonable number of blocks to avoid scheduler overhead
    const size_t blocks_needed = (total_work + blockSize - 1) / blockSize;
    const size_t max_blocks = 2048;  // Reasonable limit for most GPUs
    const size_t gridSize = (blocks_needed < max_blocks) ? blocks_needed : max_blocks;

    amplitude_encode_batch_paged_kernel<<<gridSize, blockSize, 0, stream>>>(
        input_batch_d,
        page_table,
        inv_norms_d,
        num_samples,
        input_len,
        state_len,
        page_size,
        page_shift
    );

    return (int)cudaGetLastError();
}

// TODO: Future encoding methods:
// - launch_angle_encode (angle encoding)
// - launch_basis_encode (basis encoding)
// - launch_iqp_encode (IQP encoding)

} // extern "C"
