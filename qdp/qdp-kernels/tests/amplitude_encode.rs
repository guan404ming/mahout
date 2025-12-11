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

// Tests for amplitude encoding CUDA kernel (paged memory version)

#[cfg(target_os = "linux")]
use cudarc::driver::{CudaDevice, DevicePtr, DevicePtrMut};
#[cfg(target_os = "linux")]
use qdp_kernels::{CuDoubleComplex, launch_amplitude_encode};

const EPSILON: f64 = 1e-10;

/// Helper to create a single-page page table for testing
/// Returns (state buffer, page table buffer)
#[cfg(target_os = "linux")]
fn create_single_page_state(
    device: &std::sync::Arc<CudaDevice>,
    state_len: usize,
) -> (
    cudarc::driver::CudaSlice<CuDoubleComplex>,
    cudarc::driver::CudaSlice<u64>,
) {
    // Allocate state buffer
    let state_d = device.alloc_zeros::<CuDoubleComplex>(state_len).unwrap();

    // Get pointer and create page table with single entry
    let state_ptr = *state_d.device_ptr() as u64;
    let page_table = device.htod_copy(vec![state_ptr]).unwrap();

    (state_d, page_table)
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_basic() {
    println!("Testing basic amplitude encoding...");

    // Initialize CUDA device
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test input: [3.0, 4.0] -> normalized to [0.6, 0.8]
    let input = vec![3.0, 4.0];
    let norm = (3.0_f64.powi(2) + 4.0_f64.powi(2)).sqrt(); // 5.0
    let state_len = 4; // 2 qubits

    // Allocate device memory
    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    // Launch kernel with paged API
    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0, // state_offset
            state_len, // page_size (single page covers all)
            (state_len as f64).log2() as u32, // page_shift
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    // Copy result back
    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify normalization: [0.6, 0.8, 0.0, 0.0]
    assert!(
        (state_h[0].x - 0.6).abs() < EPSILON,
        "First element should be 0.6"
    );
    assert!(
        (state_h[0].y).abs() < EPSILON,
        "First element imaginary should be 0"
    );
    assert!(
        (state_h[1].x - 0.8).abs() < EPSILON,
        "Second element should be 0.8"
    );
    assert!(
        (state_h[1].y).abs() < EPSILON,
        "Second element imaginary should be 0"
    );
    assert!((state_h[2].x).abs() < EPSILON, "Third element should be 0");
    assert!((state_h[3].x).abs() < EPSILON, "Fourth element should be 0");

    // Verify state is normalized
    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: Basic amplitude encoding works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_power_of_two() {
    println!("Testing amplitude encoding with power-of-two input...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test with 8 input values (fills 3-qubit state)
    let input: Vec<f64> = (1..=8).map(|x| x as f64).collect();
    let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
    let state_len = 8;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify all elements are correctly normalized
    for i in 0..state_len {
        let expected = input[i] / norm;
        assert!(
            (state_h[i].x - expected).abs() < EPSILON,
            "Element {} should be {}, got {}",
            i,
            expected,
            state_h[i].x
        );
        assert!((state_h[i].y).abs() < EPSILON, "Imaginary part should be 0");
    }

    // Verify normalization
    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: Power-of-two input encoding works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_odd_input_length() {
    println!("Testing amplitude encoding with odd input length...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test with 3 input values, state size 4
    let input = vec![1.0, 2.0, 2.0];
    let norm = (1.0_f64 + 4.0 + 4.0).sqrt(); // 3.0
    let state_len = 4;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify: [1/3, 2/3, 2/3, 0]
    assert!((state_h[0].x - 1.0 / 3.0).abs() < EPSILON);
    assert!((state_h[1].x - 2.0 / 3.0).abs() < EPSILON);
    assert!((state_h[2].x - 2.0 / 3.0).abs() < EPSILON);
    assert!(
        (state_h[3].x).abs() < EPSILON,
        "Fourth element should be padded with 0"
    );

    println!("PASS: Odd input length handled correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_large_state() {
    println!("Testing amplitude encoding with large state vector...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Test with 1024 elements (10 qubits)
    let input_len = 1024;
    let input: Vec<f64> = (0..input_len).map(|i| (i + 1) as f64).collect();
    let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
    let state_len = 1024;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Spot check a few values
    for i in [0, 100, 500, 1023] {
        let expected = input[i] / norm;
        assert!(
            (state_h[i].x - expected).abs() < EPSILON,
            "Element {} mismatch",
            i
        );
    }

    // Verify normalization
    let total_prob: f64 = state_h.iter().map(|c| c.x * c.x + c.y * c.y).sum();
    assert!(
        (total_prob - 1.0).abs() < EPSILON,
        "Total probability should be 1.0"
    );

    println!("PASS: Large state vector encoding works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_zero_norm_error() {
    println!("Testing amplitude encoding with zero norm (error case)...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![0.0, 0.0, 0.0];
    let norm = 0.0; // Invalid!
    let state_len = 4;

    let input_d = device.htod_copy(input).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);
    let _ = state_d; // suppress unused warning

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            3,
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    // Should return CUDA error code for invalid value
    assert_ne!(result, 0, "Should reject zero norm");
    println!(
        "PASS: Zero norm correctly rejected with error code {}",
        result
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_negative_norm_error() {
    println!("Testing amplitude encoding with negative norm (error case)...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    let input = vec![1.0, 2.0, 3.0];
    let norm = -5.0; // Invalid!
    let state_len = 4;

    let input_d = device.htod_copy(input).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);
    let _ = state_d;

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            3,
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    // Should return CUDA error code for invalid value
    assert_ne!(result, 0, "Should reject negative norm");
    println!(
        "PASS: Negative norm correctly rejected with error code {}",
        result
    );
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_vectorized_load() {
    println!("Testing vectorized double2 memory access optimization...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Use exactly 16 elements to test vectorized loads (8 threads * 2 elements each)
    let input: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
    let state_len = 16;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // Verify all elements processed correctly through vectorized loads
    for i in 0..state_len {
        let expected = input[i] / norm;
        assert!(
            (state_h[i].x - expected).abs() < EPSILON,
            "Vectorized load: element {} should be {}, got {}",
            i,
            expected,
            state_h[i].x
        );
    }

    println!("PASS: Vectorized memory access works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_small_input_large_state() {
    println!("Testing small input with large state vector...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Only 2 input values, but 16-element state (padding with zeros)
    let input = vec![3.0, 4.0];
    let norm = 5.0;
    let state_len = 16;

    let input_d = device.htod_copy(input.clone()).unwrap();
    let (state_d, page_table) = create_single_page_state(&device, state_len);

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            state_len,
            (state_len as f64).log2() as u32,
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    let state_h = device.dtoh_sync_copy(&state_d).unwrap();

    // First two elements should be normalized values
    assert!((state_h[0].x - 0.6).abs() < EPSILON);
    assert!((state_h[1].x - 0.8).abs() < EPSILON);

    // Rest should be zero
    for i in 2..state_len {
        assert!(
            state_h[i].x.abs() < EPSILON && state_h[i].y.abs() < EPSILON,
            "Element {} should be zero-padded",
            i
        );
    }

    println!("PASS: Small input with large state padding works correctly");
}

#[test]
#[cfg(target_os = "linux")]
fn test_amplitude_encode_multi_page() {
    println!("Testing paged encoding with multiple pages...");

    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(_) => {
            println!("SKIP: No CUDA device available");
            return;
        }
    };

    // Use 16 elements split across 4 pages of 4 elements each
    let input: Vec<f64> = (1..=16).map(|x| x as f64).collect();
    let norm: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
    let state_len = 16;
    let page_size = 4;
    let num_pages = state_len / page_size;

    // Allocate separate pages
    let mut pages: Vec<cudarc::driver::CudaSlice<CuDoubleComplex>> = Vec::new();
    let mut page_ptrs: Vec<u64> = Vec::new();

    for _ in 0..num_pages {
        let page = device.alloc_zeros::<CuDoubleComplex>(page_size).unwrap();
        page_ptrs.push(*page.device_ptr() as u64);
        pages.push(page);
    }

    let input_d = device.htod_copy(input.clone()).unwrap();
    let page_table = device.htod_copy(page_ptrs).unwrap();

    let result = unsafe {
        launch_amplitude_encode(
            *input_d.device_ptr() as *const f64,
            *page_table.device_ptr() as *mut *mut std::ffi::c_void,
            input.len(),
            state_len,
            0,
            page_size,
            2, // log2(4) = 2
            norm,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 0, "Kernel launch should succeed");

    // Synchronize before reading
    device.synchronize().unwrap();

    // Verify all elements across all pages
    for page_id in 0..num_pages {
        let page_h = device.dtoh_sync_copy(&pages[page_id]).unwrap();
        for offset in 0..page_size {
            let global_idx = page_id * page_size + offset;
            let expected = input[global_idx] / norm;
            assert!(
                (page_h[offset].x - expected).abs() < EPSILON,
                "Page {} element {} (global {}): expected {}, got {}",
                page_id,
                offset,
                global_idx,
                expected,
                page_h[offset].x
            );
        }
    }

    println!("PASS: Multi-page encoding works correctly");
}

#[test]
#[cfg(not(target_os = "linux"))]
fn test_amplitude_encode_dummy_non_linux() {
    println!("Testing dummy implementation on non-Linux platform...");

    // The dummy implementation should return error code 999
    let result = unsafe {
        qdp_kernels::launch_amplitude_encode(
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            0,
            0,
            0,
            0,
            1.0,
            std::ptr::null_mut(),
        )
    };

    assert_eq!(result, 999, "Dummy implementation should return 999");
    println!("PASS: Non-Linux dummy implementation returns expected error code");
}
