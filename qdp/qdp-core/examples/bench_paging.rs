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

//! Benchmark paged state vector allocation and encoding
//!
//! Run: cargo run -p qdp-core --example bench_paging --release

use std::sync::Arc;
use std::time::Instant;

fn main() {
    #[cfg(target_os = "linux")]
    {
        use cudarc::driver::CudaDevice;
        use qdp_core::gpu::memory::{GpuStateVector, DEFAULT_PAGE_SIZE_ELEMENTS};
        use qdp_core::QdpEngine;

        println!("=== Paged State Vector Benchmark ===\n");

        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("No CUDA device: {:?}", e);
                return;
            }
        };

        // Warm up
        let _ = GpuStateVector::new(&device, 10);

        println!("Default page size: {} elements ({} MB)\n",
            DEFAULT_PAGE_SIZE_ELEMENTS,
            DEFAULT_PAGE_SIZE_ELEMENTS * 16 / (1024 * 1024)
        );

        // Test allocation at different qubit counts
        println!("--- Allocation Benchmark ---");
        println!("{:>8} {:>8} {:>12} {:>12} {:>10}", "Qubits", "Pages", "Alloc (ms)", "Free (ms)", "Total (ms)");
        println!("{}", "-".repeat(60));

        for qubits in [10, 14, 18, 20, 22, 24] {
            let state_size = 1usize << qubits;
            let num_pages = (state_size + DEFAULT_PAGE_SIZE_ELEMENTS - 1) / DEFAULT_PAGE_SIZE_ELEMENTS;

            // Check if we have enough memory
            let required_bytes = state_size * 16; // 16 bytes per complex
            let (free, _total) = unsafe {
                let mut free = 0usize;
                let mut total = 0usize;
                unsafe extern "C" { fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32; }
                cudaMemGetInfo(&mut free, &mut total);
                (free, total)
            };

            if required_bytes > free {
                println!("{:>8} {:>8} {:>12} {:>12} {:>10}",
                    qubits, num_pages, "OOM", "-", "-");
                continue;
            }

            // Benchmark allocation
            let alloc_start = Instant::now();
            let state = match GpuStateVector::new(&device, qubits) {
                Ok(s) => s,
                Err(e) => {
                    println!("{:>8} {:>8} {:>12}", qubits, num_pages, format!("ERR: {:?}", e));
                    continue;
                }
            };
            let alloc_time = alloc_start.elapsed();

            // Synchronize to ensure allocation is complete
            let _ = device.synchronize();

            // Benchmark deallocation
            let free_start = Instant::now();
            drop(state);
            let _ = device.synchronize();
            let free_time = free_start.elapsed();

            println!("{:>8} {:>8} {:>12.2} {:>12.2} {:>10.2}",
                qubits,
                num_pages,
                alloc_time.as_secs_f64() * 1000.0,
                free_time.as_secs_f64() * 1000.0,
                (alloc_time + free_time).as_secs_f64() * 1000.0
            );
        }

        // Benchmark encoding throughput
        println!("\n--- Encoding Benchmark ---");
        println!("{:>8} {:>10} {:>12} {:>12}", "Qubits", "Elements", "Time (ms)", "GB/s");
        println!("{}", "-".repeat(50));

        let engine = QdpEngine::new(0).expect("Failed to create engine");

        for qubits in [10, 14, 18, 20] {
            let state_size = 1usize << qubits;

            // Check memory
            let required_bytes = state_size * 16 + state_size * 8; // state + input
            let (free, _) = unsafe {
                let mut free = 0usize;
                let mut total = 0usize;
                unsafe extern "C" { fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32; }
                cudaMemGetInfo(&mut free, &mut total);
                (free, total)
            };

            if required_bytes > free {
                println!("{:>8} {:>10} {:>12} {:>12}", qubits, state_size, "OOM", "-");
                continue;
            }

            // Create test data
            let data: Vec<f64> = (0..state_size).map(|i| i as f64 + 1.0).collect();

            // Warm up
            let _ = engine.encode(&data, qubits, "amplitude");

            // Benchmark
            let iterations = 5;
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = engine.encode(&data, qubits, "amplitude");
            }
            let elapsed = start.elapsed();
            let avg_time = elapsed.as_secs_f64() / iterations as f64;

            // Calculate throughput (input data + output state)
            let bytes_transferred = (state_size * 8 + state_size * 16) as f64; // input + output
            let throughput_gbs = bytes_transferred / avg_time / 1e9;

            println!("{:>8} {:>10} {:>12.2} {:>12.2}",
                qubits,
                state_size,
                avg_time * 1000.0,
                throughput_gbs
            );
        }

        println!("\n=== Benchmark Complete ===");
    }

    #[cfg(not(target_os = "linux"))]
    {
        println!("CUDA only available on Linux");
    }
}
