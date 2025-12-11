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

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr};
use qdp_kernels::CuDoubleComplex;
use crate::error::{MahoutError, Result};

/// Default page size: 256 MB worth of complex numbers (16M elements)
/// Power of 2 enables fast bit-shift division in kernels
pub const DEFAULT_PAGE_SIZE_ELEMENTS: usize = 16 * 1024 * 1024; // 16M elements = 256MB

#[cfg(target_os = "linux")]
fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(target_os = "linux")]
fn cuda_error_to_string(code: i32) -> &'static str {
    match code {
        0 => "cudaSuccess",
        2 => "cudaErrorMemoryAllocation",
        3 => "cudaErrorInitializationError",
        30 => "cudaErrorUnknown",
        _ => "Unknown CUDA error",
    }
}

#[cfg(target_os = "linux")]
fn query_cuda_mem_info() -> Result<(usize, usize)> {
    unsafe {
        unsafe extern "C" {
            fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        }

        let mut free_bytes: usize = 0;
        let mut total_bytes: usize = 0;
        let result = cudaMemGetInfo(&mut free_bytes as *mut usize, &mut total_bytes as *mut usize);

        if result != 0 {
            return Err(MahoutError::Cuda(format!(
                "cudaMemGetInfo failed: {} ({})",
                result,
                cuda_error_to_string(result)
            )));
        }

        Ok((free_bytes, total_bytes))
    }
}

#[cfg(target_os = "linux")]
fn build_oom_message(context: &str, requested_bytes: usize, qubits: Option<usize>, free: usize, total: usize) -> String {
    let qubit_hint = qubits
        .map(|q| format!(" (qubits={})", q))
        .unwrap_or_default();

    format!(
        "GPU out of memory during {context}{qubit_hint}: requested {:.2} MiB, free {:.2} MiB / total {:.2} MiB. Reduce qubits or batch size and retry.",
        bytes_to_mib(requested_bytes),
        bytes_to_mib(free),
        bytes_to_mib(total),
    )
}

/// Guard that checks available GPU memory before attempting a large allocation.
///
/// Returns a MemoryAllocation error with a helpful message when the request
/// exceeds the currently reported free memory.
#[cfg(target_os = "linux")]
pub(crate) fn ensure_device_memory_available(requested_bytes: usize, context: &str, qubits: Option<usize>) -> Result<()> {
    let (free, total) = query_cuda_mem_info()?;

    if (requested_bytes as u64) > (free as u64) {
        return Err(MahoutError::MemoryAllocation(build_oom_message(
            context,
            requested_bytes,
            qubits,
            free,
            total,
        )));
    }

    Ok(())
}

/// Wraps CUDA allocation errors with an OOM-aware MahoutError.
#[cfg(target_os = "linux")]
pub(crate) fn map_allocation_error(
    requested_bytes: usize,
    context: &str,
    qubits: Option<usize>,
    source: impl std::fmt::Debug,
) -> MahoutError {
    match query_cuda_mem_info() {
        Ok((free, total)) => {
            if (requested_bytes as u64) > (free as u64) {
                MahoutError::MemoryAllocation(build_oom_message(
                    context,
                    requested_bytes,
                    qubits,
                    free,
                    total,
                ))
            } else {
                MahoutError::MemoryAllocation(format!(
                    "GPU allocation failed during {context}: requested {:.2} MiB. CUDA error: {:?}",
                    bytes_to_mib(requested_bytes),
                    source,
                ))
            }
        }
        Err(e) => MahoutError::MemoryAllocation(format!(
            "GPU allocation failed during {context}: requested {:.2} MiB. Unable to fetch memory info: {:?}; CUDA error: {:?}",
            bytes_to_mib(requested_bytes),
            e,
            source,
        )),
    }
}

/// RAII wrapper for GPU memory buffer
/// Automatically frees GPU memory when dropped
pub struct GpuBufferRaw {
    pub(crate) slice: CudaSlice<CuDoubleComplex>,
}

impl GpuBufferRaw {
    /// Get raw pointer to GPU memory
    ///
    /// # Safety
    /// Valid only while GpuBufferRaw is alive
    pub fn ptr(&self) -> *mut CuDoubleComplex {
        *self.slice.device_ptr() as *mut CuDoubleComplex
    }
}

/// RAII wrapper for device-side page table (array of pointers)
pub struct GpuPageTable {
    pub(crate) slice: CudaSlice<u64>, // Store as u64 for pointer-sized values
}

impl GpuPageTable {
    /// Get raw pointer to page table on device
    pub fn ptr(&self) -> *mut *mut CuDoubleComplex {
        *self.slice.device_ptr() as *mut *mut CuDoubleComplex
    }
}

/// Paged quantum state vector on GPU
///
/// Manages complex128 array of size 2^n (n = qubits) in GPU memory using
/// multiple page allocations. This enables state vectors larger than a
/// single contiguous allocation limit.
///
/// Memory layout:
/// - `pages`: Vector of GPU allocations, each up to `page_size_elements`
/// - `page_table_d`: Device-side array of page pointers for kernel access
///
/// Thread-safe: Send + Sync
pub struct GpuStateVector {
    /// Individual page allocations
    pub(crate) pages: Vec<Arc<GpuBufferRaw>>,
    /// Device-side page table (array of page pointers)
    pub(crate) page_table: Arc<GpuPageTable>,
    /// Elements per page (power of 2)
    pub page_size_elements: usize,
    /// log2(page_size_elements) for fast bit-shift division
    pub page_shift: u32,
    /// Number of qubits
    pub num_qubits: usize,
    /// Total elements across all pages
    pub size_elements: usize,
    /// Number of pages
    pub num_pages: usize,
}

// Safety: CudaSlice and Arc are both Send + Sync
unsafe impl Send for GpuStateVector {}
unsafe impl Sync for GpuStateVector {}
unsafe impl Send for GpuPageTable {}
unsafe impl Sync for GpuPageTable {}

impl GpuStateVector {
    /// Create paged GPU state vector for n qubits
    /// Allocates 2^n complex numbers across multiple pages on GPU (freed on drop)
    pub fn new(device: &Arc<CudaDevice>, qubits: usize) -> Result<Self> {
        Self::new_with_page_size(device, qubits, DEFAULT_PAGE_SIZE_ELEMENTS)
    }

    /// Create paged GPU state vector with custom page size
    /// Page size must be a power of 2
    pub fn new_with_page_size(_device: &Arc<CudaDevice>, qubits: usize, page_size: usize) -> Result<Self> {
        // Validate page size is power of 2
        if !page_size.is_power_of_two() {
            return Err(MahoutError::InvalidInput(
                format!("Page size must be power of 2, got {}", page_size)
            ));
        }

        let total_elements: usize = 1usize << qubits;
        let page_shift = page_size.trailing_zeros();
        let num_pages = (total_elements + page_size - 1) / page_size;

        #[cfg(target_os = "linux")]
        {
            let total_bytes = total_elements
                .checked_mul(std::mem::size_of::<CuDoubleComplex>())
                .ok_or_else(|| MahoutError::MemoryAllocation(
                    format!("Requested GPU allocation size overflow (elements={})", total_elements)
                ))?;

            // Pre-flight check for total memory needed
            ensure_device_memory_available(total_bytes, "paged state vector allocation", Some(qubits))?;

            // Allocate pages
            let mut pages = Vec::with_capacity(num_pages);
            let mut page_pointers: Vec<u64> = Vec::with_capacity(num_pages);

            for page_idx in 0..num_pages {
                let remaining = total_elements - (page_idx * page_size);
                let this_page_size = remaining.min(page_size);

                let slice = unsafe {
                    _device.alloc::<CuDoubleComplex>(this_page_size)
                }.map_err(|e| map_allocation_error(
                    this_page_size * std::mem::size_of::<CuDoubleComplex>(),
                    &format!("page {} allocation", page_idx),
                    Some(qubits),
                    e,
                ))?;

                let page = Arc::new(GpuBufferRaw { slice });
                page_pointers.push(page.ptr() as u64);
                pages.push(page);
            }

            // Upload page table to device
            let page_table_slice = _device.htod_copy(page_pointers)
                .map_err(|e| MahoutError::Cuda(format!("Failed to upload page table: {:?}", e)))?;

            let page_table = Arc::new(GpuPageTable { slice: page_table_slice });

            Ok(Self {
                pages,
                page_table,
                page_size_elements: page_size,
                page_shift,
                num_qubits: qubits,
                size_elements: total_elements,
                num_pages,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA is only available on Linux. This build does not support GPU operations.".to_string()))
        }
    }

    /// Get raw pointer to page table on device (for kernel launches)
    pub fn page_table_ptr(&self) -> *mut *mut CuDoubleComplex {
        self.page_table.ptr()
    }

    /// Get pointer to first page (for single-page compatibility)
    /// Only valid if state fits in one page
    pub fn ptr(&self) -> *mut CuDoubleComplex {
        self.pages.first().map(|p| p.ptr()).unwrap_or(std::ptr::null_mut())
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the size in elements (2^n where n is number of qubits)
    pub fn size_elements(&self) -> usize {
        self.size_elements
    }

    /// Check if state vector fits in a single page
    pub fn is_single_page(&self) -> bool {
        self.num_pages == 1
    }

    /// Create paged GPU state vector for a batch of samples
    /// Allocates num_samples * 2^qubits complex numbers across pages on GPU
    pub fn new_batch(_device: &Arc<CudaDevice>, num_samples: usize, qubits: usize) -> Result<Self> {
        Self::new_batch_with_page_size(_device, num_samples, qubits, DEFAULT_PAGE_SIZE_ELEMENTS)
    }

    /// Create paged GPU state vector for a batch with custom page size
    pub fn new_batch_with_page_size(_device: &Arc<CudaDevice>, num_samples: usize, qubits: usize, page_size: usize) -> Result<Self> {
        // Validate page size is power of 2
        if !page_size.is_power_of_two() {
            return Err(MahoutError::InvalidInput(
                format!("Page size must be power of 2, got {}", page_size)
            ));
        }

        let single_state_size: usize = 1usize << qubits;
        let total_elements = num_samples.checked_mul(single_state_size)
            .ok_or_else(|| MahoutError::MemoryAllocation(
                format!("Batch size overflow: {} samples * {} elements", num_samples, single_state_size)
            ))?;

        let page_shift = page_size.trailing_zeros();
        let num_pages = (total_elements + page_size - 1) / page_size;

        #[cfg(target_os = "linux")]
        {
            let total_bytes = total_elements
                .checked_mul(std::mem::size_of::<CuDoubleComplex>())
                .ok_or_else(|| MahoutError::MemoryAllocation(
                    format!("Requested GPU allocation size overflow (elements={})", total_elements)
                ))?;

            // Pre-flight check
            ensure_device_memory_available(total_bytes, "paged batch state vector allocation", Some(qubits))?;

            // Allocate pages
            let mut pages = Vec::with_capacity(num_pages);
            let mut page_pointers: Vec<u64> = Vec::with_capacity(num_pages);

            for page_idx in 0..num_pages {
                let remaining = total_elements - (page_idx * page_size);
                let this_page_size = remaining.min(page_size);

                let slice = unsafe {
                    _device.alloc::<CuDoubleComplex>(this_page_size)
                }.map_err(|e| map_allocation_error(
                    this_page_size * std::mem::size_of::<CuDoubleComplex>(),
                    &format!("batch page {} allocation", page_idx),
                    Some(qubits),
                    e,
                ))?;

                let page = Arc::new(GpuBufferRaw { slice });
                page_pointers.push(page.ptr() as u64);
                pages.push(page);
            }

            // Upload page table to device
            let page_table_slice = _device.htod_copy(page_pointers)
                .map_err(|e| MahoutError::Cuda(format!("Failed to upload page table: {:?}", e)))?;

            let page_table = Arc::new(GpuPageTable { slice: page_table_slice });

            Ok(Self {
                pages,
                page_table,
                page_size_elements: page_size,
                page_shift,
                num_qubits: qubits,
                size_elements: total_elements,
                num_pages,
            })
        }

        #[cfg(not(target_os = "linux"))]
        {
            Err(MahoutError::Cuda("CUDA is only available on Linux. This build does not support GPU operations.".to_string()))
        }
    }
}
