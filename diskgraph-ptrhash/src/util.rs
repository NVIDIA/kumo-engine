//! Internal utilities for ptrhash implementation.

/// Prefetch the given cacheline into L1 cache.
pub(crate) fn prefetch_index<T>(s: &[T], index: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let ptr = s.as_ptr().add(index) as *const i8;
        std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        let ptr = s.as_ptr().add(index) as *const i8;
        std::arch::x86::_mm_prefetch(ptr, std::arch::x86::_MM_HINT_T0);
    }
    #[cfg(target_arch = "aarch64")]
    {
        // TODO: Put this behind a feature flag.
        let _ = (s.as_ptr(), index);
        // std::arch::aarch64::_prefetch(ptr as *const i8, std::arch::aarch64::_PREFETCH_LOCALITY3);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        // Do nothing.
    }
}

pub(crate) fn mul_high(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}
