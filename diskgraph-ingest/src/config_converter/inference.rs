/// Estimate the optimal number of shards based on total dataset size.
///
/// This function attempts to balance two competing goals:
/// 1. **Parallelism**: More shards allow for higher concurrency during ingestion and training.
/// 2. **Overhead**: Too many shards increase metadata overhead and file system fragmentation.
///
/// The estimation logic adapts to the *system's* capabilities (available RAM and CPU cores)
/// as well as the *dataset's* size. It uses a "smooth" scaling function to determine
/// the target shard size.
pub fn estimate_num_shards(total_bytes: u64) -> usize {
    if total_bytes == 0 {
        return 1;
    }
    let shard_bytes = shard_target_bytes(total_bytes);
    total_bytes.div_ceil(shard_bytes) as usize
}

/// Calculate the target size (in bytes) for a single shard.
///
/// If system memory or CPU count cannot be determined, it falls back to a safe default (1 GB).
fn shard_target_bytes(total_bytes: u64) -> u64 {
    const FALLBACK_SHARD_BYTES: u64 = 1024 * 1024 * 1024; // 1 GB
    let mem_total = read_system_memory_bytes();
    let cpu_count = std::thread::available_parallelism()
        .ok()
        .map(|n| n.get() as u64);
    match (mem_total, cpu_count) {
        (Some(mem_total), Some(cpu_count)) if cpu_count > 0 => {
            let per_cpu = mem_total / cpu_count;
            let target = smooth_shard_target(per_cpu, total_bytes);
            if target == 0 {
                FALLBACK_SHARD_BYTES
            } else {
                target
            }
        }
        _ => FALLBACK_SHARD_BYTES,
    }
}

/// Calculate the target shard size using a smooth interpolation function.
///
/// The goal is to determine what fraction (or multiple) of per-CPU RAM should be used
/// for a single shard.
///
/// Logic:
/// - **Small Datasets (< 40 GB)**: We want *small* shards to ensure we utilize all cores.
///   Divisor ranges from 16.0 down to 4.0.
///   Target Shard Size: `(RAM / 16)` to `(RAM / 4)`.
///
/// - **Medium Datasets (40 GB - 1 TB)**: We transition to "standard" sizing.
///   Divisor ranges from 4.0 down to 1.0.
///   Target Shard Size: `(RAM / 4)` to `RAM`.
///
/// - **Large Datasets (1 TB - 10 TB)**: We increase shard size to reduce metadata overhead.
///   Divisor ranges from 1.0 down to 0.5.
///   Target Shard Size: `RAM` to `2 * RAM`.
///
/// - **Huge Datasets (> 10 TB)**: Constant factor.
///   Divisor is 0.75.
///   Target Shard Size: `1.33 * RAM`.
fn smooth_shard_target(per_cpu_bytes: u64, total_bytes: u64) -> u64 {
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let divisor = if total_gb <= 40.0 {
        // Range: 10GB - 40GB
        // Smooth interpolation t: 0.0 -> 1.0
        let t = ((total_gb - 10.0) / 30.0).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t); // Smoothstep function
                                              // Divisor: 16.0 -> 4.0
        16.0 - 12.0 * smooth
    } else if total_gb <= 1024.0 {
        // Range: 40GB - 1TB
        let t = ((total_gb - 40.0) / (1024.0 - 40.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        // Divisor: 4.0 -> 1.0
        4.0 - 3.0 * smooth
    } else if total_gb <= 10_240.0 {
        // Range: 1TB - 10TB
        let t = ((total_gb - 1024.0) / (10_240.0 - 1024.0)).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        // Divisor: 1.0 -> 0.5
        1.0 - 0.5 * smooth
    } else {
        // > 10TB
        0.75
    };
    (per_cpu_bytes as f64 / divisor).round() as u64
}

/// Read total system memory from `/proc/meminfo` (Linux only).
fn read_system_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in contents.lines() {
            if let Some(value) = line.strip_prefix("MemTotal:") {
                let parts: Vec<&str> = value.split_whitespace().collect();
                if parts.len() >= 2 && parts[1] == "kB" {
                    return parts[0].parse::<u64>().ok().map(|kb| kb * 1024);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn test_estimate_num_shards_zero() {
        assert_eq!(estimate_num_shards(0), 1);
    }

    #[test]
    fn test_smooth_shard_target_small() {
        let per_cpu = 2 * GB; // 2GB per core
                              // Case: 10GB total (Start of small range)
                              // t=0, smooth=0, divisor=16. Target = 2GB/16 = 128MB.
        let target_10gb = smooth_shard_target(per_cpu, 10 * GB);
        assert_eq!(target_10gb, 128 * 1024 * 1024);

        // Case: 40GB total (End of small range)
        // t=1, smooth=1, divisor=4. Target = 2GB/4 = 512MB.
        let target_40gb = smooth_shard_target(per_cpu, 40 * GB);
        assert_eq!(target_40gb, 512 * 1024 * 1024);
    }

    #[test]
    fn test_smooth_shard_target_medium() {
        let per_cpu = 4 * GB; // 4GB per core
                              // Case: 40GB total (Start of medium range)
                              // Divisor starts at 4.0. Target = 4GB/4 = 1GB.
        let target_40gb = smooth_shard_target(per_cpu, 40 * GB);
        assert_eq!(target_40gb, GB);

        // Case: 1TB total (End of medium range)
        // Divisor ends at 1.0. Target = 4GB/1 = 4GB.
        let target_1tb = smooth_shard_target(per_cpu, 1024 * GB);
        assert_eq!(target_1tb, 4 * GB);
    }

    #[test]
    fn test_smooth_shard_target_large() {
        let per_cpu = 8 * GB;
        // Case: 1TB total (Start of large range)
        // Divisor starts at 1.0. Target = 8GB.
        let target_1tb = smooth_shard_target(per_cpu, 1024 * GB);
        assert_eq!(target_1tb, 8 * GB);

        // Case: 10TB total (End of large range)
        // Divisor ends at 0.5. Target = 8GB/0.5 = 16GB.
        let target_10tb = smooth_shard_target(per_cpu, 10_240 * GB);
        assert_eq!(target_10tb, 16 * GB);
    }

    #[test]
    fn test_smooth_shard_target_huge() {
        let per_cpu = 8 * GB;
        // Case: 20TB (> 10TB)
        // Divisor fixed at 0.75. Target = 8GB/0.75 = 10.66GB.
        let target_huge = smooth_shard_target(per_cpu, 20_000 * GB);
        // 8 / 0.75 = 32/3 = 10.666...
        let expected = (8.0 * 1024.0 * 1024.0 * 1024.0 / 0.75) as u64;
        // Allow off-by-one due to float rounding differences in implementation vs test
        assert!(
            target_huge >= expected - 1 && target_huge <= expected + 1,
            "Target {} not within 1 byte of expected {}",
            target_huge,
            expected
        );
    }

    #[test]
    fn test_monotonicity_within_ranges() {
        // Ensure shard size generally increases or stays same as dataset grows within defined ranges.
        // Note: There is a known discontinuity at 10TB where shard size drops (divisor 0.5 -> 0.75),
        // effectively increasing shard count for very large datasets.
        let per_cpu = 2 * GB;

        // Range 1: Small to Medium (< 1TB)
        let mut prev_target = 0;
        let mut bytes = GB;
        while bytes < 1_000 * GB {
            let target = smooth_shard_target(per_cpu, bytes);
            assert!(
                target >= prev_target,
                "Target shard size decreased at {} GB: {} < {}",
                bytes / GB,
                target,
                prev_target
            );
            prev_target = target;
            bytes += 10 * GB;
        }

        // Range 2: Large (1TB - 10TB)
        prev_target = 0; // Reset
        bytes = 1_024 * GB;
        while bytes <= 10_240 * GB {
            let target = smooth_shard_target(per_cpu, bytes);
            assert!(
                target >= prev_target,
                "Target shard size decreased at {} GB: {} < {}",
                bytes / GB,
                target,
                prev_target
            );
            prev_target = target;
            bytes += 100 * GB;
        }
    }
}
