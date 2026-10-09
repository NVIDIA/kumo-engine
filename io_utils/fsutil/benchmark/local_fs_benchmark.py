#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local filesystem benchmark for testing read and write latency and bandwidth.

This benchmark tests the performance of the local filesystem mounted at
/local/user/vol2/bench by measuring:
- Write latency (time to write files of various sizes)
- Read latency (time to read files of various sizes)
- Write bandwidth (MB/s for different file sizes)
- Read bandwidth (MB/s for different file sizes)
- Random access performance
- Sequential access performance
"""

import argparse
import json
import os.path as osp
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from io_utils import fsutil


def gen_data(n_bytes: int) -> bytes:
    data = (str(n_bytes) + "data" *
            ((n_bytes + 3) // 4)).encode("utf-8")[:n_bytes]
    assert len(data) == n_bytes
    return data


@dataclass
class BenchmarkResult:
    """Results from a single benchmark operation."""
    operation: str
    file_size: int
    duration: float
    bandwidth_mbps: float
    latency_ms: float
    iterations: int = 1


class LocalFSBenchmark:
    """Benchmark class for testing local filesystem performance."""
    def __init__(self, mount_point: str = "/local/user/vol2/bench"):
        self.mount_point = mount_point
        self.test_dir = osp.join(mount_point, "benchmark_test")
        self.results: List[BenchmarkResult] = []

        # Ensure test directory exists
        if not osp.exists(self.test_dir):
            fsutil.makedirs(self.test_dir)

    def cleanup(self):
        """Clean up test files."""
        try:
            fsutil.rm(self.test_dir)
        except Exception as e:
            print(f"Warning: Could not clean up {self.test_dir}: {e}")

    def _write_file(self, filename: str, data: bytes) -> float:
        """Write a file and return the time taken."""
        start_time = time.time()
        fsutil.write_binary(filename, data)
        end_time = time.time()
        return end_time - start_time

    def _read_file(self, filename: str) -> Tuple[bytes, float]:
        """Read a file and return the data and time taken."""
        start_time = time.time()
        data = fsutil.read_binary(filename)
        end_time = time.time()
        return data, end_time - start_time

    def _calculate_bandwidth(self, size_bytes: int, duration: float) -> float:
        """Calculate bandwidth in MB/s."""
        size_mb = size_bytes / (1024 * 1024)
        return size_mb / duration if duration > 0 else 0

    def _calculate_latency(self, duration: float) -> float:
        """Calculate latency in milliseconds."""
        return duration * 1000

    def benchmark_write_latency(self, file_sizes: List[int],
                                iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark write latency for different file sizes."""
        print("\n=== Write Latency Benchmark ===")
        results = []

        for size in file_sizes:
            print(f"Testing write latency for {size} bytes...")
            durations = []

            for i in range(iterations):
                filename = osp.join(self.test_dir, f"write_latency_{size}_{i}")
                data = gen_data(size)

                duration = self._write_file(filename, data)
                durations.append(duration)

                # Clean up immediately
                fsutil.rm(filename, ignore_errors=True)

            avg_duration = statistics.mean(durations)
            latency_ms = self._calculate_latency(avg_duration)
            bandwidth_mbps = self._calculate_bandwidth(size, avg_duration)

            result = BenchmarkResult(operation="write_latency", file_size=size,
                                     duration=avg_duration,
                                     bandwidth_mbps=bandwidth_mbps,
                                     latency_ms=latency_ms,
                                     iterations=iterations)
            results.append(result)

            print(f"  Size: {size} bytes, Avg Latency: {latency_ms:.2f} ms, "
                  f"Bandwidth: {bandwidth_mbps:.2f} MB/s")

        self.results.extend(results)
        return results

    def benchmark_read_latency(self, file_sizes: List[int],
                               iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark read latency for different file sizes."""
        print("\n=== Read Latency Benchmark ===")
        results = []

        for size in file_sizes:
            print(f"Testing read latency for {size} bytes...")
            durations = []

            # First create the file
            filename = osp.join(self.test_dir, f"read_latency_{size}")
            data = gen_data(size)
            fsutil.write_binary(filename, data)

            for _ in range(iterations):
                read_data, duration = self._read_file(filename)
                durations.append(duration)

                # Verify data integrity
                if read_data != data:
                    print(f"  Warning: Data mismatch for size {size}")

            # Clean up
            fsutil.rm(filename, ignore_errors=True)

            avg_duration = statistics.mean(durations)
            latency_ms = self._calculate_latency(avg_duration)
            bandwidth_mbps = self._calculate_bandwidth(size, avg_duration)

            result = BenchmarkResult(operation="read_latency", file_size=size,
                                     duration=avg_duration,
                                     bandwidth_mbps=bandwidth_mbps,
                                     latency_ms=latency_ms,
                                     iterations=iterations)
            results.append(result)

            print(f"  Size: {size} bytes, Avg Latency: {latency_ms:.2f} ms, "
                  f"Bandwidth: {bandwidth_mbps:.2f} MB/s")

        self.results.extend(results)
        return results

    def benchmark_write_bandwidth(
            self, file_sizes: List[int],
            iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark write bandwidth for different file sizes."""
        print("\n=== Write Bandwidth Benchmark ===")
        results = []

        for size in file_sizes:
            print(f"Testing write bandwidth for {size} bytes...")
            durations = []

            for _ in range(iterations):
                filename = osp.join(self.test_dir,
                                    f"write_bandwidth_{size}_{_}")
                data = gen_data(size)

                duration = self._write_file(filename, data)
                durations.append(duration)

                # Clean up immediately
                fsutil.rm(filename, ignore_errors=True)

            avg_duration = statistics.mean(durations)
            bandwidth_mbps = self._calculate_bandwidth(size, avg_duration)
            latency_ms = self._calculate_latency(avg_duration)

            result = BenchmarkResult(operation="write_bandwidth",
                                     file_size=size, duration=avg_duration,
                                     bandwidth_mbps=bandwidth_mbps,
                                     latency_ms=latency_ms,
                                     iterations=iterations)
            results.append(result)

            print(
                f"  Size: {size} bytes, Bandwidth: {bandwidth_mbps:.2f} MB/s, "
                f"Latency: {latency_ms:.2f} ms")

        self.results.extend(results)
        return results

    def benchmark_read_bandwidth(self, file_sizes: List[int],
                                 iterations: int = 3) -> List[BenchmarkResult]:
        """Benchmark read bandwidth for different file sizes."""
        print("\n=== Read Bandwidth Benchmark ===")
        results = []

        for size in file_sizes:
            print(f"Testing read bandwidth for {size} bytes...")
            durations = []

            # First create the file
            filename = osp.join(self.test_dir, f"read_bandwidth_{size}")
            data = gen_data(size)
            fsutil.write_binary(filename, data)

            for _ in range(iterations):
                read_data, duration = self._read_file(filename)
                durations.append(duration)

                # Verify data integrity
                if read_data != data:
                    print(f"  Warning: Data mismatch for size {size}")

            fsutil.rm(filename, ignore_errors=True)

            avg_duration = statistics.mean(durations)
            bandwidth_mbps = self._calculate_bandwidth(size, avg_duration)
            latency_ms = self._calculate_latency(avg_duration)

            result = BenchmarkResult(operation="read_bandwidth",
                                     file_size=size, duration=avg_duration,
                                     bandwidth_mbps=bandwidth_mbps,
                                     latency_ms=latency_ms,
                                     iterations=iterations)
            results.append(result)

            print(
                f"  Size: {size} bytes, Bandwidth: {bandwidth_mbps:.2f} MB/s, "
                f"Latency: {latency_ms:.2f} ms")

        self.results.extend(results)
        return results

    def benchmark_sequential_access(
            self, file_size: int = 100 * 1024 * 1024,
            chunk_size: int = 1024 * 1024) -> BenchmarkResult:
        """Benchmark sequential file access performance."""
        print("\n=== Sequential Access Benchmark ===")
        print(
            f"Testing sequential access with {file_size} bytes, {chunk_size} byte chunks..."
        )

        filename = osp.join(self.test_dir, "sequential_test")
        data = gen_data(file_size)

        # Write test
        write_start = time.time()
        fsutil.write_binary(filename, data)
        write_duration = time.time() - write_start

        # Read test
        read_start = time.time()
        fsutil.read_binary(filename)  # Read but don't store data
        read_duration = time.time() - read_start

        # Clean up
        fsutil.rm(filename, ignore_errors=True)

        write_bandwidth = self._calculate_bandwidth(file_size, write_duration)
        read_bandwidth = self._calculate_bandwidth(file_size, read_duration)

        result = BenchmarkResult(
            operation="sequential_access", file_size=file_size,
            duration=write_duration + read_duration,
            bandwidth_mbps=(write_bandwidth + read_bandwidth) / 2,
            latency_ms=self._calculate_latency(write_duration + read_duration))

        print(f"  Write: {write_bandwidth:.2f} MB/s")
        print(f"  Read: {read_bandwidth:.2f} MB/s")
        print(f"  Average: {result.bandwidth_mbps:.2f} MB/s")

        self.results.append(result)
        return result

    def benchmark_random_access(self, file_size: int = 100 * 1024 * 1024,
                                num_accesses: int = 1000) -> BenchmarkResult:
        """Benchmark random file access performance."""
        print("\n=== Random Access Benchmark ===")
        print(
            f"Testing random access with {file_size} bytes, {num_accesses} accesses..."
        )

        filename = osp.join(self.test_dir, "random_test")
        data = gen_data(file_size)

        # Create file
        fsutil.write_binary(filename, data)

        # Random access test
        import random
        access_times = []

        for _ in range(num_accesses):
            offset = random.randint(0, file_size - 1024)
            size = min(1024, file_size - offset)

            start_time = time.time()
            with open(filename, 'rb') as f:
                f.seek(offset)
                f.read(size)
            end_time = time.time()
            access_times.append(end_time - start_time)

        # Clean up
        fsutil.rm(filename, ignore_errors=True)

        avg_access_time = statistics.mean(access_times)
        result = BenchmarkResult(
            operation="random_access",
            file_size=file_size,
            duration=avg_access_time,
            bandwidth_mbps=0,  # Not applicable for random access
            latency_ms=self._calculate_latency(avg_access_time),
            iterations=num_accesses)

        print(f"  Average access time: {result.latency_ms:.2f} ms")

        self.results.append(result)
        return result

    def run_full_benchmark(self, iterations: int = 3) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        print(f"Starting local filesystem benchmark on {self.mount_point}")
        print(f"Test directory: {self.test_dir}")

        # Define file sizes to test (from small to large)
        small_scales = [10, 12, 14, 16]  # 1KB to 64KB
        medium_scales = [18, 20, 22]  # 256KB to 4MB
        large_scales = [24, 26, 28]  # 16MB to 256MB
        small_sizes = [2**x for x in small_scales]
        medium_sizes = [2**x for x in medium_scales]
        large_sizes = [2**x for x in large_scales]

        # Run all benchmarks
        self.benchmark_write_latency(small_sizes, iterations)
        self.benchmark_read_latency(small_sizes, iterations)
        self.benchmark_write_bandwidth(medium_sizes + large_sizes, iterations)
        self.benchmark_read_bandwidth(medium_sizes + large_sizes, iterations)
        self.benchmark_sequential_access()
        self.benchmark_random_access()

        # Generate summary
        summary = self.generate_summary()

        print("\n=== Benchmark Complete ===")
        print(f"Results saved to: {self.mount_point}/benchmark_results.json")

        return summary

    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all benchmark results."""
        summary = {
            "mount_point": self.mount_point,
            "timestamp": time.time(),
            "results": []
        }

        # Group results by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        # Calculate statistics for each operation
        for operation, results in operations.items():
            if not results:
                continue

            # Calculate bandwidth statistics only for operations with valid bandwidth values
            bandwidth_values = [
                r.bandwidth_mbps for r in results if r.bandwidth_mbps > 0
            ]

            op_summary = {
                "operation": operation,
                "count": len(results),
                "file_sizes": [r.file_size for r in results],
                "avg_latency_ms":
                statistics.mean([r.latency_ms for r in results]),
                "min_latency_ms": min([r.latency_ms for r in results]),
                "max_latency_ms": max([r.latency_ms for r in results])
            }

            # Add bandwidth statistics only if we have valid bandwidth values
            if bandwidth_values:
                op_summary.update({
                    "avg_bandwidth_mbps":
                    statistics.mean(bandwidth_values),
                    "min_bandwidth_mbps":
                    min(bandwidth_values),
                    "max_bandwidth_mbps":
                    max(bandwidth_values)
                })
            else:
                op_summary.update({
                    "avg_bandwidth_mbps": None,
                    "min_bandwidth_mbps": None,
                    "max_bandwidth_mbps": None
                })
            summary["results"].append(op_summary)

        # Save results to file
        results_file = osp.join(self.mount_point, "benchmark_results.json")
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def print_summary(self):
        """Print a formatted summary of benchmark results."""
        print(f"\n{'='*60}")
        print("LOCAL FILESYSTEM BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Mount Point: {self.mount_point}")
        print(f"Total Tests: {len(self.results)}")

        # Group by operation
        operations: dict[str, list[BenchmarkResult]] = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)

        for operation, results in operations.items():
            print(f"\n{operation.upper().replace('_', ' ')}:")
            print(
                f"{'Size (bytes)':<15} {'Latency (ms)':<15} {'Bandwidth (MB/s)':<20}"
            )
            print(f"{'-'*50}")

            for result in sorted(results, key=lambda x: x.file_size):
                bandwidth_str = f"{result.bandwidth_mbps:.2f}" if result.bandwidth_mbps > 0 else "N/A"
                print(
                    f"{result.file_size:<15} {result.latency_ms:<15.2f} {bandwidth_str:<20}"
                )


def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="Local filesystem benchmark")
    parser.add_argument(
        "--mount-point", default="/local/user/vol2/bench",
        help="Mount point to benchmark (default: /local/user/vol2/bench)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per test (default: 3)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up test files after benchmark")

    args = parser.parse_args()

    # Check if mount point exists
    if not osp.exists(args.mount_point):
        print(f"Error: Mount point {args.mount_point} does not exist")
        return 1

    # Run benchmark
    benchmark = LocalFSBenchmark(args.mount_point)

    try:
        benchmark.run_full_benchmark(args.iterations)
        benchmark.print_summary()

        if args.cleanup:
            benchmark.cleanup()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        if args.cleanup:
            benchmark.cleanup()
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if args.cleanup:
            benchmark.cleanup()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
