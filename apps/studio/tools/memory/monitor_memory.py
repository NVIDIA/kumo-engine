# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import psutil
from tabulate import tabulate

from io_utils import fsutil


@dataclass
class MemoryInfo:
    r"""Stores the memory information of a process."""
    time: int
    rss: int = 0
    pss: int = 0
    uss: int = 0
    shared: int = 0
    shared_file: int = 0

    def __str__(self) -> str:
        rss = MemoryInfo.to_human_readable(self.rss)
        pss = MemoryInfo.to_human_readable(self.pss)
        uss = MemoryInfo.to_human_readable(self.uss)
        shared = MemoryInfo.to_human_readable(self.shared)
        shared_file = MemoryInfo.to_human_readable(self.shared_file)

        return (f"time={self.time}, rss={rss}, pss={pss}, uss={uss}, "
                f"shared={shared}, shared_file={shared_file}")

    @staticmethod
    def to_human_readable(size: int) -> str:
        size_ = float(size)
        unit: str = ''
        for unit in ('', 'K', 'M', 'G'):
            if size_ < 1024:
                break
            size_ /= 1024.0
        return f"{size_:.2f}{unit}"  # type: ignore


def get_memory_info(pid: int) -> MemoryInfo:
    r"""Obtains the memory information of process `pid`."""
    out = MemoryInfo(time=get_current_time())
    for mmap in psutil.Process(pid).memory_maps():  # type: ignore
        out.rss += mmap.rss
        out.pss += mmap.pss
        out.uss += mmap.private_clean + mmap.private_dirty
        out.shared += mmap.shared_clean + mmap.shared_dirty

        if mmap.path.startswith('/'):
            # For example, /dev/shm:
            out.shared_file += mmap.shared_clean + mmap.shared_dirty
    return out


def get_python_process_pids() -> List[int]:
    import subprocess
    cmd = "ps -ef | grep python | awk '{print $2} ' "
    return [
        int(p)
        for p in subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  shell=True).communicate()[0].splitlines()
    ]


def get_current_time() -> int:
    return int(time.perf_counter())


class MemoryMonitor:
    LOG_FILE = "monitor_logs.txt"

    def __init__(self, log_dir: str, pids: Optional[List[int]] = None):
        fsutil.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        if pids is None:
            pids = get_python_process_pids()
        pids = list(set(pids))

        print(f"Attempting to monitor {len(pids)} processes.")
        self.data: Dict[int, List[MemoryInfo]] = defaultdict(list)

        # First update throws away bad pids:
        self.update(pids)

    def update(
        self,
        pids: Optional[List[int]] = None,
    ) -> Dict[str, MemoryInfo]:
        pids = pids or []
        out = {}
        for pid in (pids or self.data.keys()):
            try:
                info = get_memory_info(pid)
                self.data[pid].append(info)
                out[pid] = info
            except Exception as e:
                print(f"Could not monitor {pid} due to {str(e)}. Skipping.")
        return out

    def table(self, data: Dict[str, MemoryInfo]) -> str:
        keys = [field.name for field in dataclasses.fields(MemoryInfo(time=0))]
        table = []
        for pid, info in data.items():
            table.append((str(pid), ) + tuple(
                MemoryInfo.to_human_readable(getattr(info, k)) for k in keys))

        return tabulate(table, headers=["PID"] + keys)

    def monitor(self, interval_s: int = 5):
        log_file = os.path.join(self.log_dir, self.LOG_FILE)
        try:
            with open(log_file, 'a') as f:
                while True:
                    tick_info = self.update()
                    print(self.table(tick_info))
                    for pid in tick_info:
                        f.write(f"pid={pid}, {str(tick_info[pid])}\n")
                    time.sleep(interval_s)
        except KeyboardInterrupt:
            print("Monitoring interrupted, plotting outputs...")
            self.plot()

    def plot(self):
        def get_time_series(info_list: List[MemoryInfo],
                            attr: str) -> List[int]:
            return [getattr(info, attr) for info in info_list]

        for pid in self.data:
            info_list: List[MemoryInfo] = self.data[pid]
            print(f"Producing plot for process {pid}, with {len(info_list)} "
                  f"samples.")
            x = get_time_series(info_list, "time")
            plt.plot(x, get_time_series(info_list, "rss"), label="rss")
            plt.plot(x, get_time_series(info_list, "pss"), label="pss")
            plt.plot(x, get_time_series(info_list, "uss"), label="uss")
            plt.plot(x, get_time_series(info_list, "shared"), label="shared")
            plt.plot(x, get_time_series(info_list, "shared_file"),
                     label="shared_file")
            plt.legend()
            plt.savefig(os.path.join(self.log_dir, f"{pid}.png"))
            plt.close()


if __name__ == '__main__':
    log_dir = os.path.join(os.getcwd(), "memory_monitor")
    fsutil.makedirs(log_dir, exist_ok=True)
    monitor = MemoryMonitor(log_dir=log_dir)
    monitor.monitor()
