#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script reads the existing pyright configs
# and regenerates them with the minimal ignore list

import json
import os
import os.path as osp
import subprocess


def try_current(data: dict) -> bool:
    temp_fname = "tempconfig.json"
    with open(temp_fname, 'w') as f:
        f.write(json.dumps(data, indent=4))
    proc = subprocess.run(['pyright', '-p', temp_fname],
                          stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    os.remove(temp_fname)
    return proc.returncode == 0


def refresh_config(base_config: str):
    print("refreshing", base_config)
    with open(base_config) as f:
        data = json.loads(f.read())

    good_files = set()
    for _ in range(len(data['ignore'])):
        f = data['ignore'][0]
        del data['ignore'][0]
        if '*' not in f:
            good = try_current(data)
            print(f, good)
            if good:
                good_files.add(f)
        data['ignore'].append(f)

    data['ignore'] = [f for f in data['ignore'] if f not in good_files]
    with open(base_config, 'w') as f:
        f.write(json.dumps(data, indent=4))
        f.write('\n')


if __name__ == "__main__":
    print("This script takes several hours, fyi")
    # cd to repo root
    os.chdir(osp.dirname(osp.dirname(osp.abspath(__file__))))
    for base_config in ['pyrightconfig.json', 'precommit-pyrightconfig.json']:
        refresh_config(base_config)
