#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Decodes the arguments of temporal workflows and activities in the web UI.
#
# In the temporal UI, the workflow inputs are base64 encoded and compressed,
# making them difficult to read. Here is an example:
#
#   [
#    {
#      "metadata": {
#        "compression": "enN0ZA==",
#        "encoding": "YmluYXJ5L2NvbXByZXNzZWQ=",
#        "encodingDecoded": "binary/compressed"
#      },
#      "data": "KLUv/WBqAD0HADIOL..."
#    }
#  ]
#
# The inputs can generally be found in the workflow steps such as:
#
#   - ActivityTaskScheduled
#   - WorkflowExecutionStarted
#
# This script automatically decodes the arguments and prints them to the
# terminal. Simply copy the "Input" to the workflow from the UI and run this
# script from your laptop. Does not work on VMs.
#
# Usage:
#
#   ./tools/decode_temporal.py
#

import base64
import json
import string

import zstd
from pandas.io.clipboard import clipboard_get

printable = set(ord(ch) for ch in string.printable)


def sanitize(b: bytes) -> str:
    result = []
    for byte in b:
        if byte in printable:
            result.append(chr(byte))
        else:
            result.append(" ")
    return ''.join(result)


def run():
    user_input = clipboard_get()
    print("Found the following text on the system clipboard")
    print(user_input)
    # Decode only works when user input is a list. When copy from the workflow
    # input and output, it is a dictionary.
    if not user_input.startswith('['):
        user_input = f'[{user_input}]'

    try:
        args = json.loads(user_input)
    except Exception:
        print("Could not decode json from the clipboard. Please "
              "copy the temporal JSON input into the system "
              "clipboard before running this script from your laptop.")
        exit(1)

    for i, arg in enumerate(args):
        if not isinstance(arg, dict):
            d = arg
        else:
            try:
                d = sanitize(zstd.loads(base64.b64decode(arg["data"])))
            except Exception:
                try:
                    d = sanitize(base64.b64decode(arg["data"]))
                except Exception:
                    d = arg["data"]
        print(f"*************** Argument {i}: {d}")


if __name__ == '__main__':
    run()
