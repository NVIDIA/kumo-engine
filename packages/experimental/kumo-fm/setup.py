# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pyarrow
from setuptools import Extension, setup

setup(ext_modules=[Extension(
    "kumo.fm._native_json", ["src/kumo/fm/_native_json.cpp"],
    include_dirs=[pyarrow.get_include()], language="c++",
    define_macros=[("Py_LIMITED_API", "0x030B0000")],
    py_limited_api=True,
    extra_compile_args=(["/std:c++17", "/O2"] if os.name == "nt"
                        else ["-std=c++17", "-O3"]),
)], options={"bdist_wheel": {"py_limited_api": "cp311"}})
