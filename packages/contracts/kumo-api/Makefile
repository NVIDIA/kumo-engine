# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

install:
	pip install --verbose -e .

package:
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	python setup.py sdist bdist_wheel
