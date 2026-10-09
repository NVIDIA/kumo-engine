# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class CodegenError(Exception):
    pass


class CyclicDependencyError(CodegenError):
    pass


class UnsupportedEntityError(CodegenError):
    pass
