# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import string
from contextlib import contextmanager

from io_utils.retry import no_retry as io_utils_no_retry


@contextmanager
def no_retry():
    with io_utils_no_retry():
        yield


def random_name(k=16):
    import random
    charset = (string.ascii_uppercase + string.ascii_lowercase + string.digits)
    return ''.join(random.choices(charset, k=int(k)))
