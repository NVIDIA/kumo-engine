# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import urllib.request
from dataclasses import dataclass


@dataclass
class Info:
    instance_id: str
    az: str

    @staticmethod
    def is_on_aws() -> bool:
        try:
            urllib.request.urlopen(
                'http://169.254.169.254/latest/meta-data/instance-id',
                timeout=0.1)
        # Non-AWS networks may reject or reset link-local metadata traffic
        # instead of returning a conventional URL error. Both outcomes mean
        # this host is not an EC2 instance and must not block local devserver
        # startup.
        except OSError:
            return False
        return True

    @staticmethod
    def get_aws_data(name: str) -> str:
        return urllib.request.urlopen(
            f'http://169.254.169.254/latest/meta-data/{name}').read().decode()

    @classmethod
    def get(cls) -> 'Info':
        return cls(
            instance_id=Info.get_aws_data('instance-id'),
            az=Info.get_aws_data('placement/availability-zone')[:-1],
        )
