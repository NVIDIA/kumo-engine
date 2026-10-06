// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const outputTableResponse = [
    {
        "name": "ENTITY",
        "colType": "string",
        "samples": [
            12425, 12425, 12425, 12528, 12528, 12528, 12536, 12536, 12536, 12569
        ]
    },
    {
        "name": "TARGET_PRED",
        "colType": "float",
        "samples": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
];

export const repeatedColumnNameResponse = [
    {
        "name": "ENTITY",
        "colType": "string",
        "samples": [
            12425, 12425, 12425, 12528, 12528, 12528, 12536, 12536, 12536, 12569
        ]
    },
    {
        "name": "ENTITY",
        "colType": "string",
        "samples": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
];

export const repeatedColumnSamplesResponse = [
    {
        "name": "ENTITY",
        "colType": "string",
        "samples": [
            12425, 12425, 12425, 12528, 12528, 12528, 12536, 12536, 12536, 12569
        ]
    },
    {
        "name": "TARGET",
        "colType": "string",
        "samples": [
            12425, 12425, 12425, 12528, 12528, 12528, 12536, 12536, 12536, 12569
        ]
    }
];
