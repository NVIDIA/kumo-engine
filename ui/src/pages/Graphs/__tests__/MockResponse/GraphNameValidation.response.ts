// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const invalidName = {
    "is_valid": false,
    "error_msg":
        'Graph identifier may only contain alphanumeric characters and underscores, and must begin with a letter. Got: "<name>"'
};

export const validName = {
    "is_valid": true,
    "error_msg": ""
};

export const alreadyNameExists = {
    "is_valid": false,
    "error_msg":
        "Graph with name <name> already exists. Please choose a different name."
};
