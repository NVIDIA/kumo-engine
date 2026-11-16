# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clean openapi.json and write the result to openapi_cleaned.json."""
import json

SOURCE_FILE = 'openapi.json'
TARGET_FILE = 'openapi_cleaned.json'


def clean_openapi_doc(data):
    """
    Recursively clean OpenAPI doc by removing unwanted keys
    and fixing schema issues.
    """
    if isinstance(data, dict):
        # Create new dict without unwanted keys
        cleaned = {}
        for k, v in data.items():
            if not k.startswith('HTTPSTATUS') and k != 'metadata':
                if k in ['exclusiveMinimum', 'exclusiveMaximum']:
                    cleaned[k] = bool(v)
                # Handle array items that are defined as tuple validation
                elif k == 'items' and isinstance(v, list):
                    # Convert tuple validation to single schema using the
                    # first item
                    if v:
                        cleaned[k] = clean_openapi_doc(v[0])
                else:
                    cleaned[k] = clean_openapi_doc(v)
        return cleaned
    elif isinstance(data, list):
        return [clean_openapi_doc(item) for item in data]
    else:
        return data


def main():
    # Read OpenAPI doc
    with open(SOURCE_FILE) as f:
        data = json.load(f)

    # Clean the doc
    cleaned = clean_openapi_doc(data)

    # Write back to file
    with open(TARGET_FILE, 'w') as f:
        json.dump(cleaned, f, indent=4)


if __name__ == '__main__':
    main()
