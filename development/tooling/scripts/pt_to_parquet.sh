#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Check if the right number of arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <s3_prefix> <val_output.pt/test_output.pt> <output_file_name.parquet>"
  exit 1
fi

# Assign command line arguments to meaningful variable names
s3_prefix="$1"
input_file="$2"
output_file="$3"

# Temporary files
temp_pt_file="temp.pt"
temp_parquet_file="temp.parquet"

# Copy file from S3 to temporary file
aws s3 cp "${s3_prefix}${input_file}" "$temp_pt_file"

# Load the versioned EvalOutput archive and convert its frame to Parquet.
python -c \
  "from kumo_core.train.output import EvalOutput; \
output = EvalOutput.load('$temp_pt_file'); \
output.df.to_parquet('$temp_parquet_file')"

# Copy the temporary Parquet file back to S3
aws s3 cp "$temp_parquet_file" "${s3_prefix}${output_file}"

# Clean up temporary files
rm "$temp_parquet_file"
rm "$temp_pt_file"
