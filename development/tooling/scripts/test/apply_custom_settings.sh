#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file expects: 1)a string which is a semicolon-separated string of key=value pairs.
# 2) a file name which is the settings.env of a docker container
# It then replaces the key=value pairs in the settings.env. It appends to the end of file
# if the key is not in the settings.env

# Validate arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <semicolon-separated-string> <filename>"
    echo "This file expects: 1)a string which is a semicolon-separated string of key=value pairs."
    echo "2) a file name which is the settings.env of a docker container"
    echo "It then replaces the key=value pairs in the settings.env. It appends to the end of file"
    echo "if the key is not in the settings.env"
    exit 1
fi

input_string="$1"
filename="$2"

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "Error: File $filename not found!"
    exit 1
fi

# Create a temporary file to store the updated content
temp_file=$(mktemp)

# Convert the input string to an associative array for easy lookup
declare -A input_kv
IFS=";" read -ra splits <<< "$input_string"
for split in "${splits[@]}"; do
    key="${split%=*}"
    value="${split#*=}"
    input_kv["$key"]="$value"
done

# Flag to keep track of unmatched keys
declare -A unmatched_keys
for key in "${!input_kv[@]}"; do
    unmatched_keys["$key"]=1
done

# Read the file line by line
while IFS= read -r line || [ -n "$line" ]; do
    key="${line%=*}"

    # If the line's key matches an input key, replace it
    if [[ "${input_kv["$key"]}" ]]; then
        echo "$key=${input_kv["$key"]}" >> "$temp_file"
        # Mark key as matched
        unset unmatched_keys["$key"]
    else
        echo "$line" >> "$temp_file"
    fi
done < "$filename"

# Append unmatched keys to the end of the file
for key in "${!unmatched_keys[@]}"; do
    echo "$key=${input_kv["$key"]}" >> "$temp_file"
done

# Move the temporary file to the original file
mv "$temp_file" "$filename"

echo "$filename updated successfully."
