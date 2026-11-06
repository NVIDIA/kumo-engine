# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn prediction explanation prompts into natural-language explanations.
Synthetic data setup (run from the repository root):
  python synthetic-data/engine_other/generate.py --dataset modules --output-dir ./example-data/modules
Use ./example-data/modules as SOURCE_ROOT for connector examples.
For prompts, pass prompts.parquet as --input; for collections, read interests.parquet.
The generator writes local files only.
Synthetic evaluation metrics are demonstration only.
Install: pip install google-genai pandas pyarrow
Set GEMINI_API_KEY and pass --model naming a model enabled for your Google account.
This script makes billed API calls when run; it does not train an Engine model.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from google import genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=5)
    args = parser.parse_args()
    if args.n < 1:
        parser.error('--n must be positive.')
    rows = pd.read_parquet(args.input)
    if 'prompt' not in rows or rows['prompt'].isna().any():
        raise ValueError('Input needs a non-null prompt column.')
    if args.output.exists():
        raise FileExistsError(args.output)
    # Keep source row identity with each returned explanation.
    with genai.Client() as client, args.output.open('x') as output:
        for index, prompt in rows['prompt'].head(args.n).items():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Row {index} needs a nonempty prompt string.')
            response = client.models.generate_content(model=args.model,
                                                      contents=prompt)
            if response.text is None:
                raise RuntimeError(f'No explanation returned for row {index}.')
            record = {'source_row': str(index), 'explanation': response.text}
            output.write(json.dumps(record) + '\n')
            print(json.dumps(record))


if __name__ == '__main__':
    main()
