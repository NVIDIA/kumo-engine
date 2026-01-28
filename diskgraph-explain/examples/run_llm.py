#!/usr/bin/env python3
"""Run prompts through an LLM and collect explanations.

Example usage:
    python run_llm.py --input explanations.parquet --api-key YOUR_KEY --n 5

Requires:
    pip install google-genai pandas pyarrow
"""

import pandas as pd
from google import genai
from google.genai import types
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run explanation prompts through Gemini")
    parser.add_argument('--input', default='explanations.parquet',
                        help='Input parquet with prompt column')
    parser.add_argument('--output', default='explanations_output.txt',
                        help='Output text file')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of samples to process')
    parser.add_argument('--model', default='gemini-2.5-flash-preview-05-20',
                        help='Gemini model to use')
    parser.add_argument('--api-key', required=True,
                        help='Google API key')
    parser.add_argument('--thinking', default='low',
                        choices=['none', 'low', 'medium', 'high'],
                        help='Thinking level for Gemini')
    parser.add_argument('--pop-mean', type=float, default=0.5,
                        help='Population mean prediction for risk categorization')
    args = parser.parse_args()

    # Configure client
    client = genai.Client(api_key=args.api_key)

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} entities")
    print(f"Using model: {args.model}, thinking: {args.thinking}")

    # Check for prompt column
    if 'prompt' not in df.columns:
        print("Error: No 'prompt' column found. Run explain pipeline with prompt_config.")
        return

    # Categorize by risk level (relative to population mean)
    df['risk_level'] = df['prediction'].apply(
        lambda p: 'HIGH' if p > args.pop_mean + 0.15 else (
            'LOW' if p < args.pop_mean - 0.15 else 'MEDIUM')
    )

    # Sample: prioritize variety across risk levels
    high = df[df['risk_level'] == 'HIGH'].head(2)
    medium = df[df['risk_level'] == 'MEDIUM'].head(1)
    low = df[df['risk_level'] == 'LOW'].head(2)
    samples = pd.concat([high, medium, low]).head(args.n)

    # Config with thinking
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=args.thinking)
    )

    results = []
    for idx, row in samples.iterrows():
        entity_id = row['entity_id'][:20]
        prediction = row['prediction']
        risk = row['risk_level']
        prompt = row['prompt']

        print(f"\n{'='*60}")
        print(f"Entity: {entity_id}... | Pred: {prediction*100:.1f}% | Risk: {risk}")
        print(f"{'='*60}")

        try:
            response = client.models.generate_content(
                model=args.model,
                contents=prompt,
                config=config,
            )
            explanation = response.text.strip()
            print(f"\nExplanation:\n{explanation}")
            results.append({
                'entity_id': row['entity_id'],
                'prediction': prediction,
                'risk_level': risk,
                'explanation': explanation
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                'entity_id': row['entity_id'],
                'prediction': prediction,
                'risk_level': risk,
                'explanation': f"ERROR: {e}"
            })

        time.sleep(1)  # Rate limiting

    # Write results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(f"{'='*80}\n")
            f.write(f"Entity: {r['entity_id'][:30]}...\n")
            f.write(f"Prediction: {r['prediction']*100:.1f}% ({r['risk_level']})\n")
            f.write(f"\nExplanation:\n{r['explanation']}\n\n")

    print(f"\nSaved {len(results)} explanations to {args.output}")


if __name__ == '__main__':
    main()
