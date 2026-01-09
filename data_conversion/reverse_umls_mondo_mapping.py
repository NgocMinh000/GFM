#!/usr/bin/env python3
"""
Reverse UMLS-MONDO mapping

Converts umls_mondo.csv (UMLS→MONDO) to mondo_to_umls.csv (MONDO→UMLS)

Usage:
    python reverse_umls_mondo_mapping.py umls_mondo.csv mondo_to_umls.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

def reverse_mapping(input_path: str, output_path: str):
    """Reverse UMLS-MONDO mapping"""

    print(f"Loading mapping from: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Validate columns
    if 'umls_id' not in df.columns or 'mondo_id' not in df.columns:
        print("Error: File must have 'umls_id' and 'mondo_id' columns")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)

    print(f"Total mappings: {len(df)}")

    # Reverse: swap columns
    df_reversed = df[['mondo_id', 'umls_id']].copy()
    df_reversed.columns = ['mondo_id', 'umls_id']

    # Remove duplicates (many UMLS → one MONDO)
    # Strategy: Keep first occurrence (you can change to keep last, or aggregate)
    original_count = len(df_reversed)
    df_reversed = df_reversed.drop_duplicates(subset=['mondo_id'], keep='first')
    duplicates_removed = original_count - len(df_reversed)

    if duplicates_removed > 0:
        print(f"Warning: Removed {duplicates_removed} duplicate MONDO IDs")
        print(f"         (Many UMLS CUIs map to same MONDO ID)")
        print(f"         Kept first occurrence for each MONDO ID")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_reversed.to_csv(output_path, index=False)

    print(f"\nReversed mapping saved to: {output_path}")
    print(f"Total unique MONDO→UMLS mappings: {len(df_reversed)}")

    # Show preview
    print("\nPreview (first 10):")
    print(df_reversed.head(10).to_string(index=False))

    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print(f"Input (UMLS→MONDO):   {len(df)} mappings")
    print(f"Output (MONDO→UMLS):  {len(df_reversed)} unique mappings")
    print(f"Reduction:            {duplicates_removed} duplicates removed")
    print(f"Compression ratio:    {len(df_reversed)/len(df)*100:.1f}%")
    print("="*60)

    return df_reversed


def main():
    parser = argparse.ArgumentParser(
        description="Reverse UMLS-MONDO mapping file"
    )

    parser.add_argument('input', help='Input umls_mondo.csv file')
    parser.add_argument('output', help='Output mondo_to_umls.csv file')

    args = parser.parse_args()

    try:
        reverse_mapping(args.input, args.output)
        print("\n✅ Conversion complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
