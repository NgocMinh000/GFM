#!/usr/bin/env python3
"""
Convert entire kg.csv to triples format
No filtering, no mapping - just pure conversion
"""

import pandas as pd
from tqdm import tqdm
import sys

print("="*60)
print("CONVERTING kg.csv TO TRIPLES FORMAT")
print("="*60)

kg_path = sys.argv[1] if len(sys.argv) > 1 else 'primekg_data/kg.csv'
output_path = sys.argv[2] if len(sys.argv) > 2 else 'primekg_output/kg_triples.txt'

print(f"\nInput:  {kg_path}")
print(f"Output: {output_path}")

# Check which relation column to use
print("\nChecking columns...")
df_sample = pd.read_csv(kg_path, nrows=100)
print(f"Columns: {list(df_sample.columns)}")

# Use display_relation if available, otherwise relation
relation_col = 'display_relation' if 'display_relation' in df_sample.columns else 'relation'
print(f"Using relation column: {relation_col}")

# Convert
print(f"\nConverting {kg_path}...")
print("Reading file...")

chunk_size = 500000
total_rows = 0
triples = []

for chunk in tqdm(pd.read_csv(kg_path, chunksize=chunk_size, low_memory=False), desc="Processing"):
    total_rows += len(chunk)

    for _, row in chunk.iterrows():
        head = str(row['x_id'])
        relation = str(row[relation_col])
        tail = str(row['y_id'])

        triples.append(f"{head},{relation},{tail}")

print(f"\nTotal rows processed: {total_rows:,}")
print(f"Triples generated: {len(triples):,}")

# Write to file
print(f"\nWriting to {output_path}...")
with open(output_path, 'w') as f:
    for triple in tqdm(triples, desc="Writing"):
        f.write(triple + '\n')

print(f"\n✅ Done!")
print(f"\nOutput file: {output_path}")
print(f"Triples: {len(triples):,}")

# Show sample
print(f"\nSample triples (first 10):")
for i, triple in enumerate(triples[:10]):
    print(f"  {triple}")

print("\n" + "="*60)
print(f"✅ SUCCESS! Converted {total_rows:,} rows → {len(triples):,} triples")
print("="*60)
