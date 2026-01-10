#!/usr/bin/env python3
"""
Test script: Convert PrimeKG to UMLS triples using FILTER strategy
No mapping file needed - just filters entities with source=UMLS
"""

import sys
from pathlib import Path

# Assuming kg.csv is in the location user downloaded it
# If not, update this path
kg_paths = [
    "./primekg_data/kg.csv",
    "../primekg_data/kg.csv",
    "./kg.csv",
    "/home/user/kg.csv",
    "/root/kg.csv"
]

# Find kg.csv
kg_path = None
for path in kg_paths:
    if Path(path).exists():
        kg_path = path
        print(f"✓ Found kg.csv at: {kg_path}")
        break

if not kg_path:
    print("❌ kg.csv not found!")
    print(f"\nSearched locations:")
    for p in kg_paths:
        print(f"  - {p}")
    print(f"\nPlease specify kg.csv location:")
    print(f"  python test_filter_strategy.py /path/to/kg.csv")

    if len(sys.argv) > 1:
        kg_path = sys.argv[1]
        if not Path(kg_path).exists():
            print(f"❌ File not found: {kg_path}")
            sys.exit(1)
    else:
        sys.exit(1)

# Run converter with FILTER strategy (no mapping needed)
output_path = "./test_umls_triples.txt"

print(f"\n{'='*60}")
print(f"Testing FILTER strategy (UMLS only)")
print(f"{'='*60}")
print(f"Input:  {kg_path}")
print(f"Output: {output_path}")
print(f"Strategy: filter (no mapping file needed)")
print(f"")

# Import and run
from primekg_to_umls_triples import PrimeKGToUMLSConverter

converter = PrimeKGToUMLSConverter(
    kg_path=kg_path,
    umls_mondo_path=None,  # No mapping file
    strategy="filter",      # Use filter strategy
    keep_unmapped=False
)

print("Converting...")
converter.convert(output_path)

print(f"\n{'='*60}")
print(f"✅ DONE!")
print(f"{'='*60}")
print(f"\nOutput file: {output_path}")

# Show preview
if Path(output_path).exists():
    print(f"\nPreview (first 10 lines):")
    with open(output_path) as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"  {line.rstrip()}")

    # Count lines
    with open(output_path) as f:
        line_count = sum(1 for _ in f)
    print(f"\nTotal triples: {line_count:,}")
