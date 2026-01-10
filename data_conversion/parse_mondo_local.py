#!/usr/bin/env python3
"""
Parse MONDO .obo locally and create umls_mondo.csv
Run this on your LOCAL machine where mondo.obo is located
"""

import re
import sys
from pathlib import Path

def parse_mondo_obo_simple(obo_path: str, output_path: str):
    """Parse MONDO .obo and extract UMLS mappings"""

    print(f"Reading {obo_path}...")

    mappings = []
    current_mondo_id = None
    in_term = False
    term_count = 0

    with open(obo_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 100000 == 0:
                print(f"  Processed {line_num:,} lines, {len(mappings)} mappings found...")

            line = line.strip()

            # New term
            if line == "[Term]":
                in_term = True
                current_mondo_id = None
                term_count += 1
                continue

            # End term
            if not line and in_term:
                in_term = False
                current_mondo_id = None
                continue

            if not in_term:
                continue

            # Get MONDO ID
            if line.startswith("id: MONDO:"):
                current_mondo_id = line.split(": ")[1]
                continue

            # Skip obsolete
            if line.startswith("is_obsolete: true"):
                in_term = False
                current_mondo_id = None
                continue

            # Extract UMLS from xref
            # Format: xref: UMLS:C0012634 {source="MEDGEN:4347", ...}
            if current_mondo_id and line.startswith("xref: UMLS:"):
                umls_cui = line.split("xref: UMLS:")[1].split()[0]
                mappings.append((current_mondo_id, umls_cui))

    print(f"\nParsing complete!")
    print(f"  Total terms: {term_count:,}")
    print(f"  UMLS mappings: {len(mappings):,}")

    # Deduplicate
    unique_mappings = list(set(mappings))
    print(f"  Unique mappings: {len(unique_mappings):,}")

    # Write CSV
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        f.write("mondo_id,umls_id\n")
        for mondo_id, umls_cui in sorted(unique_mappings):
            f.write(f"{mondo_id},{umls_cui}\n")

    print(f"✅ Done! Created {output_path}")
    print(f"\nSample mappings:")
    for mondo_id, umls_cui in sorted(unique_mappings)[:5]:
        print(f"  {mondo_id} → {umls_cui}")

    return len(unique_mappings)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parse_mondo_local.py <path_to_mondo.obo> [output.csv]")
        print("\nExample:")
        print("  python parse_mondo_local.py mondo.obo umls_mondo.csv")
        sys.exit(1)

    obo_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "umls_mondo.csv"

    if not Path(obo_path).exists():
        print(f"❌ File not found: {obo_path}")
        sys.exit(1)

    try:
        count = parse_mondo_obo_simple(obo_path, output_path)
        print(f"\n{'='*60}")
        print(f"SUCCESS! {count:,} mappings extracted")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Upload {output_path} to server:")
        print(f"   scp {output_path} user@server:/home/user/GFM/data_conversion/primekg_data/")
        print(f"2. Run conversion on server")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
