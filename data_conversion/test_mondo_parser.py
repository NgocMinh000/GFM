#!/usr/bin/env python3
"""
Quick test to verify MONDO .obo file has UMLS cross-references
"""

import sys
import re
from pathlib import Path

def test_mondo_obo(obo_path: str):
    """Test if MONDO .obo has UMLS references"""

    path = Path(obo_path)
    if not path.exists():
        print(f"❌ File not found: {obo_path}")
        return False

    print(f"✓ Reading {obo_path}")
    print(f"  Size: {path.stat().st_size / (1024*1024):.1f} MB")

    umls_refs = []
    current_mondo_id = None
    in_term = False
    term_count = 0

    print("\nParsing...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} lines, found {len(umls_refs)} UMLS refs in {term_count} terms...")

            line = line.strip()

            if line == "[Term]":
                in_term = True
                current_mondo_id = None
                term_count += 1
                continue

            if not line and in_term:
                in_term = False
                current_mondo_id = None
                continue

            if not in_term:
                continue

            # Extract MONDO ID
            if line.startswith("id: MONDO:"):
                current_mondo_id = line.split(": ")[1]
                continue

            # Skip obsolete
            if line.startswith("is_obsolete: true"):
                in_term = False
                current_mondo_id = None
                continue

            # Extract UMLS from xref
            if current_mondo_id and line.startswith("xref: UMLS:"):
                umls_cui = line.split(": ")[1].split()[0].replace("UMLS:", "")
                umls_refs.append((current_mondo_id, umls_cui))

            # Extract UMLS from property_value or closeMatch
            if current_mondo_id and ('umls' in line.lower() or 'UMLS' in line):
                # Try to extract UMLS CUI (format: C followed by digits)
                matches = re.findall(r'C\d{7}', line)
                for cui in matches:
                    umls_refs.append((current_mondo_id, cui))

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total terms: {term_count:,}")
    print(f"UMLS cross-references found: {len(umls_refs):,}")

    if len(umls_refs) > 0:
        # Deduplicate
        unique_refs = list(set(umls_refs))
        print(f"Unique mappings: {len(unique_refs):,}")

        print(f"\nSample mappings (first 10):")
        for mondo_id, umls_cui in unique_refs[:10]:
            print(f"  {mondo_id} → {umls_cui}")

        print(f"\n✅ SUCCESS! MONDO .obo has UMLS mappings")
        return True
    else:
        print(f"\n❌ WARNING: No UMLS references found!")
        print(f"This file may not have UMLS cross-references.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mondo_parser.py <path_to_mondo.obo>")
        print("\nSearching for mondo.obo in common locations...")

        search_paths = [
            "./primekg_data/mondo.obo",
            "./mondo.obo",
            "../mondo.obo",
            "/tmp/mondo.obo"
        ]

        for path in search_paths:
            if Path(path).exists():
                print(f"✓ Found: {path}")
                test_mondo_obo(path)
                sys.exit(0)

        print("❌ No mondo.obo found. Please provide path:")
        print("  python test_mondo_parser.py /path/to/mondo.obo")
        sys.exit(1)

    obo_path = sys.argv[1]
    success = test_mondo_obo(obo_path)
    sys.exit(0 if success else 1)
