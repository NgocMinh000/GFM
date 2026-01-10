#!/usr/bin/env python3
"""
MONDO .obo Parser Test Script

Test if mondo.obo exists and can be parsed correctly.
Run this on server after uploading mondo.obo.
"""

import sys
from pathlib import Path

def quick_test_mondo(obo_path: str):
    """Quick test to verify mondo.obo format"""

    path = Path(obo_path)
    if not path.exists():
        print(f"❌ File not found: {obo_path}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"✓ Found mondo.obo: {size_mb:.1f} MB")

    if size_mb < 10:
        print(f"⚠️  Warning: File seems too small (expected ~130-150 MB)")
        if size_mb == 0:
            print(f"❌ File is empty! Please upload mondo.obo properly.")
            return False

    # Test parse first 1000 lines
    print(f"\nTesting first 1000 lines...")

    term_count = 0
    umls_xref_count = 0
    umls_property_count = 0

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break

            line = line.strip()

            if line == "[Term]":
                term_count += 1

            if line.startswith("xref: UMLS:"):
                umls_xref_count += 1

            if "skos:exactMatch UMLS:" in line:
                umls_property_count += 1

    print(f"\nResults from first 1000 lines:")
    print(f"  [Term] blocks: {term_count}")
    print(f"  xref: UMLS: {umls_xref_count}")
    print(f"  skos:exactMatch UMLS: {umls_property_count}")

    if term_count == 0:
        print(f"❌ No [Term] blocks found! File may be corrupted.")
        return False

    if umls_xref_count == 0 and umls_property_count == 0:
        print(f"⚠️  No UMLS references found in first 1000 lines.")
        print(f"   This is normal - UMLS refs may appear later in file.")
    else:
        print(f"✅ File looks good! UMLS cross-references detected.")

    return True

if __name__ == "__main__":
    obo_path = sys.argv[1] if len(sys.argv) > 1 else "./primekg_data/mondo.obo"

    print("="*60)
    print("MONDO .obo Quick Test")
    print("="*60)

    success = quick_test_mondo(obo_path)

    if success:
        print(f"\n✅ Ready to parse! Run:")
        print(f"   python create_umls_mondo_mapping.py")
    else:
        print(f"\n❌ Please fix the issues above before parsing.")

    sys.exit(0 if success else 1)
