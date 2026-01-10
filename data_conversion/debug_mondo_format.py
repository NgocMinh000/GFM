#!/usr/bin/env python3
"""
Debug script to check MONDO ID format mismatch
"""

import pandas as pd

print("="*60)
print("DEBUG: MONDO ID Format Check")
print("="*60)

# Check kg.csv MONDO IDs
print("\n1. Checking kg.csv...")
df_kg = pd.read_csv('primekg_data/kg.csv', nrows=10000)

mondo_rows = df_kg[df_kg['x_source'] == 'MONDO']
print(f"   Rows with x_source=MONDO: {len(mondo_rows)}")

if len(mondo_rows) > 0:
    print(f"\n   Sample MONDO IDs from kg.csv (x_id):")
    for idx, row in mondo_rows.head(10).iterrows():
        print(f"     {row['x_id']} (source: {row['x_source']}, type: {row['x_type']})")

mondo_rows_y = df_kg[df_kg['y_source'] == 'MONDO']
if len(mondo_rows_y) > 0:
    print(f"\n   Sample MONDO IDs from kg.csv (y_id):")
    for idx, row in mondo_rows_y.head(5).iterrows():
        print(f"     {row['y_id']} (source: {row['y_source']}, type: {row['y_type']})")

# Check umls_mondo.csv
print("\n2. Checking umls_mondo.csv...")
df_mapping = pd.read_csv('primekg_data/umls_mondo.csv')
print(f"   Total mappings: {len(df_mapping)}")
print(f"   Columns: {list(df_mapping.columns)}")
print(f"\n   Sample mappings:")
for idx, row in df_mapping.head(10).iterrows():
    print(f"     {row['mondo_id']} → {row['umls_id']}")

# Check format difference
print("\n3. Format Analysis:")
if len(mondo_rows) > 0:
    kg_mondo_id = str(mondo_rows.iloc[0]['x_id'])
    mapping_mondo_id = str(df_mapping.iloc[0]['mondo_id'])

    print(f"   kg.csv MONDO ID:      '{kg_mondo_id}'")
    print(f"   umls_mondo.csv ID:    '{mapping_mondo_id}'")

    if kg_mondo_id == mapping_mondo_id:
        print("   ✅ Format MATCHES!")
    else:
        print("   ❌ Format MISMATCH!")

        # Check if kg has prefix
        if kg_mondo_id.startswith("MONDO:"):
            print("   kg.csv has 'MONDO:' prefix")
        else:
            print("   kg.csv does NOT have 'MONDO:' prefix")

        # Check if mapping has prefix
        if mapping_mondo_id.startswith("MONDO:"):
            print("   umls_mondo.csv has 'MONDO:' prefix")
        else:
            print("   umls_mondo.csv does NOT have 'MONDO:' prefix")

print("\n" + "="*60)
print("SOLUTION:")
print("="*60)

if len(mondo_rows) > 0:
    kg_format = "with prefix" if str(mondo_rows.iloc[0]['x_id']).startswith("MONDO:") else "without prefix"
    mapping_format = "with prefix" if str(df_mapping.iloc[0]['mondo_id']).startswith("MONDO:") else "without prefix"

    if kg_format != mapping_format:
        print(f"\nkg.csv uses MONDO IDs {kg_format}")
        print(f"umls_mondo.csv uses MONDO IDs {mapping_format}")
        print(f"\nNeed to normalize format in converter!")
    else:
        print("\nFormats match - need to debug converter logic")
