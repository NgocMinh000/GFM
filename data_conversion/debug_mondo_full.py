#!/usr/bin/env python3
"""
Full debug script to find MONDO ID format in entire kg.csv
"""

import pandas as pd

print("="*60)
print("DEBUG: Full MONDO ID Format Check")
print("="*60)

# Check umls_mondo.csv first
print("\n1. umls_mondo.csv format:")
df_mapping = pd.read_csv('primekg_data/umls_mondo.csv')
sample_mapping_id = str(df_mapping.iloc[0]['mondo_id'])
print(f"   Sample: {sample_mapping_id}")
print(f"   Format: {'WITH prefix MONDO:' if sample_mapping_id.startswith('MONDO:') else 'WITHOUT prefix'}")

# Scan kg.csv for MONDO
print("\n2. Scanning kg.csv for MONDO entities...")
print("   (This may take a minute...)")

chunk_size = 100000
mondo_x_samples = []
mondo_y_samples = []
total_mondo_x = 0
total_mondo_y = 0

for chunk_num, chunk in enumerate(pd.read_csv('primekg_data/kg.csv', chunksize=chunk_size)):
    # Count MONDO
    mondo_x = chunk[chunk['x_source'] == 'MONDO']
    mondo_y = chunk[chunk['y_source'] == 'MONDO']

    total_mondo_x += len(mondo_x)
    total_mondo_y += len(mondo_y)

    # Collect samples
    if len(mondo_x) > 0 and len(mondo_x_samples) < 5:
        for _, row in mondo_x.head(5).iterrows():
            mondo_x_samples.append(str(row['x_id']))

    if len(mondo_y) > 0 and len(mondo_y_samples) < 5:
        for _, row in mondo_y.head(5).iterrows():
            mondo_y_samples.append(str(row['y_id']))

    if (chunk_num + 1) % 10 == 0:
        print(f"   Processed {(chunk_num + 1) * chunk_size:,} rows... (found {total_mondo_x + total_mondo_y} MONDO so far)")

    # Stop early if we have enough samples
    if len(mondo_x_samples) >= 5 and len(mondo_y_samples) >= 5:
        break

print(f"\n   Total MONDO in x_source: {total_mondo_x:,}")
print(f"   Total MONDO in y_source: {total_mondo_y:,}")

# Show samples
if mondo_x_samples:
    print(f"\n3. Sample MONDO IDs from kg.csv (x_id):")
    for mid in mondo_x_samples[:5]:
        print(f"     '{mid}'")

    sample_kg_id = mondo_x_samples[0]
else:
    print(f"\n3. No MONDO found in x_source")

if mondo_y_samples:
    print(f"\n4. Sample MONDO IDs from kg.csv (y_id):")
    for mid in mondo_y_samples[:5]:
        print(f"     '{mid}'")

    if not mondo_x_samples:
        sample_kg_id = mondo_y_samples[0]
else:
    print(f"\n4. No MONDO found in y_source")

# Compare formats
print("\n" + "="*60)
print("FORMAT COMPARISON")
print("="*60)

if mondo_x_samples or mondo_y_samples:
    sample_kg_id = mondo_x_samples[0] if mondo_x_samples else mondo_y_samples[0]

    print(f"\nkg.csv MONDO ID:       '{sample_kg_id}'")
    print(f"umls_mondo.csv ID:     '{sample_mapping_id}'")

    kg_has_prefix = sample_kg_id.startswith('MONDO:')
    mapping_has_prefix = sample_mapping_id.startswith('MONDO:')

    print(f"\nkg.csv format:         {'WITH prefix' if kg_has_prefix else 'WITHOUT prefix'}")
    print(f"umls_mondo.csv format: {'WITH prefix' if mapping_has_prefix else 'WITHOUT prefix'}")

    if kg_has_prefix == mapping_has_prefix:
        print("\n✅ FORMATS MATCH!")
        print("Issue is elsewhere - need to debug converter logic")
    else:
        print("\n❌ FORMAT MISMATCH DETECTED!")

        if kg_has_prefix and not mapping_has_prefix:
            print("\nSOLUTION: Add 'MONDO:' prefix to umls_mondo.csv IDs")
        elif not kg_has_prefix and mapping_has_prefix:
            print("\nSOLUTION: Remove 'MONDO:' prefix from umls_mondo.csv IDs")
            print("OR: Strip 'MONDO:' from kg.csv IDs in converter")

        # Show exact fix needed
        print(f"\nConverter should normalize:")
        print(f"  kg.csv: {sample_kg_id}")
        print(f"  mapping: {sample_mapping_id}")
        print(f"  → Need to make them match!")
else:
    print("\n❌ NO MONDO ENTITIES FOUND IN kg.csv!")
    print("This is strange given the stats showed 536,698 MONDO entities.")
    print("Need to investigate source column values.")

print("\n" + "="*60)
