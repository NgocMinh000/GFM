#!/usr/bin/env python3
"""
Debug MONDO mapping coverage - tìm hiểu tại sao chỉ 0.4% được map
"""

import pandas as pd
from collections import Counter

print("="*60)
print("DEBUG: MONDO MAPPING COVERAGE ANALYSIS")
print("="*60)

# Load mapping file
print("\n1. Loading umls_mondo.csv...")
df_mapping = pd.read_csv('primekg_data/umls_mondo.csv')

# Normalize MONDO IDs (strip prefix and leading zeros)
df_mapping['mondo_id'] = df_mapping['mondo_id'].str.replace('MONDO:', '', regex=False).str.lstrip('0').replace('', '0')

print(f"   Total mappings: {len(df_mapping):,}")
print(f"   Unique MONDO IDs in mapping: {df_mapping['mondo_id'].nunique():,}")
print(f"   Unique UMLS CUIs in mapping: {df_mapping['umls_id'].nunique():,}")

# Create mapping dict
mondo_to_umls = dict(zip(df_mapping['mondo_id'], df_mapping['umls_id']))

# Scan kg.csv
print("\n2. Scanning kg.csv for MONDO entities...")

chunk_size = 500000
total_rows = 0
mondo_x_count = 0
mondo_y_count = 0
mondo_x_mapped_count = 0
mondo_y_mapped_count = 0
both_mapped_count = 0

mondo_x_ids = set()
mondo_y_ids = set()
mondo_x_unmapped = []
mondo_y_unmapped = []

for chunk_num, chunk in enumerate(pd.read_csv('primekg_data/kg.csv', chunksize=chunk_size, low_memory=False)):
    total_rows += len(chunk)

    # Count MONDO
    mondo_x = chunk[chunk['x_source'] == 'MONDO']
    mondo_y = chunk[chunk['y_source'] == 'MONDO']

    mondo_x_count += len(mondo_x)
    mondo_y_count += len(mondo_y)

    # Collect unique MONDO IDs
    mondo_x_ids.update(mondo_x['x_id'].astype(str).unique())
    mondo_y_ids.update(mondo_y['y_id'].astype(str).unique())

    # Check which are mapped
    for _, row in mondo_x.iterrows():
        x_id = str(row['x_id'])
        if x_id in mondo_to_umls:
            mondo_x_mapped_count += 1
        else:
            if len(mondo_x_unmapped) < 20:
                mondo_x_unmapped.append((x_id, row['x_name']))

    for _, row in mondo_y.iterrows():
        y_id = str(row['y_id'])
        if y_id in mondo_to_umls:
            mondo_y_mapped_count += 1
        else:
            if len(mondo_y_unmapped) < 20:
                mondo_y_unmapped.append((y_id, row['y_name']))

    # Count rows where BOTH x and y are mapped
    for _, row in chunk.iterrows():
        x_source = row['x_source']
        y_source = row['y_source']
        x_id = str(row['x_id'])
        y_id = str(row['y_id'])

        x_mapped = False
        y_mapped = False

        # Check if x is mapped
        if x_source == 'MONDO' and x_id in mondo_to_umls:
            x_mapped = True
        elif x_source == 'UMLS':
            x_mapped = True

        # Check if y is mapped
        if y_source == 'MONDO' and y_id in mondo_to_umls:
            y_mapped = True
        elif y_source == 'UMLS':
            y_mapped = True

        if x_mapped and y_mapped:
            both_mapped_count += 1

    if (chunk_num + 1) % 5 == 0:
        print(f"   Processed {total_rows:,} rows...")

print(f"\n3. MONDO Entity Statistics:")
print(f"   Total rows in kg.csv: {total_rows:,}")
print(f"   Rows with x_source=MONDO: {mondo_x_count:,}")
print(f"   Rows with y_source=MONDO: {mondo_y_count:,}")
print(f"   Total MONDO appearances: {mondo_x_count + mondo_y_count:,}")

print(f"\n4. Unique MONDO IDs in kg.csv:")
print(f"   Unique in x_id: {len(mondo_x_ids):,}")
print(f"   Unique in y_id: {len(mondo_y_ids):,}")
all_mondo_ids = mondo_x_ids | mondo_y_ids
print(f"   Total unique MONDO IDs: {len(all_mondo_ids):,}")

print(f"\n5. Mapping Coverage:")
print(f"   MONDO x entities mapped: {mondo_x_mapped_count:,} / {mondo_x_count:,} ({mondo_x_mapped_count/mondo_x_count*100:.1f}%)")
print(f"   MONDO y entities mapped: {mondo_y_mapped_count:,} / {mondo_y_count:,} ({mondo_y_mapped_count/mondo_y_count*100:.1f}%)")

# Check overlap
mapped_ids_in_kg = all_mondo_ids & set(mondo_to_umls.keys())
unmapped_ids_in_kg = all_mondo_ids - set(mondo_to_umls.keys())

print(f"\n6. ID Overlap Analysis:")
print(f"   MONDO IDs in kg.csv: {len(all_mondo_ids):,}")
print(f"   MONDO IDs in mapping: {len(mondo_to_umls):,}")
print(f"   IDs in BOTH (mapped): {len(mapped_ids_in_kg):,} ({len(mapped_ids_in_kg)/len(all_mondo_ids)*100:.1f}%)")
print(f"   IDs in kg.csv but NOT in mapping: {len(unmapped_ids_in_kg):,} ({len(unmapped_ids_in_kg)/len(all_mondo_ids)*100:.1f}%)")

print(f"\n7. Triple Generation:")
print(f"   Rows where BOTH x and y mapped to UMLS: {both_mapped_count:,}")
print(f"   → This explains why only {both_mapped_count:,} triples generated")

print(f"\n8. Sample unmapped MONDO IDs:")
if mondo_x_unmapped:
    print(f"   From x_id (first 10):")
    for mid, name in mondo_x_unmapped[:10]:
        in_mapping = mid in mondo_to_umls
        print(f"     {mid:15} | {name[:40]:40} | In mapping: {in_mapping}")

if mondo_y_unmapped:
    print(f"\n   From y_id (first 10):")
    for mid, name in mondo_y_unmapped[:10]:
        in_mapping = mid in mondo_to_umls
        print(f"     {mid:15} | {name[:40]:40} | In mapping: {in_mapping}")

print("\n" + "="*60)
print("FINDINGS")
print("="*60)

coverage_pct = len(mapped_ids_in_kg) / len(all_mondo_ids) * 100 if all_mondo_ids else 0

print(f"""
MONDO Coverage: {coverage_pct:.1f}%
- kg.csv has {len(all_mondo_ids):,} unique MONDO IDs
- mapping has {len(mondo_to_umls):,} MONDO→UMLS entries
- Only {len(mapped_ids_in_kg):,} IDs match

Why only {both_mapped_count:,} triples?
- Converter requires BOTH head and tail to be mapped
- If only one side is MONDO→UMLS, the triple is skipped
- Most MONDO entities connect to non-MONDO entities (drugs, genes, etc.)

Example:
  MONDO:8019 (disease) → DrugBank:DB00001 (drug)
  ✅ MONDO:8019 maps to UMLS CUI
  ❌ DrugBank:DB00001 not mapped
  → Triple SKIPPED

SOLUTIONS:
1. Use --keep-unmapped flag to keep partially mapped triples
   → Will use original IDs for unmapped entities

2. Add mappings for other sources (DrugBank, NCBI, etc.)
   → More entities will have UMLS CUIs

3. Change strategy to allow mixed IDs
   → Keep MONDO→UMLS + original IDs for others
""")

print("="*60)
print("\nRECOMMENDATIONS:")
print("="*60)
print("""
Option 1: Keep unmapped entities (Quick fix)
  python primekg_pipeline.py --skip-download --strategy map --keep-unmapped
  → Expected: ~240K triples (10x improvement)
  → Triples will have mixed IDs (UMLS CUIs + original IDs)

Option 2: Improve MONDO mapping coverage
  - Check if mondo.obo has more recent mappings
  - Parse mondo.obo again with more patterns
  - Expected: +5-10% coverage

Option 3: Add HPO/GO mappings first
  - Map phenotypes and gene ontology terms
  - Then MONDO→disease connections will work better
  - Expected: ~300K-500K triples
""")

print("="*60)
