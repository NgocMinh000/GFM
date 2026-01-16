#!/usr/bin/env python3
"""
Debug HPO ID format in kg.csv vs hp_to_umls.csv mapping
"""

import pandas as pd
from tqdm import tqdm

print("="*60)
print("DEBUGGING HPO FORMAT")
print("="*60)

# Step 1: Find HPO IDs in kg.csv
print("\n1. Scanning kg.csv for HPO entities...")

chunk_size = 500000
hpo_ids_sample = set()

for chunk in tqdm(pd.read_csv('primekg_data/kg.csv', chunksize=chunk_size, low_memory=False), desc="Scanning"):
    # Get HPO from x_source
    x_hpo = chunk[chunk['x_source'] == 'HPO']['x_id']
    hpo_ids_sample.update(x_hpo.astype(str).head(50).tolist())

    # Get HPO from y_source
    y_hpo = chunk[chunk['y_source'] == 'HPO']['y_id']
    hpo_ids_sample.update(y_hpo.astype(str).head(50).tolist())

    # Stop after collecting 100 samples
    if len(hpo_ids_sample) >= 100:
        break

print(f"\nFound {len(hpo_ids_sample)} HPO ID samples from kg.csv")
print("Sample kg.csv HPO IDs:")
for i, hpo_id in enumerate(sorted(hpo_ids_sample)[:20]):
    print(f"  {i+1}. {hpo_id}")

# Step 2: Load hp_to_umls.csv
print("\n2. Loading hp_to_umls.csv...")

hp_mapping = pd.read_csv('primekg_data/mappings/hp_to_umls.csv')
print(f"Total mappings: {len(hp_mapping)}")

# Get column names
source_col = hp_mapping.columns[0]
umls_col = hp_mapping.columns[1]

print(f"Columns: {source_col}, {umls_col}")
print("\nSample hp_to_umls.csv IDs:")
for i, hpo_id in enumerate(hp_mapping[source_col].astype(str).head(20)):
    print(f"  {i+1}. {hpo_id}")

# Step 3: Check direct matches
print("\n3. Checking direct matches...")

kg_hpo_set = set(str(id) for id in hpo_ids_sample)
mapping_hpo_set = set(hp_mapping[source_col].astype(str))

direct_matches = kg_hpo_set & mapping_hpo_set
print(f"Direct matches: {len(direct_matches)}")

if direct_matches:
    print("Matched IDs:")
    for match_id in sorted(direct_matches)[:10]:
        print(f"  {match_id}")

# Step 4: Try normalizations
print("\n4. Trying normalizations...")

# Try 1: Remove HP: prefix from mapping
normalized_mapping = hp_mapping[source_col].astype(str).str.replace('HP:', '', regex=False)
matches = len(kg_hpo_set & set(normalized_mapping))
print(f"  Remove 'HP:' from mapping: {matches} matches")

# Try 2: Strip leading zeros from mapping
normalized_mapping = hp_mapping[source_col].astype(str).str.lstrip('0').replace('', '0')
matches = len(kg_hpo_set & set(normalized_mapping))
print(f"  Strip leading zeros from mapping: {matches} matches")

# Try 3: Add HP: prefix to kg.csv
normalized_kg = set('HP:' + str(id) for id in hpo_ids_sample)
matches = len(normalized_kg & mapping_hpo_set)
print(f"  Add 'HP:' to kg.csv: {matches} matches")

# Try 4: Add leading zeros to kg.csv (7 digits)
normalized_kg = set(str(id).zfill(7) for id in hpo_ids_sample)
matches = len(normalized_kg & mapping_hpo_set)
print(f"  Add leading zeros to kg.csv (7 digits): {matches} matches")

# Try 5: Add HP: + leading zeros to kg.csv
normalized_kg = set('HP:' + str(id).zfill(7) for id in hpo_ids_sample)
matches = len(normalized_kg & mapping_hpo_set)
print(f"  Add 'HP:' + leading zeros to kg.csv: {matches} matches")

# Try 6: Both sides - strip everything
kg_stripped = set(str(id).replace('HP:', '').lstrip('0') if str(id).replace('HP:', '').lstrip('0') else '0' for id in hpo_ids_sample)
mapping_stripped = set(str(id).replace('HP:', '').lstrip('0') if str(id).replace('HP:', '').lstrip('0') else '0' for id in hp_mapping[source_col])
matches = len(kg_stripped & mapping_stripped)
print(f"  Strip both sides (prefix + zeros): {matches} matches")

if matches > 0:
    print(f"\n  ✅ SOLUTION FOUND: Strip prefix and leading zeros from both sides")
    print(f"     {matches} IDs can be matched")

    # Show examples
    matched_ids = kg_stripped & mapping_stripped
    print("\n  Example matches:")
    for i, match_id in enumerate(sorted(matched_ids)[:10]):
        # Find original IDs
        orig_kg = [id for id in hpo_ids_sample if str(id).replace('HP:', '').lstrip('0') == match_id or (match_id == '0' and str(id).replace('HP:', '').lstrip('0') == '')]
        orig_map = hp_mapping[hp_mapping[source_col].astype(str).str.replace('HP:', '').str.lstrip('0').replace('', '0') == match_id][source_col].values

        if orig_kg and len(orig_map) > 0:
            print(f"    kg.csv: {orig_kg[0]} | mapping: {orig_map[0]} | normalized: {match_id}")

print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

if matches > 0:
    print(f"""
HPO mapping is VIABLE with normalization!

Normalization rule:
  1. Remove 'HP:' prefix if present
  2. Strip leading zeros
  3. Handle edge case: all zeros → '0'

Expected coverage:
  - {matches} matches found in sample of {len(hpo_ids_sample)} IDs
  - Estimated match rate: {matches/len(hpo_ids_sample)*100:.1f}%
  - With 514,192 HPO entities in kg.csv and 550 mappings:
    → Up to 550 HPO entities can be mapped to UMLS
    → Additional ~500-1000 triples (still small impact)
""")
else:
    print("""
HPO mapping is NOT VIABLE.

ID formats are incompatible even with normalization.
Recommend skipping HPO mapping and using MONDO only.
""")
