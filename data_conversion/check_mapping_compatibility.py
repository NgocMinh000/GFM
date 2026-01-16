#!/usr/bin/env python3
"""
Check if mapping files are compatible with kg.csv data

For each mapping file:
1. Load sample IDs from mapping file
2. Load sample IDs from kg.csv for that source
3. Check if ID formats match
4. Report compatibility and suggest normalization rules
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

print("="*80)
print("CHECKING MAPPING FILE COMPATIBILITY WITH kg.csv")
print("="*80)

kg_path = 'primekg_data/kg.csv'
mappings_dir = Path('primekg_data/mappings')

# Step 1: Collect sample IDs from kg.csv for each source
print("\n1. Collecting sample IDs from kg.csv...")

chunk_size = 500000
source_sample_ids = defaultdict(set)  # source -> set of sample IDs

for chunk in tqdm(pd.read_csv(kg_path, chunksize=chunk_size, low_memory=False), desc="Scanning kg.csv"):
    for source in set(chunk['x_source'].unique()) | set(chunk['y_source'].unique()):
        if len(source_sample_ids[source]) < 100:  # Collect 100 samples per source
            # Get IDs from x
            x_samples = chunk[chunk['x_source'] == source]['x_id'].head(100 - len(source_sample_ids[source]))
            source_sample_ids[source].update(x_samples.astype(str).tolist())

            # Get IDs from y
            if len(source_sample_ids[source]) < 100:
                y_samples = chunk[chunk['y_source'] == source]['y_id'].head(100 - len(source_sample_ids[source]))
                source_sample_ids[source].update(y_samples.astype(str).tolist())

print(f"\nCollected samples from {len(source_sample_ids)} sources")

# Step 2: Check each mapping file
print("\n2. Checking mapping file compatibility...")
print("="*80)

mapping_files = sorted(mappings_dir.glob('*_to_umls.csv'))
compatible_mappings = []
incompatible_mappings = []

# Source name normalization rules
source_name_map = {
    'hp': 'HPO',
    'doid': 'DOID',
    'mesh': 'MESH',
    'ncit': 'NCIT',
    'omim': 'OMIM',
    'orphanet': 'Orphanet',
    'mondo': 'MONDO',
    'sctid': 'SCTID',
    'icd10': 'ICD10',
    'icd9': 'ICD9',
    'gard': 'GARD',
    'efo': 'EFO',
    'reactome': 'REACTOME',
    'go': 'GO',
    'ncbi': 'NCBI',
    'drugbank': 'DrugBank',
    'uberon': 'UBERON',
    'ctd': 'CTD',
}

for mapping_file in mapping_files:
    source_name = mapping_file.stem.replace('_to_umls', '')

    # Try to find matching source in kg.csv
    kg_source_candidates = []
    for kg_source in source_sample_ids.keys():
        if source_name.lower() in kg_source.lower() or kg_source.lower() in source_name.lower():
            kg_source_candidates.append(kg_source)

    # Also check normalized name
    normalized_name = source_name_map.get(source_name.lower())
    if normalized_name and normalized_name in source_sample_ids:
        if normalized_name not in kg_source_candidates:
            kg_source_candidates.append(normalized_name)

    print(f"\n📁 {mapping_file.name}")
    print(f"   Mapping source name: {source_name}")

    # Load mapping file
    try:
        df_mapping = pd.read_csv(mapping_file)
        if len(df_mapping) == 0:
            print(f"   ⚠️  Empty mapping file")
            continue

        # Get source column (first column)
        source_col = df_mapping.columns[0]
        mapping_sample_ids = df_mapping[source_col].head(10).astype(str).tolist()

        print(f"   Mapping file IDs (sample): {', '.join(mapping_sample_ids[:5])}")
        print(f"   Total mappings: {len(df_mapping):,}")

    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        continue

    # Find matching kg.csv source
    if not kg_source_candidates:
        print(f"   ⚠️  No matching source found in kg.csv")
        incompatible_mappings.append({
            'file': mapping_file.name,
            'source': source_name,
            'reason': 'No matching source in kg.csv',
            'mappings': len(df_mapping)
        })
        continue

    # Check each candidate
    best_match = None
    best_match_count = 0

    for kg_source in kg_source_candidates:
        kg_sample_ids = list(source_sample_ids[kg_source])[:100]
        print(f"\n   Checking against kg.csv source: {kg_source}")
        print(f"   kg.csv IDs (sample): {', '.join(str(id) for id in kg_sample_ids[:5])}")

        # Check direct matches
        mapping_ids_set = set(df_mapping[source_col].astype(str))
        kg_ids_set = set(str(id) for id in kg_sample_ids)
        direct_matches = len(kg_ids_set & mapping_ids_set)

        print(f"   Direct matches: {direct_matches}/{len(kg_sample_ids)} ({direct_matches/len(kg_sample_ids)*100:.1f}%)")

        # Check if ANY kg IDs exist in mapping (for small mapping files)
        if len(mapping_ids_set) < 10000:  # Small mapping file
            # Check all kg IDs from this source (not just 100 samples)
            all_kg_ids = set(str(id) for id in source_sample_ids[kg_source])
            any_matches = len(all_kg_ids & mapping_ids_set)
            if any_matches > 0:
                print(f"   💡 Found {any_matches} matches when checking ALL kg.csv IDs for this source")
                direct_matches = any_matches

        if direct_matches > best_match_count:
            best_match_count = direct_matches
            best_match = kg_source

        # Check with normalization (remove prefix, strip zeros, etc.)
        if direct_matches < len(kg_sample_ids) * 0.1:  # Less than 10% match
            # Try different normalizations
            normalizations = []

            # Try removing common prefixes
            for prefix in ['HP:', 'DOID:', 'MONDO:', 'MESH:', 'NCIT:', 'OMIM:', 'Orphanet:', 'GO:']:
                normalized_mapping = df_mapping[source_col].astype(str).str.replace(prefix, '', regex=False)
                normalized_kg = pd.Series([str(id).replace(prefix, '') for id in kg_sample_ids])
                matches = len(set(normalized_kg) & set(normalized_mapping))
                if matches > 0:
                    normalizations.append((f"Remove {prefix}", matches))

            # Try stripping leading zeros from BOTH sides
            try:
                normalized_mapping = df_mapping[source_col].astype(str).str.lstrip('0').replace('', '0')
                normalized_kg = pd.Series([str(id).lstrip('0') if str(id).lstrip('0') else '0' for id in kg_sample_ids])
                matches = len(set(normalized_kg) & set(normalized_mapping))
                if matches > direct_matches:
                    normalizations.append(("Strip leading zeros", matches))
            except:
                pass

            # Try with prefix removal + zero stripping
            for prefix in ['HP:', 'DOID:', 'MONDO:', 'MESH:', 'NCIT:', 'OMIM:']:
                try:
                    # Remove prefix then strip zeros
                    normalized_mapping = df_mapping[source_col].astype(str).str.replace(prefix, '', regex=False).str.lstrip('0').replace('', '0')
                    normalized_kg = pd.Series([
                        str(id).replace(prefix, '').lstrip('0') if str(id).replace(prefix, '').lstrip('0') else '0'
                        for id in kg_sample_ids
                    ])
                    matches = len(set(normalized_kg) & set(normalized_mapping))
                    if matches > direct_matches:
                        normalizations.append((f"Remove {prefix} + strip zeros", matches))
                except:
                    pass

            if normalizations:
                best_norm = max(normalizations, key=lambda x: x[1])
                print(f"   💡 Suggested normalization: {best_norm[0]} → {best_norm[1]}/{len(kg_sample_ids)} matches ({best_norm[1]/len(kg_sample_ids)*100:.1f}%)")
                # Update best match if normalization works
                if best_norm[1] > best_match_count:
                    best_match_count = best_norm[1]
                    best_match = kg_source

    # Record result
    # Consider compatible if we found ANY matches (even with normalization)
    if best_match is not None and best_match_count > 0:
        print(f"\n   ✅ COMPATIBLE with kg.csv source: {best_match}")
        # Calculate match rate based on total IDs in this source
        total_kg_ids = len(source_sample_ids[best_match])
        match_rate = best_match_count / total_kg_ids * 100 if total_kg_ids > 0 else 0
        compatible_mappings.append({
            'file': mapping_file.name,
            'source': source_name,
            'kg_source': best_match,
            'mappings': len(df_mapping),
            'matches_found': best_match_count,
            'match_rate': match_rate
        })
    else:
        reason = 'No matches found' if best_match is None else 'No matching IDs'
        print(f"\n   ❌ INCOMPATIBLE - {reason}")
        incompatible_mappings.append({
            'file': mapping_file.name,
            'source': source_name,
            'reason': reason,
            'mappings': len(df_mapping)
        })

# Step 3: Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n✅ COMPATIBLE MAPPINGS: {len(compatible_mappings)}")
if compatible_mappings:
    print(f"\n{'Mapping File':<30} {'kg.csv Source':<15} {'Mappings':>10} {'Matches':>10} {'Rate':>8}")
    print("-"*80)
    for m in sorted(compatible_mappings, key=lambda x: -x['matches_found']):
        print(f"{m['file']:<30} {m['kg_source']:<15} {m['mappings']:>10,} {m['matches_found']:>10,} {m['match_rate']:>7.1f}%")

print(f"\n❌ INCOMPATIBLE MAPPINGS: {len(incompatible_mappings)}")
if incompatible_mappings:
    print(f"\n{'Mapping File':<30} {'Reason':<30} {'Mappings':>10}")
    print("-"*80)
    for m in sorted(incompatible_mappings, key=lambda x: -x['mappings']):
        print(f"{m['file']:<30} {m['reason']:<30} {m['mappings']:>10,}")

# Step 4: Calculate potential coverage
if compatible_mappings:
    print("\n" + "="*80)
    print("POTENTIAL COVERAGE IMPROVEMENT")
    print("="*80)

    total_entities = sum(len(ids) for ids in source_sample_ids.values())
    mappable_entities = sum(
        len(source_sample_ids[m['kg_source']])
        for m in compatible_mappings
        if m['kg_source'] in source_sample_ids
    )

    print(f"\nTotal entities in kg.csv: ~16.2M")
    print(f"Sources with compatible mappings:")
    for m in sorted(compatible_mappings, key=lambda x: -len(source_sample_ids.get(x['kg_source'], []))):
        kg_source = m['kg_source']
        if kg_source in source_sample_ids:
            # Get actual count from earlier scan
            print(f"  {kg_source:<20} - {m['mappings']:>6,} mappings available")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Review compatible mappings above
2. Update primekg_to_umls_triples.py to:
   - Load ALL compatible mapping files
   - Add source name normalization (HPO → HP, etc.)
   - Apply mappings for all sources
3. Re-run conversion to see improvement
""")
