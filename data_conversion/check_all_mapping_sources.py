#!/usr/bin/env python3
"""
Exhaustive check: Test ALL 38 mapping files against ALL sources in kg.csv

This will find unexpected matches where mapping file names don't match kg.csv sources.
For example, MESH mappings might work with CTD source, etc.
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

print("="*80)
print("EXHAUSTIVE MAPPING FILE COMPATIBILITY CHECK")
print("="*80)

kg_path = 'primekg_data/kg.csv'
mappings_dir = Path('primekg_data/mappings')

# Step 1: Collect ALL IDs from kg.csv for each source
print("\n1. Collecting all IDs from kg.csv for each source...")

chunk_size = 500000
source_ids = defaultdict(set)  # source -> set of ALL IDs

for chunk in tqdm(pd.read_csv(kg_path, chunksize=chunk_size, low_memory=False), desc="Scanning kg.csv"):
    for source in chunk['x_source'].unique():
        x_ids = chunk[chunk['x_source'] == source]['x_id'].astype(str)
        source_ids[source].update(x_ids)

    for source in chunk['y_source'].unique():
        y_ids = chunk[chunk['y_source'] == source]['y_id'].astype(str)
        source_ids[source].update(y_ids)

print(f"\nFound {len(source_ids)} sources in kg.csv:")
for source, ids in sorted(source_ids.items(), key=lambda x: -len(x[1])):
    print(f"  {source:<20} {len(ids):>10,} unique IDs")

# Step 2: Load all mapping files
print("\n2. Loading all mapping files...")

mapping_files = sorted(mappings_dir.glob('*_to_umls.csv'))
all_mappings = {}  # mapping_name -> {source_ids_set, df}

for mapping_file in tqdm(mapping_files, desc="Loading mappings"):
    try:
        df = pd.read_csv(mapping_file)
        if len(df) > 0:
            source_col = df.columns[0]
            mapping_ids = set(df[source_col].astype(str))
            all_mappings[mapping_file.stem.replace('_to_umls', '')] = {
                'ids': mapping_ids,
                'df': df,
                'file': mapping_file.name
            }
    except Exception as e:
        print(f"  Error loading {mapping_file.name}: {e}")

print(f"\nLoaded {len(all_mappings)} mapping files")

# Step 3: Test ALL combinations
print("\n3. Testing all mapping file × kg.csv source combinations...")
print("="*80)

results = []

for mapping_name, mapping_data in tqdm(all_mappings.items(), desc="Testing mappings"):
    mapping_ids = mapping_data['ids']
    mapping_file = mapping_data['file']

    print(f"\n📁 {mapping_file} ({len(mapping_ids):,} mappings)")

    best_match = None
    best_match_count = 0
    best_match_source = None

    for kg_source, kg_ids in source_ids.items():
        # Direct match
        direct_matches = len(mapping_ids & kg_ids)

        if direct_matches > 0:
            match_rate = direct_matches / min(len(mapping_ids), len(kg_ids)) * 100
            print(f"  ✅ {kg_source:<20} {direct_matches:>6,} matches ({match_rate:>5.1f}%)")

            if direct_matches > best_match_count:
                best_match_count = direct_matches
                best_match_source = kg_source
                best_match = ('direct', direct_matches, match_rate)

        # Try normalization: strip prefix and leading zeros
        if direct_matches == 0:
            # Normalize both sides
            normalized_mapping = set()
            for mid in mapping_ids:
                # Remove common prefixes
                normalized = str(mid)
                for prefix in ['HP:', 'DOID:', 'MONDO:', 'MESH:', 'NCIT:', 'OMIM:', 'Orphanet:', 'GO:', 'UBERON:', 'DB']:
                    normalized = normalized.replace(prefix, '')
                # Strip leading zeros
                normalized = normalized.lstrip('0') if normalized.lstrip('0') else '0'
                normalized_mapping.add(normalized)

            normalized_kg = set()
            for kid in kg_ids:
                normalized = str(kid)
                for prefix in ['HP:', 'DOID:', 'MONDO:', 'MESH:', 'NCIT:', 'OMIM:', 'Orphanet:', 'GO:', 'UBERON:', 'DB']:
                    normalized = normalized.replace(prefix, '')
                normalized = normalized.lstrip('0') if normalized.lstrip('0') else '0'
                normalized_kg.add(normalized)

            normalized_matches = len(normalized_mapping & normalized_kg)

            if normalized_matches > 0:
                match_rate = normalized_matches / min(len(mapping_ids), len(kg_ids)) * 100
                print(f"  💡 {kg_source:<20} {normalized_matches:>6,} normalized matches ({match_rate:>5.1f}%)")

                if normalized_matches > best_match_count:
                    best_match_count = normalized_matches
                    best_match_source = kg_source
                    best_match = ('normalized', normalized_matches, match_rate)

    if best_match:
        results.append({
            'mapping_file': mapping_file,
            'mapping_name': mapping_name,
            'kg_source': best_match_source,
            'match_type': best_match[0],
            'matches': best_match[1],
            'match_rate': best_match[2],
            'total_mappings': len(mapping_ids)
        })
    else:
        print(f"  ❌ No matches with any kg.csv source")

# Step 4: Summary
print("\n" + "="*80)
print("SUMMARY - COMPATIBLE MAPPINGS")
print("="*80)

if results:
    print(f"\nFound {len(results)} compatible mapping files:\n")
    print(f"{'Mapping File':<30} {'kg.csv Source':<15} {'Type':<12} {'Matches':>10} {'Rate':>8}")
    print("-"*80)

    for r in sorted(results, key=lambda x: -x['matches']):
        print(f"{r['mapping_file']:<30} {r['kg_source']:<15} {r['match_type']:<12} {r['matches']:>10,} {r['match_rate']:>7.1f}%")

    # Group by kg.csv source
    print("\n" + "="*80)
    print("MAPPINGS GROUPED BY kg.csv SOURCE")
    print("="*80)

    by_source = defaultdict(list)
    for r in results:
        by_source[r['kg_source']].append(r)

    for kg_source in sorted(by_source.keys()):
        mappings = by_source[kg_source]
        total_matches = sum(m['matches'] for m in mappings)
        print(f"\n{kg_source} ({len(source_ids[kg_source]):,} IDs in kg.csv):")
        print(f"  ✅ {len(mappings)} mapping file(s) available, {total_matches:,} total matches")

        for m in sorted(mappings, key=lambda x: -x['matches']):
            print(f"     - {m['mapping_file']:<30} {m['matches']:>6,} matches ({m['match_type']})")

    # Calculate potential coverage
    print("\n" + "="*80)
    print("POTENTIAL COVERAGE")
    print("="*80)

    mappable_sources = set(r['kg_source'] for r in results)
    total_entities = sum(len(ids) for ids in source_ids.values())
    mappable_entities = sum(len(source_ids[s]) for s in mappable_sources)

    print(f"\nkg.csv statistics:")
    print(f"  Total entities: {total_entities:,}")
    print(f"  Mappable entities: {mappable_entities:,} ({mappable_entities/total_entities*100:.1f}%)")
    print(f"  Sources with mappings: {len(mappable_sources)}/{len(source_ids)}")

    print(f"\nMappable sources:")
    for source in sorted(mappable_sources):
        count = len(source_ids[source])
        pct = count / total_entities * 100
        print(f"  {source:<20} {count:>10,} entities ({pct:>5.1f}%)")

else:
    print("\n❌ No compatible mappings found!")

print("\n" + "="*80)
