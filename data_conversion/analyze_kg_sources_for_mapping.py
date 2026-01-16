#!/usr/bin/env python3
"""
Analyze kg.csv to identify which sources can be mapped using mondo.obo xrefs

After running extract_all_mondo_xrefs.py, this script shows:
1. Which sources exist in kg.csv
2. Which mapping files are available
3. Potential coverage improvement
"""

import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm

print("="*60)
print("ANALYZING kg.csv SOURCES FOR MAPPING POTENTIAL")
print("="*60)

kg_path = 'primekg_data/kg.csv'
mappings_dir = Path('primekg_data/mappings')

# Step 1: Scan kg.csv for all sources
print("\n1. Scanning kg.csv for sources...")

chunk_size = 500000
source_counts = Counter()
source_id_samples = {}  # source -> list of sample IDs

for chunk in tqdm(pd.read_csv(kg_path, chunksize=chunk_size, low_memory=False), desc="Scanning"):
    # Count x_source
    for source in chunk['x_source'].values:
        source_counts[source] += 1

    # Count y_source
    for source in chunk['y_source'].values:
        source_counts[source] += 1

    # Collect sample IDs (first 5 per source)
    for source in set(chunk['x_source'].values) | set(chunk['y_source'].values):
        if source not in source_id_samples:
            source_id_samples[source] = []

        if len(source_id_samples[source]) < 5:
            # Get sample IDs from x
            x_samples = chunk[chunk['x_source'] == source]['x_id'].head(5 - len(source_id_samples[source])).tolist()
            source_id_samples[source].extend(x_samples)

        if len(source_id_samples[source]) < 5:
            # Get sample IDs from y
            y_samples = chunk[chunk['y_source'] == source]['y_id'].head(5 - len(source_id_samples[source])).tolist()
            source_id_samples[source].extend(y_samples)

print("\n2. Sources in kg.csv (sorted by count):")
print(f"{'Source':<20} {'Count':>12} {'Sample IDs'}")
print("-" * 80)

for source, count in source_counts.most_common(20):
    samples = ', '.join(str(sid) for sid in source_id_samples[source][:3])
    print(f"{source:<20} {count:>12,} {samples}")

# Step 2: Check which mapping files exist
print("\n3. Checking available mapping files...")

if not mappings_dir.exists():
    print(f"   ⚠️  Mappings directory not found: {mappings_dir}")
    print(f"   → Run: python extract_all_mondo_xrefs.py first")
    mapping_files = []
else:
    mapping_files = list(mappings_dir.glob('*_to_umls.csv'))

    if not mapping_files:
        print(f"   ⚠️  No mapping files found in {mappings_dir}")
        print(f"   → Run: python extract_all_mondo_xrefs.py first")
    else:
        print(f"   ✅ Found {len(mapping_files)} mapping files:")

        mapping_stats = {}

        for mapping_file in sorted(mapping_files):
            df = pd.read_csv(mapping_file)
            source_name = mapping_file.stem.replace('_to_umls', '').upper()
            mapping_stats[source_name] = len(df)
            print(f"      {mapping_file.name:<30} {len(df):>8,} mappings")

# Step 3: Estimate potential coverage
print("\n4. Estimating potential coverage improvement...")

if mapping_files:
    print("\n   Sources in kg.csv that have mapping files:")
    print(f"   {'Source':<20} {'Entities in kg.csv':>20} {'Mappings available':>20} {'Potential'}")
    print("-" * 80)

    total_entities = sum(source_counts.values())
    mappable_entities = 0

    for source_name, mapping_count in sorted(mapping_stats.items(), key=lambda x: -source_counts.get(x[0], 0)):
        kg_count = source_counts.get(source_name, 0)

        if kg_count > 0:
            potential = "✅ High" if mapping_count > kg_count * 0.5 else "⚠️  Partial"
            print(f"   {source_name:<20} {kg_count:>20,} {mapping_count:>20,} {potential}")
            mappable_entities += kg_count

    print(f"\n   Total entities in kg.csv: {total_entities:,}")
    print(f"   Entities with mapping files: {mappable_entities:,} ({mappable_entities/total_entities*100:.1f}%)")

    # Check unmapped sources
    print("\n   Major sources WITHOUT mapping files:")
    unmapped_sources = set(source_counts.keys()) - set(mapping_stats.keys())

    for source in sorted(unmapped_sources, key=lambda x: -source_counts[x])[:10]:
        count = source_counts[source]
        if count > 10000:  # Only show significant sources
            print(f"      {source:<20} {count:>12,} entities")

else:
    print("   ⚠️  No mapping files available yet")
    print("   → Run: python extract_all_mondo_xrefs.py to generate mappings")

# Step 4: Recommendations
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if not mapping_files:
    print("""
1. Generate mapping files first:
   python extract_all_mondo_xrefs.py

2. Then re-run this analysis:
   python analyze_kg_sources_for_mapping.py

3. Update converter to use all mapping files
4. Re-run conversion pipeline
""")
else:
    print("""
NEXT STEPS:

1. ✅ Mapping files generated successfully

2. Update primekg_to_umls_triples.py to:
   - Load ALL mapping files (not just mondo_to_umls.csv)
   - Apply mappings for all sources

3. Re-run conversion:
   python primekg_pipeline.py --skip-download --strategy map_all

4. Expected improvement:
   - Current: 32,886 triples (0.4%)
   - With multiple mappings: 300K-500K triples (3-6%)

5. For sources without mappings:
   - Consider using --keep-unmapped flag
   - Or create additional mapping files (DrugBank, NCBI, etc.)
""")

print("="*60)
