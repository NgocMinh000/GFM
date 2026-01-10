#!/usr/bin/env python3
"""
Analyze kg.csv to find potential UMLS mappings for other sources
"""

import pandas as pd

print("="*60)
print("ANALYZING kg.csv FOR UMLS MAPPING OPPORTUNITIES")
print("="*60)

# Read sample to see all columns
print("\n1. Checking kg.csv columns...")
df_sample = pd.read_csv('primekg_data/kg.csv', nrows=100)
print(f"   Columns: {list(df_sample.columns)}")

# Check all sources
print("\n2. Checking all data sources...")
chunk_size = 500000
source_counts = {}

for chunk in pd.read_csv('primekg_data/kg.csv', chunksize=chunk_size):
    for source in chunk['x_source'].unique():
        source_counts[source] = source_counts.get(source, 0) + len(chunk[chunk['x_source'] == source])
    for source in chunk['y_source'].unique():
        source_counts[source] = source_counts.get(source, 0) + len(chunk[chunk['y_source'] == source])

print("\n   All sources found:")
for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"     {source:20} {count:>12,}")

# Check if there are any UMLS IDs in the data
print("\n3. Searching for UMLS patterns in IDs...")
df_sample = pd.read_csv('primekg_data/kg.csv', nrows=10000)

# Check x_id and y_id for UMLS patterns (C followed by 7 digits)
x_umls_pattern = df_sample['x_id'].astype(str).str.match(r'^C\d{7}$')
y_umls_pattern = df_sample['y_id'].astype(str).str.match(r'^C\d{7}$')

print(f"   x_id with UMLS pattern (C0000000): {x_umls_pattern.sum()}")
print(f"   y_id with UMLS pattern (C0000000): {y_umls_pattern.sum()}")

if x_umls_pattern.sum() > 0:
    print(f"\n   Sample UMLS IDs found in x_id:")
    umls_samples = df_sample[x_umls_pattern][['x_id', 'x_source', 'x_type', 'x_name']].head(5)
    for _, row in umls_samples.iterrows():
        print(f"     {row['x_id']} | {row['x_source']} | {row['x_type']} | {row['x_name']}")

# Check for potential mapping opportunities
print("\n4. Checking DrugBank, NCBI, HPO, GO sample IDs...")
samples_to_check = {
    'DrugBank': df_sample[df_sample['x_source'] == 'DrugBank']['x_id'].head(5).tolist(),
    'NCBI': df_sample[df_sample['x_source'] == 'NCBI']['x_id'].head(5).tolist(),
    'HPO': df_sample[df_sample['x_source'] == 'HPO']['x_id'].head(5).tolist(),
    'GO': df_sample[df_sample['x_source'] == 'GO']['x_id'].head(5).tolist(),
}

for source, ids in samples_to_check.items():
    if ids:
        print(f"\n   {source} sample IDs:")
        for id in ids:
            print(f"     {id}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

print("""
To map other sources to UMLS, you need mapping files for:

1. DrugBank → UMLS
   - RxNorm provides DrugBank mappings
   - File: drugbank_to_umls.csv

2. NCBI Gene → UMLS
   - UMLS includes NCBI gene IDs
   - File: ncbi_to_umls.csv

3. HPO → UMLS
   - HPO provides UMLS mappings
   - File: hpo_to_umls.csv

4. GO → UMLS
   - UMLS includes GO terms
   - File: go_to_umls.csv

5. UBERON → UMLS
   - May have limited UMLS coverage
   - Need to check UBERON releases

These mapping files typically come from:
- UMLS Metathesaurus (requires license)
- Individual ontology releases
- BioPortal mappings
- PrimeKG processing scripts (if available)

NEXT STEPS:
1. Check if PrimeKG repo has these mapping files
2. Download from ontology sources (e.g., HPO, GO)
3. Or use UMLS Metathesaurus if you have access
""")

print("="*60)
