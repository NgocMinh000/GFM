#!/usr/bin/env python3
"""
Analyze kg.csv to understand drug and disease feature extraction possibilities
"""

import pandas as pd

print("="*60)
print("ANALYZING kg.csv FOR DRUG & DISEASE FEATURES")
print("="*60)

# Load sample
print("\n1. Loading kg.csv sample...")
df_sample = pd.read_csv('primekg_data/kg.csv', nrows=1000)

print(f"\n2. Columns in kg.csv:")
for i, col in enumerate(df_sample.columns, 1):
    print(f"   {i:2}. {col}")

print(f"\n3. Column descriptions:")
print(f"   x_index:       Row index for x entity")
print(f"   x_id:          Entity ID (MONDO:xxx, DB00001, etc.)")
print(f"   x_type:        Entity type (disease, drug, gene/protein, etc.)")
print(f"   x_name:        Entity name (human-readable)")
print(f"   x_source:      Data source (MONDO, DrugBank, NCBI, etc.)")
print(f"   relation:      Relationship type (e.g., drug_protein)")
print(f"   display_relation: Human-readable relation (e.g., ppi)")
print(f"   y_*:           Same fields for y entity")

# Analyze sources
print(f"\n4. Data sources breakdown:")
print("\n   Scanning full kg.csv for sources...")

chunk_size = 500000
source_entity_counts = {}
source_type_counts = {}

for chunk in pd.read_csv('primekg_data/kg.csv', chunksize=chunk_size, low_memory=False):
    # Count entities by source
    for source in chunk['x_source'].value_counts().items():
        source_entity_counts[source[0]] = source_entity_counts.get(source[0], 0) + source[1]
    for source in chunk['y_source'].value_counts().items():
        source_entity_counts[source[0]] = source_entity_counts.get(source[0], 0) + source[1]

    # Count entity types by source
    for _, row in chunk.iterrows():
        key = (row['x_source'], row['x_type'])
        source_type_counts[key] = source_type_counts.get(key, 0) + 1

        key = (row['y_source'], row['y_type'])
        source_type_counts[key] = source_type_counts.get(key, 0) + 1

print("\n   Sources and entity counts:")
for source, count in sorted(source_entity_counts.items(), key=lambda x: -x[1])[:15]:
    print(f"     {source:20} {count:>12,}")

# Drug features
print(f"\n5. DRUG FEATURES:")
print(f"\n   DrugBank drugs in kg.csv:")

df_drug = pd.read_csv('primekg_data/kg.csv', nrows=100000)
drugbank_drugs = df_drug[df_drug['x_source'] == 'DrugBank'][['x_id', 'x_name']].drop_duplicates().head(10)

print(f"   Sample DrugBank drugs:")
for _, row in drugbank_drugs.iterrows():
    print(f"     {row['x_id']:15} | {row['x_name']}")

print(f"\n   Available drug information in kg.csv:")
print(f"     - Drug ID (e.g., DB00001)")
print(f"     - Drug name (e.g., 'Lepirudin')")
print(f"     - Relationships (drug-drug, drug-protein, drug-disease, etc.)")

print(f"\n   DrugBank provides additional features (NOT in kg.csv):")
print(f"     - Chemical structure (SMILES)")
print(f"     - Molecular weight")
print(f"     - Drug categories")
print(f"     - Pharmacology descriptions")
print(f"     - Side effects")
print(f"     - Indications")

# Disease features
print(f"\n6. DISEASE FEATURES:")
print(f"\n   MONDO diseases in kg.csv:")

mondo_diseases = df_drug[df_drug['x_source'] == 'MONDO'][['x_id', 'x_name']].drop_duplicates().head(10)

print(f"   Sample MONDO diseases:")
for _, row in mondo_diseases.iterrows():
    print(f"     {row['x_id']:15} | {row['x_name']}")

print(f"\n   Available disease information in kg.csv:")
print(f"     - Disease ID (e.g., MONDO:0005148)")
print(f"     - Disease name (e.g., 'type 2 diabetes mellitus')")
print(f"     - Relationships (disease-gene, disease-phenotype, etc.)")

print(f"\n   External sources provide additional features (NOT in kg.csv):")
print(f"     - Clinical descriptions")
print(f"     - ICD codes")
print(f"     - Prevalence")
print(f"     - Symptoms")
print(f"     - Treatment guidelines")

print("\n" + "="*60)
print("FINDINGS")
print("="*60)

print("""
drug_feature.csv và disease_feature.csv là từ:

1. DrugBank XML files (for drugs)
   - Contains chemical structures, molecular properties
   - Pharmacology information
   - NOT extractable from kg.csv

2. Mayo Clinic / UMLS / Orphanet (for diseases)
   - Clinical descriptions
   - Disease classifications
   - NOT extractable from kg.csv

kg.csv chỉ chứa:
  ✅ Entity IDs (DB00001, MONDO:0005148)
  ✅ Entity names
  ✅ Relationships (connections between entities)

  ❌ KHÔNG có detailed features (chemical structures, descriptions, etc.)

SOURCES:
- drug_feature.csv: From DrugBank XML downloads
- disease_feature.csv: From Mayo/UMLS/Orphanet text mining

Harvard Dataverse links:
- Check: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
- May have feature files available for download

CAN WE EXTRACT FROM kg.csv?
  ❌ NO - kg.csv is relationship data only
  ✅ BUT - We can create basic features from kg.csv:
     - Node degrees (how many connections)
     - Entity co-occurrence
     - Network statistics
""")

print("="*60)
print("\nRECOMMENDATIONS:")
print("="*60)

print("""
Option 1: Download from Harvard Dataverse
  - Check if drug_feature.csv and disease_feature.csv available
  - URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM

Option 2: Use kg.csv for graph features (extractable)
  - Node degree
  - Centrality measures
  - Community detection
  - Entity embeddings

Option 3: Download source data and extract features
  - DrugBank: Download XML → parse features
  - UMLS: Clinical descriptions
  - Mayo/Orphanet: Disease info
  (Complex, requires licenses)

Option 4: Use only kg.csv relationships
  - Focus on graph structure
  - No need for external features
  - Sufficient for many use cases
""")

print("="*60)
