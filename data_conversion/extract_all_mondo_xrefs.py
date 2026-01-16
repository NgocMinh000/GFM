#!/usr/bin/env python3
"""
Extract ALL cross-references from mondo.obo to maximize UMLS mapping coverage

Strategy:
1. Parse mondo.obo for ALL xrefs (MESH, NCIT, DOID, SCTID, etc.)
2. Create transitive mappings: Source → MONDO → UMLS
3. Generate multiple mapping files for different sources

Example:
  MONDO:0000984 has:
    - xref: MESH:D013789
    - xref: NCIT:C35069
    - xref: UMLS:C0039730

  → Create mappings:
    MESH:D013789 → UMLS:C0039730
    NCIT:C35069 → UMLS:C0039730
"""

import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

print("="*60)
print("EXTRACTING ALL CROSS-REFERENCES FROM MONDO.OBO")
print("="*60)

obo_path = 'primekg_data/mondo.obo'
output_dir = Path('primekg_data/mappings')
output_dir.mkdir(exist_ok=True)

print(f"\nInput: {obo_path}")
print(f"Output directory: {output_dir}")

# Parse mondo.obo
print("\nParsing mondo.obo...")

mondo_xrefs = defaultdict(lambda: defaultdict(set))  # mondo_id -> {source -> set(ids)}
current_mondo_id = None
in_term = False
term_count = 0

with open(obo_path, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(tqdm(f, desc="Reading"), 1):
        line = line.strip()

        # New term
        if line == "[Term]":
            in_term = True
            current_mondo_id = None
            term_count += 1
            continue

        # End term
        if not line and in_term:
            in_term = False
            current_mondo_id = None
            continue

        if not in_term:
            continue

        # Get MONDO ID
        if line.startswith("id: MONDO:"):
            current_mondo_id = line.split(": ")[1]
            continue

        # Skip obsolete
        if line.startswith("is_obsolete: true"):
            in_term = False
            current_mondo_id = None
            continue

        # Extract xref
        if current_mondo_id and line.startswith("xref: "):
            # Format: xref: UMLS:C0039730 {source="..."}
            xref = line.split(": ", 1)[1].split()[0]  # Get ID before any metadata

            # Parse xref (format: SOURCE:ID)
            if ":" in xref:
                source, xref_id = xref.split(":", 1)
                mondo_xrefs[current_mondo_id][source].add(xref_id)

print(f"\nParsing complete!")
print(f"  Total terms processed: {term_count:,}")
print(f"  MONDO IDs with xrefs: {len(mondo_xrefs):,}")

# Analyze sources
print(f"\nAnalyzing cross-reference sources...")

all_sources = set()
source_counts = defaultdict(int)

for mondo_id, xrefs in mondo_xrefs.items():
    for source, ids in xrefs.items():
        all_sources.add(source)
        source_counts[source] += len(ids)

print(f"\nFound {len(all_sources)} different sources:")
for source in sorted(source_counts.keys(), key=lambda x: -source_counts[x])[:20]:
    print(f"  {source:20} {source_counts[source]:>8,} references")

# Create transitive mappings: Source → UMLS
print(f"\nCreating transitive mappings (Source → UMLS)...")

source_to_umls = defaultdict(dict)  # source -> {source_id -> umls_cui}

for mondo_id, xrefs in tqdm(mondo_xrefs.items(), desc="Building mappings"):
    # Check if this MONDO has UMLS
    if 'UMLS' not in xrefs:
        continue

    # Get UMLS CUI(s) for this MONDO
    umls_cuis = list(xrefs['UMLS'])
    if not umls_cuis:
        continue

    # Use first UMLS CUI (usually only one)
    umls_cui = umls_cuis[0]

    # Map all other xrefs to this UMLS CUI
    for source, source_ids in xrefs.items():
        if source == 'UMLS':
            continue

        for source_id in source_ids:
            # Store mapping
            source_to_umls[source][source_id] = umls_cui

# Save mapping files
print(f"\nSaving mapping files...")

mapping_stats = {}

for source in sorted(source_to_umls.keys()):
    mappings = source_to_umls[source]

    if len(mappings) == 0:
        continue

    # Create DataFrame
    df = pd.DataFrame([
        {'source_id': sid, 'umls_id': ucui}
        for sid, ucui in mappings.items()
    ])

    # Save to CSV
    output_file = output_dir / f"{source.lower()}_to_umls.csv"
    df.to_csv(output_file, index=False)

    mapping_stats[source] = len(df)
    print(f"  ✓ {output_file.name:30} {len(df):>8,} mappings")

# Also save direct MONDO → UMLS mapping
print(f"\nSaving MONDO → UMLS mapping...")
mondo_umls_mappings = []

for mondo_id, xrefs in mondo_xrefs.items():
    if 'UMLS' in xrefs:
        for umls_cui in xrefs['UMLS']:
            # Normalize MONDO ID (strip prefix and leading zeros for kg.csv)
            mondo_id_normalized = mondo_id.replace('MONDO:', '').lstrip('0')
            if not mondo_id_normalized:
                mondo_id_normalized = '0'

            mondo_umls_mappings.append({
                'mondo_id': mondo_id_normalized,
                'umls_id': umls_cui
            })

df_mondo = pd.DataFrame(mondo_umls_mappings)
mondo_output = output_dir / 'mondo_to_umls.csv'
df_mondo.to_csv(mondo_output, index=False)
print(f"  ✓ {mondo_output.name:30} {len(df_mondo):>8,} mappings")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nExtracted {len(mapping_stats)} source → UMLS mappings:")
for source in sorted(mapping_stats.keys(), key=lambda x: -mapping_stats[x])[:15]:
    print(f"  {source:20} → UMLS: {mapping_stats[source]:>8,} mappings")

print(f"\nTop sources with UMLS mappings:")
top_sources = sorted(mapping_stats.items(), key=lambda x: -x[1])[:10]
for source, count in top_sources:
    print(f"  {source:20} {count:>8,}")

# Recommendations
print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)

print(f"""
Check kg.csv sources and use appropriate mappings:

1. If kg.csv has MESH IDs:
   → Use mesh_to_umls.csv

2. If kg.csv has NCIT IDs:
   → Use ncit_to_umls.csv

3. If kg.csv has DOID IDs:
   → Use doid_to_umls.csv

4. For MONDO (already done):
   → Use mondo_to_umls.csv

Run this to check kg.csv sources:
  python analyze_mapping_opportunities.py

Then update converter to use multiple mapping files!
""")

print("="*60)
