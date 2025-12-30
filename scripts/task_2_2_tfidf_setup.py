#!/usr/bin/env python3
"""
Task 2.2: TF-IDF Vectorizer Setup (ONE-TIME ONLY)

Creates:
- tfidf_vectorizer.pkl
- tfidf_matrix.pkl (~500 MB)
- alias_to_cuis.pkl
- all_aliases.pkl

Runtime: 10-15 minutes
"""

import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path


def main():
    print("=" * 70)
    print("TASK 2.2: TF-IDF VECTORIZER SETUP")
    print("=" * 70)

    # Output directory
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load UMLS concepts
    print("\n1. Loading UMLS concepts...")
    umls_concepts_path = output_dir / "umls_concepts.pkl"

    if not umls_concepts_path.exists():
        print(f"  ✗ ERROR: {umls_concepts_path} not found!")
        print(f"  Please ensure Stage 1 Task 1.4 has been completed.")
        return 1

    with open(umls_concepts_path, 'rb') as f:
        umls_concepts = pickle.load(f)

    print(f"  ✓ Loaded {len(umls_concepts):,} concepts")

    # Step 2: Collect all aliases
    print("\n2. Collecting aliases...")

    all_aliases = []
    alias_to_cuis = defaultdict(list)

    for cui, concept in tqdm(umls_concepts.items(), desc="Collecting"):
        # Get all aliases for this concept
        aliases = concept.aliases

        for alias in aliases:
            all_aliases.append(alias)
            if cui not in alias_to_cuis[alias]:
                alias_to_cuis[alias].append(cui)

    # Convert defaultdict to regular dict
    alias_to_cuis = dict(alias_to_cuis)

    print(f"  ✓ Collected {len(all_aliases):,} aliases")
    print(f"  ✓ Unique aliases: {len(alias_to_cuis):,}")

    # Step 3: Build TF-IDF vectorizer
    print("\n3. Building TF-IDF vectorizer (10-15 minutes)...")

    vectorizer = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 3),  # Character trigrams
        lowercase=True,
        min_df=2,
        max_features=100000
    )

    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(all_aliases)

    print(f"  ✓ Matrix shape: {tfidf_matrix.shape}")
    print(f"  ✓ Features (character trigrams): {len(vectorizer.get_feature_names_out()):,}")

    # Step 4: Save all outputs
    print("\n4. Saving outputs...")

    # Save vectorizer
    vectorizer_path = output_dir / "tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Saved: {vectorizer_path}")

    # Save matrix
    matrix_path = output_dir / "tfidf_matrix.pkl"
    with open(matrix_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    size_mb = os.path.getsize(matrix_path) / (1024**2)
    print(f"  ✓ Saved: {matrix_path} ({size_mb:.1f} MB)")

    # Save alias_to_cuis mapping
    alias_to_cuis_path = output_dir / "alias_to_cuis.pkl"
    with open(alias_to_cuis_path, 'wb') as f:
        pickle.dump(alias_to_cuis, f)
    print(f"  ✓ Saved: {alias_to_cuis_path}")

    # Save all_aliases list
    all_aliases_path = output_dir / "all_aliases.pkl"
    with open(all_aliases_path, 'wb') as f:
        pickle.dump(all_aliases, f)
    print(f"  ✓ Saved: {all_aliases_path}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ TASK 2.2 COMPLETED - NEVER NEED TO RUN AGAIN!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  • {vectorizer_path}")
    print(f"  • {matrix_path} ({size_mb:.1f} MB)")
    print(f"  • {alias_to_cuis_path}")
    print(f"  • {all_aliases_path}")
    print(f"\nStatistics:")
    print(f"  Total aliases: {len(all_aliases):,}")
    print(f"  Unique aliases: {len(alias_to_cuis):,}")
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")
    print(f"  Character trigrams: {len(vectorizer.get_feature_names_out()):,}")

    return 0


if __name__ == "__main__":
    exit(main())
