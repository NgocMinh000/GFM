#!/usr/bin/env python3
"""
Validation script for Stage 2 Setup (Tasks 2.1 & 2.2)

Validates all 7 output files from Stage 2 setup:
- Task 2.1: umls_embeddings.pkl, umls_faiss.index, umls_cui_order.pkl
- Task 2.2: tfidf_vectorizer.pkl, tfidf_matrix.pkl, alias_to_cuis.pkl, all_aliases.pkl
"""

import os
import pickle
import sys
from pathlib import Path


def format_size(size_bytes):
    """Format file size in human-readable format"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / 1024:.1f} KB"


def validate_stage2_setup():
    """Validate all Stage 2 setup files"""

    print("=" * 70)
    print("STAGE 2 SETUP VALIDATION")
    print("=" * 70)

    output_dir = Path('./outputs')
    errors = []
    warnings = []

    # Define expected files
    files_to_check = [
        ('umls_embeddings.pkl', 'Task 2.1', 10 * 1024**3),  # ~12 GB
        ('umls_faiss.index', 'Task 2.1', 10 * 1024**3),      # ~12 GB
        ('umls_cui_order.pkl', 'Task 2.1', 10 * 1024**2),    # ~10 MB
        ('tfidf_vectorizer.pkl', 'Task 2.2', 1 * 1024**2),   # ~1 MB
        ('tfidf_matrix.pkl', 'Task 2.2', 400 * 1024**2),     # ~500 MB
        ('alias_to_cuis.pkl', 'Task 2.2', 50 * 1024**2),     # ~50 MB
        ('all_aliases.pkl', 'Task 2.2', 50 * 1024**2),       # ~50 MB
    ]

    print("\n1. Checking file existence and sizes:")
    for filename, task, min_size in files_to_check:
        file_path = output_dir / filename
        if file_path.exists():
            size = os.path.getsize(file_path)
            print(f"  ✓ {filename}: {format_size(size)}")

            # Check if size is reasonable
            if size < min_size * 0.5:  # Allow 50% variance
                warnings.append(f"{filename} is smaller than expected ({format_size(size)} < {format_size(min_size)})")
        else:
            errors.append(f"{filename} not found! ({task})")
            print(f"  ✗ {filename}: MISSING!")

    # Validate FAISS index
    print("\n2. Validating FAISS index:")
    try:
        import faiss
        faiss_path = output_dir / "umls_faiss.index"

        if faiss_path.exists():
            index = faiss.read_index(str(faiss_path))
            print(f"  ✓ FAISS index loaded")
            print(f"  ✓ Total vectors: {index.ntotal:,}")

            if index.ntotal < 4_000_000:
                warnings.append(f"FAISS index has fewer vectors than expected ({index.ntotal:,} < 4,000,000)")
            else:
                print(f"  ✓ Vector count meets requirements (>4M)")
        else:
            errors.append("FAISS index file not found")
    except ImportError:
        warnings.append("faiss-cpu or faiss-gpu not installed, skipping FAISS validation")
    except Exception as e:
        errors.append(f"Failed to load FAISS index: {e}")

    # Validate TF-IDF matrix
    print("\n3. Validating TF-IDF matrix:")
    try:
        tfidf_matrix_path = output_dir / "tfidf_matrix.pkl"

        if tfidf_matrix_path.exists():
            with open(tfidf_matrix_path, 'rb') as f:
                tfidf_matrix = pickle.load(f)

            print(f"  ✓ TF-IDF matrix loaded")
            print(f"  ✓ Matrix shape: {tfidf_matrix.shape}")

            rows, cols = tfidf_matrix.shape
            if rows < 10_000_000:
                warnings.append(f"TF-IDF matrix has fewer rows than expected ({rows:,} < 10,000,000)")
            else:
                print(f"  ✓ Row count meets requirements (>10M)")

            if cols < 80_000 or cols > 100_000:
                warnings.append(f"TF-IDF matrix has unexpected feature count ({cols:,}, expected ~100K)")
            else:
                print(f"  ✓ Feature count is appropriate (~100K)")
        else:
            errors.append("TF-IDF matrix file not found")
    except Exception as e:
        errors.append(f"Failed to load TF-IDF matrix: {e}")

    # Validate embeddings
    print("\n4. Validating UMLS embeddings:")
    try:
        embeddings_path = output_dir / "umls_embeddings.pkl"
        cui_order_path = output_dir / "umls_cui_order.pkl"

        if embeddings_path.exists() and cui_order_path.exists():
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)

            with open(cui_order_path, 'rb') as f:
                cui_order = pickle.load(f)

            print(f"  ✓ Embeddings loaded: {len(embeddings):,} CUIs")
            print(f"  ✓ CUI order loaded: {len(cui_order):,} CUIs")

            if len(embeddings) != len(cui_order):
                errors.append(f"Mismatch between embeddings ({len(embeddings):,}) and cui_order ({len(cui_order):,})")
            else:
                print(f"  ✓ Embeddings and CUI order match")

            if len(embeddings) < 4_000_000:
                warnings.append(f"Fewer embeddings than expected ({len(embeddings):,} < 4,000,000)")
            else:
                print(f"  ✓ Embedding count meets requirements (>4M)")

            # Check embedding dimension
            first_cui = list(embeddings.keys())[0]
            emb_dim = len(embeddings[first_cui])
            if emb_dim != 768:
                errors.append(f"Incorrect embedding dimension ({emb_dim}, expected 768)")
            else:
                print(f"  ✓ Embedding dimension is correct (768)")
        else:
            errors.append("Embeddings or CUI order file not found")
    except Exception as e:
        errors.append(f"Failed to validate embeddings: {e}")

    # Validate alias mappings
    print("\n5. Validating alias mappings:")
    try:
        alias_to_cuis_path = output_dir / "alias_to_cuis.pkl"
        all_aliases_path = output_dir / "all_aliases.pkl"

        if alias_to_cuis_path.exists() and all_aliases_path.exists():
            with open(alias_to_cuis_path, 'rb') as f:
                alias_to_cuis = pickle.load(f)

            with open(all_aliases_path, 'rb') as f:
                all_aliases = pickle.load(f)

            print(f"  ✓ Alias-to-CUI mapping loaded: {len(alias_to_cuis):,} unique aliases")
            print(f"  ✓ All aliases list loaded: {len(all_aliases):,} total aliases")

            if len(all_aliases) < 10_000_000:
                warnings.append(f"Fewer aliases than expected ({len(all_aliases):,} < 10,000,000)")
            else:
                print(f"  ✓ Alias count meets requirements (>10M)")
        else:
            errors.append("Alias mapping files not found")
    except Exception as e:
        errors.append(f"Failed to validate alias mappings: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if warnings:
        print(f"\n⚠️  Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  • {warning}")

    if errors:
        print(f"\n✗ Errors ({len(errors)}):")
        for error in errors:
            print(f"  • {error}")
        print("\n❌ VALIDATION FAILED")
        return 1
    elif warnings:
        print("\n⚠️  VALIDATION PASSED WITH WARNINGS")
        return 0
    else:
        print("\n✅ VALIDATION PASSED - ALL CHECKS SUCCESSFUL!")
        return 0


if __name__ == "__main__":
    sys.exit(validate_stage2_setup())
