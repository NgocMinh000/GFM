#!/usr/bin/env python3
"""
Final validation for complete UMLS mapping pipeline

Validates all output files and displays final statistics
"""

import json
import os
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


def validate_pipeline():
    """Validate complete pipeline outputs"""
    print("=" * 70)
    print("FINAL PIPELINE VALIDATION")
    print("=" * 70)

    output_dir = Path('./outputs')
    errors = []
    warnings = []

    # Define expected outputs
    outputs = [
        # Stage 1 outputs
        ('entities.txt', 'Stage 1.1', 100 * 1024),
        ('synonym_clusters.json', 'Stage 1.2', 100 * 1024),
        ('normalized_entities.json', 'Stage 1.3', 100 * 1024),
        ('umls_concepts.pkl', 'Stage 1.4', 100 * 1024**2),
        ('umls_aliases.pkl', 'Stage 1.4', 50 * 1024**2),
        ('umls_stats.json', 'Stage 1.4', 1024),

        # Stage 2 setup outputs
        ('umls_embeddings.pkl', 'Stage 2 Setup', 10 * 1024**3),
        ('umls_faiss.index', 'Stage 2 Setup', 10 * 1024**3),
        ('umls_cui_order.pkl', 'Stage 2 Setup', 10 * 1024**2),
        ('tfidf_vectorizer.pkl', 'Stage 2 Setup', 1 * 1024**2),
        ('tfidf_matrix.pkl', 'Stage 2 Setup', 400 * 1024**2),
        ('alias_to_cuis.pkl', 'Stage 2 Setup', 50 * 1024**2),
        ('all_aliases.pkl', 'Stage 2 Setup', 50 * 1024**2),

        # Stage 2-6 outputs
        ('stage2_candidates.json', 'Stage 2', 1 * 1024**2),
        ('stage4_filtered.json', 'Stage 4', 500 * 1024),
        ('stage5_reranked.json', 'Stage 5', 500 * 1024),
        ('final_umls_mappings.json', 'Stage 6', 500 * 1024),
        ('umls_mapping_triples.txt', 'Stage 6', 100 * 1024),
        ('mapping_statistics.json', 'Stage 6', 1024),
    ]

    print("\n1. Checking output files:")
    for filename, stage, min_size in outputs:
        file_path = output_dir / filename

        if file_path.exists():
            size = os.path.getsize(file_path)
            print(f"  ✓ {filename}: {format_size(size)}")

            # Check if size is reasonable (allow 50% variance)
            if size < min_size * 0.5:
                warnings.append(
                    f"{filename} is smaller than expected "
                    f"({format_size(size)} < {format_size(min_size)})"
                )
        else:
            errors.append(f"{filename} not found! ({stage})")
            print(f"  ✗ {filename}: MISSING!")

    # Validate final mappings
    print("\n2. Validating final mappings:")
    try:
        mappings_path = output_dir / 'final_umls_mappings.json'

        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)

            print(f"  ✓ Loaded {len(mappings):,} mappings")

            # Check structure
            required_fields = ['entity', 'cui', 'preferred_name', 'confidence', 'confidence_tier']
            sample = mappings[0] if mappings else {}

            for field in required_fields:
                if field not in sample:
                    errors.append(f"Missing required field: {field}")
                else:
                    print(f"  ✓ Field '{field}' present")

        else:
            errors.append("Final mappings file not found")
    except Exception as e:
        errors.append(f"Failed to validate final mappings: {e}")

    # Load and display statistics
    print("\n3. Mapping statistics:")
    try:
        stats_path = output_dir / 'mapping_statistics.json'

        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            print(f"\n  Total entities mapped: {stats['total_entities']:,}")
            print(f"\n  By confidence tier:")
            for tier, data in stats['by_tier'].items():
                print(f"    {tier.upper()}: {data['count']:,} ({data['percentage']}%)")

            print(f"\n  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Real UMLS coverage: {stats['real_umls_coverage']:,} "
                  f"({stats['real_umls_coverage_percentage']}%)")

            # Check if metrics meet targets
            high_pct = stats['by_tier']['high']['percentage']
            if high_pct < 60:
                warnings.append(
                    f"High confidence rate ({high_pct}%) below target (65-75%)"
                )
            elif high_pct > 80:
                warnings.append(
                    f"High confidence rate ({high_pct}%) suspiciously high (check for overfitting)"
                )
            else:
                print(f"\n  ✓ High confidence rate within target range (65-75%)")

        else:
            errors.append("Statistics file not found")
    except Exception as e:
        errors.append(f"Failed to load statistics: {e}")

    # Validate triples
    print("\n4. Validating triples:")
    try:
        triples_path = output_dir / 'umls_mapping_triples.txt'

        if triples_path.exists():
            with open(triples_path) as f:
                triples = f.readlines()

            print(f"  ✓ {len(triples):,} triples generated")

            # Check format
            if triples:
                sample_triple = triples[0].strip()
                if ' | mapped_to_cui | ' in sample_triple:
                    print(f"  ✓ Triple format correct")
                else:
                    warnings.append(f"Triple format may be incorrect: {sample_triple}")
        else:
            errors.append("Triples file not found")
    except Exception as e:
        errors.append(f"Failed to validate triples: {e}")

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
        print("\n" + "=" * 70)
        print("🎉 PIPELINE HOÀN TẤT!")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(validate_pipeline())
