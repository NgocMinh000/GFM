#!/usr/bin/env python3
"""
Stage 6: Confidence scoring and final output

Produces final UMLS mappings with confidence scores and tiers
"""

import json
import pickle
from tqdm import tqdm
from pathlib import Path


def load_data():
    """Load required data"""
    print("\nLoading data...")
    output_dir = Path('./outputs')

    # Load Stage 5 results
    with open(output_dir / 'stage5_reranked.json') as f:
        stage5_results = json.load(f)
    print(f"  ✓ Loaded Stage 5 results ({len(stage5_results):,} entities)")

    # Load synonym clusters (for propagation stats)
    with open(output_dir / 'synonym_clusters.json') as f:
        clusters = json.load(f)
    print(f"  ✓ Loaded synonym clusters ({len(clusters):,})")

    # Load UMLS concepts
    with open(output_dir / 'umls_concepts.pkl', 'rb') as f:
        umls_concepts = pickle.load(f)
    print(f"  ✓ Loaded UMLS concepts ({len(umls_concepts):,})")

    return {
        'stage5_results': stage5_results,
        'clusters': clusters,
        'umls_concepts': umls_concepts
    }


def compute_confidence(entity, prediction):
    """
    Compute confidence score for prediction

    Factors:
    - Margin between top-1 and top-2 (35%)
    - Top-1 score (25%)
    - Cluster consensus (25%)
    - Method diversity (15%)

    Returns confidence in [0, 1]
    """
    candidates = prediction['candidates']

    # Need at least 2 candidates for margin
    if len(candidates) < 2:
        return 0.5

    top1_score = candidates[0]['final_score']
    top2_score = candidates[1]['final_score']

    # 1. Margin contribution (35%)
    margin = (top1_score - top2_score) / (top1_score + 1e-10)
    margin_contrib = min(0.35, margin)

    # 2. Score contribution (25%)
    score_contrib = 0.25 * top1_score

    # 3. Consensus contribution (25%)
    consensus = candidates[0].get('cluster_consensus', 0.5)
    consensus_contrib = 0.25 * consensus

    # 4. Method diversity contribution (15%)
    methods = candidates[0].get('methods', [])
    method_diversity = len(set(methods)) / 2  # Normalize to [0, 1]
    method_contrib = 0.15 * method_diversity

    # Total confidence
    confidence = (
        margin_contrib +
        score_contrib +
        consensus_contrib +
        method_contrib
    )

    # Clamp to [0, 1]
    return max(0, min(1, confidence))


def classify_tier(confidence):
    """
    Classify confidence into tier

    - High: >= 0.75
    - Medium: >= 0.50
    - Low: < 0.50
    """
    if confidence >= 0.75:
        return 'high'
    elif confidence >= 0.50:
        return 'medium'
    else:
        return 'low'


def generate_final_mappings(data):
    """
    Generate final UMLS mappings

    Returns list of mappings with confidence scores and tiers
    """
    print("\n" + "=" * 70)
    print("STAGE 6: FINAL OUTPUT")
    print("=" * 70)

    final_mappings = []

    for entity, prediction in tqdm(data['stage5_results'].items(), desc="Stage 6"):
        # Get top-1 prediction
        cui = prediction['top1_cui']
        concept = data['umls_concepts'][cui]

        # Compute confidence
        confidence = compute_confidence(entity, prediction)
        tier = classify_tier(confidence)

        # Build mapping
        mapping = {
            'entity': entity,
            'cui': cui,
            'preferred_name': concept.preferred_name,
            'semantic_types': concept.semantic_types,
            'confidence': round(confidence, 3),
            'confidence_tier': tier,
            'propagated': False,  # Not implemented yet
            'cluster_agreement': prediction['candidates'][0].get('cluster_consensus', 0)
        }

        # Add alternatives for medium/low confidence
        if tier in ['medium', 'low']:
            mapping['alternatives'] = [
                {
                    'cui': c['cui'],
                    'name': c['preferred_name'],
                    'score': round(c['final_score'], 3)
                }
                for c in prediction['candidates'][1:4]
            ]

        final_mappings.append(mapping)

    print(f"  ✓ Generated {len(final_mappings):,} mappings")

    return final_mappings


def save_outputs(final_mappings, data):
    """Save all output files"""
    print("\nSaving outputs...")
    output_dir = Path('./outputs')

    # 1. Save final mappings
    mappings_path = output_dir / 'final_umls_mappings.json'
    with open(mappings_path, 'w') as f:
        json.dump(final_mappings, f, indent=2)
    print(f"  ✓ Saved: {mappings_path}")

    # 2. Save triples
    triples_path = output_dir / 'umls_mapping_triples.txt'
    with open(triples_path, 'w') as f:
        for item in final_mappings:
            f.write(f"{item['entity']} | mapped_to_cui | {item['cui']}\n")
    print(f"  ✓ Saved: {triples_path}")

    # 3. Compute and save statistics
    stats = compute_statistics(final_mappings, data['umls_concepts'])
    stats_path = output_dir / 'mapping_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved: {stats_path}")


def compute_statistics(final_mappings, umls_concepts):
    """Compute mapping statistics"""
    stats = {
        'total_entities': len(final_mappings),
        'by_tier': {},
        'real_umls_coverage': 0,
        'avg_confidence': 0
    }

    # Count by tier
    for tier in ['high', 'medium', 'low']:
        count = sum(1 for m in final_mappings if m['confidence_tier'] == tier)
        percentage = round(count / len(final_mappings) * 100, 1)

        stats['by_tier'][tier] = {
            'count': count,
            'percentage': percentage
        }

    # Real UMLS coverage
    real_cuis = sum(1 for m in final_mappings if m['cui'] in umls_concepts)
    stats['real_umls_coverage'] = real_cuis
    stats['real_umls_coverage_percentage'] = round(
        real_cuis / len(final_mappings) * 100, 1
    )

    # Average confidence
    stats['avg_confidence'] = round(
        sum(m['confidence'] for m in final_mappings) / len(final_mappings),
        3
    )

    return stats


def print_summary(stats):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Total entities: {stats['total_entities']:,}")
    print(f"\nBy confidence tier:")
    print(f"  HIGH:   {stats['by_tier']['high']['count']:,} "
          f"({stats['by_tier']['high']['percentage']}%)")
    print(f"  MEDIUM: {stats['by_tier']['medium']['count']:,} "
          f"({stats['by_tier']['medium']['percentage']}%)")
    print(f"  LOW:    {stats['by_tier']['low']['count']:,} "
          f"({stats['by_tier']['low']['percentage']}%)")
    print(f"\nAverage confidence: {stats['avg_confidence']:.3f}")
    print(f"Real UMLS coverage: {stats['real_umls_coverage']:,} "
          f"({stats['real_umls_coverage_percentage']}%)")


def main():
    print("=" * 70)
    print("STAGE 6: FINAL OUTPUT")
    print("=" * 70)

    # Load data
    data = load_data()

    # Generate final mappings
    final_mappings = generate_final_mappings(data)

    # Save outputs
    save_outputs(final_mappings, data)

    # Compute and print statistics
    with open('./outputs/mapping_statistics.json') as f:
        stats = json.load(f)
    print_summary(stats)

    print("\n" + "=" * 70)
    print("✅ STAGE 6 COMPLETED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
