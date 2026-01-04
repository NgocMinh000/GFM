#!/usr/bin/env python3
"""
Stages 3 & 4: Cluster aggregation and hard negative filtering

Stage 3: Aggregate candidates across synonym clusters
Stage 4: Filter hard negatives (similar names, different CUIs)
"""

import json
import pickle
from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher


def load_data():
    """Load required data"""
    print("\nLoading data...")
    output_dir = Path('./outputs')

    # Load synonym clusters
    with open(output_dir / 'synonym_clusters.json') as f:
        clusters = json.load(f)
    print(f"  ✓ Loaded {len(clusters):,} synonym clusters")

    # Load Stage 2 results
    with open(output_dir / 'stage2_candidates.json') as f:
        stage2_results = json.load(f)
    print(f"  ✓ Loaded Stage 2 results ({len(stage2_results):,} entities)")

    # Load UMLS concepts
    with open(output_dir / 'umls_concepts.pkl', 'rb') as f:
        umls_concepts = pickle.load(f)
    print(f"  ✓ Loaded UMLS concepts ({len(umls_concepts):,})")

    return {
        'clusters': clusters,
        'stage2_results': stage2_results,
        'umls_concepts': umls_concepts
    }


def aggregate_cluster(cluster_entities, stage2_results):
    """
    Aggregate candidates across cluster entities

    Computes cluster consensus and boosts scores
    """
    cui_agg = {}
    cluster_size = len(cluster_entities)

    # Aggregate votes and scores
    for entity in cluster_entities:
        if entity not in stage2_results:
            continue

        for cand in stage2_results[entity]:
            cui = cand['cui']

            if cui not in cui_agg:
                cui_agg[cui] = {
                    'cui': cui,
                    'vote_count': 0,
                    'scores': [],
                    'methods': []
                }

            cui_agg[cui]['vote_count'] += 1
            cui_agg[cui]['scores'].append(cand['avg_score'])
            cui_agg[cui]['methods'].extend(cand['methods'])

    # Compute final scores
    for cui, data in cui_agg.items():
        # Cluster consensus: fraction of cluster members voting for this CUI
        data['cluster_consensus'] = data['vote_count'] / cluster_size

        # Average score
        data['avg_score'] = sum(data['scores']) / len(data['scores'])

        # Method diversity
        method_diversity = len(set(data['methods'])) / 2  # Normalize to [0, 1]

        # Final score: weighted combination
        data['final_score'] = (
            0.5 * data['avg_score'] +
            0.3 * data['cluster_consensus'] +
            0.2 * method_diversity
        )

    # Sort by final score
    return sorted(cui_agg.values(), key=lambda x: x['final_score'], reverse=True)


def stage3_cluster_aggregation(data):
    """
    Stage 3: Cluster aggregation

    Refines candidates using cluster consensus
    Reduces to 64 candidates per entity
    """
    print("\n" + "=" * 70)
    print("STAGE 3: CLUSTER AGGREGATION")
    print("=" * 70)

    stage3_results = {}

    for entity, candidates in tqdm(data['stage2_results'].items(), desc="Stage 3"):
        # Find entity's cluster
        cluster_entities = None
        for cluster_id, members in data['clusters'].items():
            if entity in members:
                cluster_entities = members
                break

        # If not in cluster, treat as singleton
        if not cluster_entities:
            cluster_entities = [entity]

        # Aggregate cluster
        cluster_agg = aggregate_cluster(cluster_entities, data['stage2_results'])

        # Build cluster info lookup
        cluster_info = {c['cui']: c for c in cluster_agg}

        # Refine entity's candidates
        refined = []
        for cand in candidates:
            cui = cand['cui']

            # Add cluster consensus if available
            if cui in cluster_info:
                cand['cluster_consensus'] = cluster_info[cui]['cluster_consensus']
                cand['boosted_score'] = (
                    cand['avg_score'] +
                    0.2 * cluster_info[cui]['cluster_consensus']
                )
            else:
                cand['cluster_consensus'] = 0
                cand['boosted_score'] = cand['avg_score']

            refined.append(cand)

        # Sort and take top 64
        refined.sort(key=lambda x: x['boosted_score'], reverse=True)
        stage3_results[entity] = refined[:64]

    print(f"  ✓ Refined to 64 candidates per entity")

    return stage3_results


def string_similarity(s1, s2):
    """Compute string similarity ratio"""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def stage4_hard_negative_filtering(stage3_results, umls_concepts):
    """
    Stage 4: Hard negative filtering

    Identifies and penalizes hard negatives:
    - Similar preferred names but different CUIs
    - Reduces to 32 candidates per entity
    """
    print("\n" + "=" * 70)
    print("STAGE 4: HARD NEGATIVE FILTERING")
    print("=" * 70)

    stage4_results = {}

    for entity, candidates in tqdm(stage3_results.items(), desc="Stage 4"):
        # Identify hard negatives (among top 32)
        hard_negs = []
        top_cands = candidates[:32]

        for i in range(len(top_cands)):
            for j in range(i + 1, len(top_cands)):
                if top_cands[i]['cui'] != top_cands[j]['cui']:
                    sim = string_similarity(
                        top_cands[i]['preferred_name'],
                        top_cands[j]['preferred_name']
                    )

                    # Similar names = hard negative
                    if sim > 0.7:
                        hard_negs.append((top_cands[i]['cui'], top_cands[j]['cui']))

        # Re-score with hard negative penalty
        for cand in candidates:
            hn_penalty = 0

            for cui1, cui2 in hard_negs:
                if cand['cui'] == cui1 or cand['cui'] == cui2:
                    hn_penalty += 0.05

            # Final score: 90% boosted score - 10% hard negative penalty
            cand['final_score_stage4'] = 0.9 * cand['boosted_score'] - 0.1 * hn_penalty

        # Sort and take top 32
        candidates.sort(key=lambda x: x['final_score_stage4'], reverse=True)
        stage4_results[entity] = candidates[:32]

    print(f"  ✓ Filtered to 32 candidates per entity")

    return stage4_results


def main():
    print("=" * 70)
    print("STAGES 3 & 4: AGGREGATION AND FILTERING")
    print("=" * 70)

    # Load data
    data = load_data()

    # Stage 3: Cluster aggregation
    stage3_results = stage3_cluster_aggregation(data)

    # Stage 4: Hard negative filtering
    stage4_results = stage4_hard_negative_filtering(
        stage3_results,
        data['umls_concepts']
    )

    # Save results
    output_path = Path('./outputs/stage4_filtered.json')
    with open(output_path, 'w') as f:
        json.dump(stage4_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("STAGES 3 & 4 COMPLETED")
    print("=" * 70)
    print(f"Total entities: {len(stage4_results):,}")
    print(f"Candidates per entity after Stage 3: 64")
    print(f"Candidates per entity after Stage 4: 32")
    print(f"Output: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
