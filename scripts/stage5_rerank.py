#!/usr/bin/env python3
"""
Stage 5: Cross-encoder reranking

Re-ranks candidates using cross-encoder scoring
Note: Uses weighted combination as cross-encoder approximation
For production, train a proper cross-encoder model
"""

import json
import pickle
from tqdm import tqdm
from pathlib import Path


def load_data():
    """Load required data"""
    print("\nLoading data...")
    output_dir = Path('./outputs')

    # Load Stage 4 results
    with open(output_dir / 'stage4_filtered.json') as f:
        stage4_results = json.load(f)
    print(f"  ✓ Loaded Stage 4 results ({len(stage4_results):,} entities)")

    # Load UMLS concepts
    with open(output_dir / 'umls_concepts.pkl', 'rb') as f:
        umls_concepts = pickle.load(f)
    print(f"  ✓ Loaded UMLS concepts ({len(umls_concepts):,})")

    return {
        'stage4_results': stage4_results,
        'umls_concepts': umls_concepts
    }


def compute_cross_encoder_score(entity, cui, concept, stage4_score, avg_score):
    """
    Compute cross-encoder score

    Note: This is a simplified approximation.
    In production, you would:
    1. Load a trained cross-encoder model (e.g., BiomedNLP-PubMedBERT fine-tuned)
    2. Encode "[entity] [SEP] [cui_name] [SEP] [definition]"
    3. Get classification score

    For now, we use a weighted combination of existing scores
    """
    # Placeholder implementation
    # In production, replace with actual cross-encoder inference:
    #
    # model_name = "path/to/trained/cross-encoder"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #
    # cui_name = concept.preferred_name
    # cui_def = concept.definitions[0] if concept.definitions else ''
    # text = f"{entity} [SEP] {cui_name}"
    # if cui_def:
    #     text += f" [SEP] {cui_def[:100]}"
    #
    # inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     score = torch.sigmoid(outputs.logits[0, 0]).item()
    #
    # return score

    # Approximation: weighted combination
    cross_encoder_score = 0.7 * stage4_score + 0.3 * avg_score

    return cross_encoder_score


def stage5_rerank(data):
    """
    Stage 5: Cross-encoder reranking

    Re-ranks candidates using cross-encoder scores
    """
    print("\n" + "=" * 70)
    print("STAGE 5: CROSS-ENCODER RERANKING")
    print("=" * 70)
    print("\nNote: Using weighted combination as cross-encoder approximation")
    print("For production, train a proper cross-encoder model")

    stage5_results = {}

    for entity, candidates in tqdm(data['stage4_results'].items(), desc="Stage 5"):
        # Compute cross-encoder scores
        for cand in candidates:
            cui = cand['cui']
            concept = data['umls_concepts'][cui]

            # Compute cross-encoder score
            cand['cross_encoder_score'] = compute_cross_encoder_score(
                entity=entity,
                cui=cui,
                concept=concept,
                stage4_score=cand['final_score_stage4'],
                avg_score=cand['avg_score']
            )

            # Final score is cross-encoder score
            cand['final_score'] = cand['cross_encoder_score']

        # Re-rank by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)

        # Store results
        stage5_results[entity] = {
            'top1_cui': candidates[0]['cui'],
            'top1_score': candidates[0]['final_score'],
            'top1_name': candidates[0]['preferred_name'],
            'candidates': candidates
        }

    print(f"  ✓ Re-ranked all entities")

    return stage5_results


def main():
    print("=" * 70)
    print("STAGE 5: CROSS-ENCODER RERANKING")
    print("=" * 70)

    # Load data
    data = load_data()

    # Stage 5: Rerank
    stage5_results = stage5_rerank(data)

    # Save results
    output_path = Path('./outputs/stage5_reranked.json')
    with open(output_path, 'w') as f:
        json.dump(stage5_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("STAGE 5 COMPLETED")
    print("=" * 70)
    print(f"Total entities: {len(stage5_results):,}")
    print(f"Candidates per entity: 32 (re-ranked)")
    print(f"Output: {output_path}")

    # Sample top-1 predictions
    print("\nSample top-1 predictions:")
    sample_entities = list(stage5_results.keys())[:5]
    for entity in sample_entities:
        result = stage5_results[entity]
        print(f"  {entity} → {result['top1_cui']} ({result['top1_name']}) "
              f"[score: {result['top1_score']:.3f}]")

    return 0


if __name__ == "__main__":
    exit(main())
