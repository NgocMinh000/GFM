#!/usr/bin/env python3
"""
Stage 2: Generate 128 candidates per entity

Uses ensemble of SapBERT + TF-IDF with Reciprocal Rank Fusion (RRF)
"""

import pickle
import json
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path


def load_data():
    """Load all required data"""
    print("\nLoading data...")
    output_dir = Path('./outputs')

    # Load entities
    with open(output_dir / 'entities.txt') as f:
        entities = [line.strip() for line in f]
    print(f"  ✓ Loaded {len(entities):,} entities")

    # Load normalized entities
    with open(output_dir / 'normalized_entities.json') as f:
        normalized = json.load(f)
    print(f"  ✓ Loaded normalized entities")

    # Load SapBERT components
    index = faiss.read_index(str(output_dir / "umls_faiss.index"))
    print(f"  ✓ Loaded FAISS index ({index.ntotal:,} vectors)")

    with open(output_dir / 'umls_cui_order.pkl', 'rb') as f:
        cui_order = pickle.load(f)

    with open(output_dir / 'umls_concepts.pkl', 'rb') as f:
        umls_concepts = pickle.load(f)
    print(f"  ✓ Loaded UMLS concepts ({len(umls_concepts):,})")

    # Load TF-IDF components
    with open(output_dir / 'tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open(output_dir / 'tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    print(f"  ✓ Loaded TF-IDF matrix {tfidf_matrix.shape}")

    with open(output_dir / 'all_aliases.pkl', 'rb') as f:
        all_aliases = pickle.load(f)

    with open(output_dir / 'alias_to_cuis.pkl', 'rb') as f:
        alias_to_cuis = pickle.load(f)
    print(f"  ✓ Loaded aliases ({len(all_aliases):,})")

    return {
        'entities': entities,
        'normalized': normalized,
        'index': index,
        'cui_order': cui_order,
        'umls_concepts': umls_concepts,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'all_aliases': all_aliases,
        'alias_to_cuis': alias_to_cuis
    }


def load_sapbert_model():
    """Load SapBERT model for encoding"""
    print("\nLoading SapBERT model...")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded on {device}")

    return tokenizer, model, device


def encode_text(text, tokenizer, model, device):
    """Encode text using SapBERT"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]
    emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()[0]


def retrieve_sapbert(entity, data, tokenizer, model, device, k=64):
    """Retrieve top-k candidates using SapBERT + FAISS"""
    # Encode entity
    emb = encode_text(entity, tokenizer, model, device)
    emb = emb.reshape(1, -1).astype('float32')
    faiss.normalize_L2(emb)

    # Search FAISS index
    scores, indices = data['index'].search(emb, k)

    # Build candidates
    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        cui = data['cui_order'][idx]
        candidates.append({
            'cui': cui,
            'score': float(score),
            'method': 'sapbert',
            'preferred_name': data['umls_concepts'][cui].preferred_name
        })

    return candidates


def retrieve_tfidf(entity, data, k=64):
    """Retrieve top-k candidates using TF-IDF"""
    # Vectorize entity
    entity_vec = data['vectorizer'].transform([entity])

    # Compute similarities
    sims = cosine_similarity(entity_vec, data['tfidf_matrix']).flatten()

    # Get top indices (retrieve more to account for CUI collapsing)
    top_indices = sims.argsort()[-k*5:][::-1]

    # Aggregate by CUI
    cui_scores = {}
    for idx in top_indices:
        alias = data['all_aliases'][idx]
        score = sims[idx]

        if score < 0.3:  # Skip low scores
            break

        for cui in data['alias_to_cuis'][alias]:
            if cui not in cui_scores:
                cui_scores[cui] = []
            cui_scores[cui].append(score)

    # Build candidates
    candidates = []
    for cui, scores in cui_scores.items():
        candidates.append({
            'cui': cui,
            'score': max(scores),  # Take best score
            'method': 'tfidf',
            'preferred_name': data['umls_concepts'][cui].preferred_name
        })

    # Sort and take top-k
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:k]


def ensemble_fusion(entity, data, tokenizer, model, device, k=128):
    """
    Ensemble SapBERT + TF-IDF using Reciprocal Rank Fusion

    Returns top-k candidates
    """
    # Get expanded entity (with abbreviations expanded)
    expanded = data['normalized'][entity]['expanded']

    # Retrieve from both methods
    cands_sapbert = retrieve_sapbert(expanded, data, tokenizer, model, device, k=64)
    cands_tfidf = retrieve_tfidf(expanded, data, k=64)

    # Build CUI data structure
    cui_data = {}
    k_const = 60  # RRF constant

    # Process SapBERT candidates
    for rank, cand in enumerate(cands_sapbert):
        cui = cand['cui']
        if cui not in cui_data:
            cui_data[cui] = {
                'cui': cui,
                'methods': [],
                'ranks': [],
                'scores': []
            }
        cui_data[cui]['methods'].append('sapbert')
        cui_data[cui]['ranks'].append(rank)
        cui_data[cui]['scores'].append(cand['score'])

    # Process TF-IDF candidates
    for rank, cand in enumerate(cands_tfidf):
        cui = cand['cui']
        if cui not in cui_data:
            cui_data[cui] = {
                'cui': cui,
                'methods': [],
                'ranks': [],
                'scores': []
            }
        cui_data[cui]['methods'].append('tfidf')
        cui_data[cui]['ranks'].append(rank)
        cui_data[cui]['scores'].append(cand['score'])

    # Compute RRF scores
    for cui, cdata in cui_data.items():
        # RRF score: sum of 1/(k + rank)
        rrf = sum(1.0 / (k_const + r) for r in cdata['ranks'])

        # Diversity bonus: both methods agree
        diversity = len(set(cdata['methods'])) * 0.05

        cdata['rrf_score'] = rrf + diversity
        cdata['avg_score'] = sum(cdata['scores']) / len(cdata['scores'])

    # Sort by RRF score
    sorted_cands = sorted(cui_data.values(), key=lambda x: x['rrf_score'], reverse=True)

    # Build final candidates
    final = []
    for cand in sorted_cands[:k]:
        cui = cand['cui']
        concept = data['umls_concepts'][cui]

        final.append({
            'cui': cui,
            'rrf_score': cand['rrf_score'],
            'methods': cand['methods'],
            'avg_score': cand['avg_score'],
            'preferred_name': concept.preferred_name,
            'semantic_types': concept.semantic_types
        })

    return final


def main():
    print("=" * 70)
    print("STAGE 2: CANDIDATE GENERATION")
    print("=" * 70)

    # Load data
    data = load_data()

    # Load SapBERT model
    tokenizer, model, device = load_sapbert_model()

    # Process all entities
    print("\nGenerating candidates...")
    stage2_results = {}

    for entity in tqdm(data['entities'], desc="Stage 2"):
        candidates = ensemble_fusion(
            entity, data, tokenizer, model, device, k=128
        )
        stage2_results[entity] = candidates

    # Save results
    output_path = Path('./outputs/stage2_candidates.json')
    with open(output_path, 'w') as f:
        json.dump(stage2_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETED")
    print("=" * 70)
    print(f"Total entities: {len(stage2_results):,}")
    print(f"Candidates per entity: 128")
    print(f"Output: {output_path}")

    # Sample statistics
    total_cands = sum(len(cands) for cands in stage2_results.values())
    avg_cands = total_cands / len(stage2_results)
    print(f"Average candidates per entity: {avg_cands:.1f}")

    return 0


if __name__ == "__main__":
    exit(main())
