#!/usr/bin/env python3
"""
Task 2.1: SapBERT Embeddings Setup (ONE-TIME ONLY)

Creates:
- umls_embeddings.pkl (~12 GB)
- umls_faiss.index (~12 GB)
- umls_cui_order.pkl

Runtime: 2-3 hours on GPU, 4-6 hours on CPU
"""

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import faiss
from pathlib import Path


def encode_text(text, tokenizer, model, device):
    """Encode single text using SapBERT"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64  # As per requirements
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Use CLS token embedding
    emb = outputs.last_hidden_state[:, 0, :]
    # L2 normalization
    emb = F.normalize(emb, p=2, dim=1)

    return emb.cpu().numpy()[0]


def main():
    print("=" * 70)
    print("TASK 2.1: SapBERT EMBEDDINGS SETUP")
    print("=" * 70)

    # Create output directory
    output_dir = Path('./outputs')
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load SapBERT model
    print("\n1. Loading SapBERT model...")
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"  ✓ Model loaded on {device}")

    # Step 2: Load UMLS concepts
    print("\n2. Loading UMLS concepts...")
    umls_concepts_path = output_dir / "umls_concepts.pkl"

    if not umls_concepts_path.exists():
        print(f"  ✗ ERROR: {umls_concepts_path} not found!")
        print(f"  Please ensure Stage 1 Task 1.4 has been completed.")
        return 1

    with open(umls_concepts_path, 'rb') as f:
        umls_concepts = pickle.load(f)

    print(f"  ✓ Loaded {len(umls_concepts):,} concepts")

    # Step 3: Encode all concepts
    print("\n3. Encoding UMLS concepts (2-3 hours on GPU)...")
    print("   ⚠️  This is the longest step - be patient!")

    umls_embeddings = {}
    cuis = list(umls_concepts.keys())

    for cui in tqdm(cuis, desc="Encoding"):
        concept = umls_concepts[cui]
        pref_name = concept.preferred_name

        if pref_name:
            emb = encode_text(pref_name, tokenizer, model, device)
            umls_embeddings[cui] = emb

    print(f"\n  ✓ Encoded {len(umls_embeddings):,} concepts")

    # Step 4: Save embeddings
    print("\n4. Saving embeddings...")
    embeddings_path = output_dir / "umls_embeddings.pkl"

    with open(embeddings_path, 'wb') as f:
        pickle.dump(umls_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_gb = os.path.getsize(embeddings_path) / (1024**3)
    print(f"  ✓ Saved: {embeddings_path}")
    print(f"  ✓ Size: {size_gb:.2f} GB")

    # Step 5: Build FAISS index
    print("\n5. Building FAISS index...")

    # Create cui_order list and embedding matrix
    cui_order = list(umls_embeddings.keys())
    vectors = np.array([umls_embeddings[cui] for cui in cui_order]).astype('float32')

    print(f"  ✓ Vector matrix shape: {vectors.shape}")

    # Normalize vectors (L2 normalization)
    faiss.normalize_L2(vectors)

    # Build FAISS index (Inner Product = cosine similarity after normalization)
    dim = 768  # SapBERT embedding dimension
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    print(f"  ✓ FAISS index built: {index.ntotal:,} vectors")

    # Save FAISS index
    faiss_path = output_dir / "umls_faiss.index"
    faiss.write_index(index, str(faiss_path))

    size_gb = os.path.getsize(faiss_path) / (1024**3)
    print(f"  ✓ Saved: {faiss_path}")
    print(f"  ✓ Size: {size_gb:.2f} GB")

    # Save CUI order
    cui_order_path = output_dir / "umls_cui_order.pkl"
    with open(cui_order_path, 'wb') as f:
        pickle.dump(cui_order, f)

    print(f"  ✓ Saved: {cui_order_path}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ TASK 2.1 COMPLETED - NEVER NEED TO RUN AGAIN!")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  • {embeddings_path} ({size_gb:.2f} GB)")
    print(f"  • {faiss_path}")
    print(f"  • {cui_order_path}")
    print(f"\nTotal concepts encoded: {len(umls_embeddings):,}")
    print(f"FAISS index vectors: {index.ntotal:,}")

    return 0


if __name__ == "__main__":
    exit(main())
