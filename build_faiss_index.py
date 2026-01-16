#!/usr/bin/env python3
"""
Build FAISS Index for UMLS Hard Negative Mining

This script builds a FAISS index for all UMLS concepts using SapBERT embeddings.
The index is used during training to find hard negatives (similar but incorrect CUIs).

Usage:
    python build_faiss_index.py [--output-dir OUTPUT_DIR] [--use-gpu]

Example:
    python build_faiss_index.py --output-dir tmp/umls_faiss_index --use-gpu
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import faiss
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def build_faiss_index(
    umls_loader,
    output_path: str = "tmp/umls_faiss_index",
    use_gpu: bool = True,
    nlist: int = 100,  # Number of clusters for IVF index
):
    """
    Build FAISS index for UMLS concepts using SapBERT embeddings.

    Args:
        umls_loader: UMLSLoader instance with concepts loaded
        output_path: Directory to save FAISS index
        use_gpu: Whether to use GPU for index building
        nlist: Number of clusters for IVF index (affects speed vs accuracy)
    """
    logger.info("="*80)
    logger.info("Building FAISS Index for UMLS Concepts")
    logger.info("="*80)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all CUIs and preferred names
    logger.info(f"Processing {len(umls_loader.concepts):,} UMLS concepts")
    cuis = list(umls_loader.concepts.keys())
    texts = [umls_loader.concepts[cui].preferred_name for cui in cuis]

    # Encode texts using SapBERT
    logger.info("Encoding texts with SapBERT...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device="cuda" if use_gpu else "cpu"
    )

    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalization
    )

    logger.info(f"Generated embeddings shape: {embeddings.shape}")

    # Build FAISS index
    logger.info("Building FAISS index...")
    dim = embeddings.shape[1]

    # Use IVF index for faster search on large datasets
    # IVF = Inverted File, clusters data for faster approximate search
    quantizer = faiss.IndexFlatIP(dim)  # Inner Product (for normalized vectors = cosine sim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    # Move to GPU if requested
    if use_gpu and faiss.get_num_gpus() > 0:
        logger.info(f"Using GPU (found {faiss.get_num_gpus()} GPU(s))")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # Train index (required for IVF)
    logger.info("Training FAISS index...")
    index.train(embeddings)

    # Add vectors
    logger.info("Adding vectors to index...")
    index.add(embeddings)

    # Move back to CPU for saving
    if use_gpu and faiss.get_num_gpus() > 0:
        index = faiss.index_gpu_to_cpu(index)

    # Save index
    index_file = output_dir / "umls.index"
    logger.info(f"Saving FAISS index to {index_file}")
    faiss.write_index(index, str(index_file))

    # Save CUI list (maps index positions to CUIs)
    cui_list_file = output_dir / "cui_list.txt"
    logger.info(f"Saving CUI list to {cui_list_file}")
    with open(cui_list_file, 'w') as f:
        for cui in cuis:
            f.write(f"{cui}\n")

    logger.info("="*80)
    logger.info("✅ FAISS Index Built Successfully!")
    logger.info("="*80)
    logger.info(f"Index file: {index_file}")
    logger.info(f"CUI list: {cui_list_file}")
    logger.info(f"Total vectors: {index.ntotal:,}")
    logger.info(f"Dimension: {dim}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index for UMLS concepts")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tmp/umls_faiss_index",
        help="Output directory for FAISS index (default: tmp/umls_faiss_index)"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for faster index building"
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=100,
        help="Number of clusters for IVF index (default: 100)"
    )
    args = parser.parse_args()

    # Load UMLS
    logger.info("Loading UMLS data...")
    from gfmrag.umls_mapping.umls_loader import UMLSLoader
    from gfmrag.umls_mapping.config import UMLSMappingConfig

    config = UMLSMappingConfig(
        kg_clean_path='dummy',
        umls_data_dir='data/umls',
        output_root='tmp/umls_training',
        mrconso_path='data/umls/META/MRCONSO.RRF',
        mrsty_path='data/umls/META/MRSTY.RRF',
        umls_cache_dir='data/umls/processed',
    )

    umls_loader = UMLSLoader(config)
    umls_loader.load()
    logger.info(f"Loaded {len(umls_loader.concepts):,} UMLS concepts")

    # Build index
    build_faiss_index(
        umls_loader=umls_loader,
        output_path=args.output_dir,
        use_gpu=args.use_gpu,
        nlist=args.nlist,
    )

    logger.info("\nNext step: Run training")
    logger.info("  python -m gfmrag.umls_mapping.training.cross_encoder_trainer \\")
    logger.info("      --config gfmrag/umls_mapping/training/config/training_config.yaml \\")
    logger.info("      --output_dir models/cross_encoder_finetuned")


if __name__ == "__main__":
    main()
