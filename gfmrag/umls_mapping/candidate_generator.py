"""
Stage 3.2: Candidate Generation
Generates candidate UMLS concepts using SapBERT + TF-IDF ensemble
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import logging
import pickle

from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import UMLSMappingConfig
from .umls_loader import UMLSLoader

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """UMLS candidate with score"""
    cui: str
    name: str
    score: float
    method: str  # 'sapbert', 'tfidf', or 'ensemble'


class CandidateGenerator:
    """
    Generates top-K UMLS candidates using ensemble of:
    1. SapBERT semantic similarity
    2. TF-IDF character n-gram similarity
    3. Reciprocal Rank Fusion
    """

    def __init__(self, config: UMLSMappingConfig, umls_loader: UMLSLoader):
        self.config = config
        self.umls_loader = umls_loader

        # Cache paths
        cache_dir = Path(config.umls_cache_dir)
        self.sapbert_cache = cache_dir / "sapbert_embeddings.pkl"
        self.tfidf_cache = cache_dir / "tfidf_vectorizer.pkl"

        # Initialize models (lazy loading)
        self.sapbert_model = None
        self.sapbert_tokenizer = None
        self.sapbert_embeddings = None

        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

        self.umls_names = None
        self.name_to_cui = None

    def generate_candidates(self, entity: str, k: int = None) -> List[Candidate]:
        """
        Generate top-K UMLS candidates for entity

        Args:
            entity: Input entity text
            k: Number of candidates (default from config)

        Returns:
            List of Candidate objects
        """
        if k is None:
            k = self.config.ensemble_final_k

        # Get candidates from both methods
        sapbert_candidates = self._get_sapbert_candidates(entity, self.config.sapbert_top_k)
        tfidf_candidates = self._get_tfidf_candidates(entity, self.config.sapbert_top_k)

        # Ensemble using Reciprocal Rank Fusion
        ensemble_candidates = self._reciprocal_rank_fusion(
            sapbert_candidates,
            tfidf_candidates,
            k=k
        )

        return ensemble_candidates

    def _get_sapbert_candidates(self, entity: str, k: int) -> List[Candidate]:
        """Get candidates using SapBERT semantic similarity"""

        # Lazy load SapBERT
        if self.sapbert_model is None:
            self._load_sapbert()

        # Encode query entity
        query_emb = self._encode_sapbert([entity])[0]

        # Compute similarities
        similarities = cosine_similarity([query_emb], self.sapbert_embeddings)[0]

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        candidates = []
        for idx in top_k_indices:
            name = self.umls_names[idx]
            score = float(similarities[idx])
            cui = self.name_to_cui[name]

            candidates.append(Candidate(
                cui=cui,
                name=name,
                score=score,
                method='sapbert'
            ))

        return candidates

    def _get_tfidf_candidates(self, entity: str, k: int) -> List[Candidate]:
        """Get candidates using TF-IDF character n-grams"""

        # Lazy load TF-IDF
        if self.tfidf_vectorizer is None:
            self._load_tfidf()

        # Vectorize query
        query_vec = self.tfidf_vectorizer.transform([entity.lower()])

        # Compute similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        candidates = []
        for idx in top_k_indices:
            name = self.umls_names[idx]
            score = float(similarities[idx])
            cui = self.name_to_cui[name]

            candidates.append(Candidate(
                cui=cui,
                name=name,
                score=score,
                method='tfidf'
            ))

        return candidates

    def _reciprocal_rank_fusion(
        self,
        sapbert_candidates: List[Candidate],
        tfidf_candidates: List[Candidate],
        k: int,
        k_constant: int = 60
    ) -> List[Candidate]:
        """
        Combine candidates using Reciprocal Rank Fusion

        RRF(d) = sum_{r in R} 1 / (k + rank_r(d))
        """

        # Build rank dictionaries
        sapbert_ranks = {c.cui: (rank + 1, c.score) for rank, c in enumerate(sapbert_candidates)}
        tfidf_ranks = {c.cui: (rank + 1, c.score) for rank, c in enumerate(tfidf_candidates)}

        # Compute RRF scores
        all_cuis = set(sapbert_ranks.keys()) | set(tfidf_ranks.keys())
        rrf_scores = {}

        for cui in all_cuis:
            rrf_score = 0.0

            if cui in sapbert_ranks:
                rank, _ = sapbert_ranks[cui]
                rrf_score += 1.0 / (k_constant + rank)

            if cui in tfidf_ranks:
                rank, _ = tfidf_ranks[cui]
                rrf_score += 1.0 / (k_constant + rank)

            # Diversity bonus: if both methods agree
            if cui in sapbert_ranks and cui in tfidf_ranks:
                rrf_score *= (1.0 + self.config.ensemble_final_k)  # Using final_k as diversity bonus placeholder

            rrf_scores[cui] = rrf_score

        # Sort by RRF score
        sorted_cuis = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Build candidate list
        ensemble_candidates = []
        for cui, rrf_score in sorted_cuis:
            # Get name from either method
            if cui in sapbert_ranks:
                name = next(c.name for c in sapbert_candidates if c.cui == cui)
            else:
                name = next(c.name for c in tfidf_candidates if c.cui == cui)

            ensemble_candidates.append(Candidate(
                cui=cui,
                name=name,
                score=rrf_score,
                method='ensemble'
            ))

        return ensemble_candidates

    def _load_sapbert(self):
        """Load SapBERT model and precomputed embeddings"""
        logger.info("Loading SapBERT model and embeddings...")

        # Load model
        device = torch.device(self.config.sapbert_device if torch.cuda.is_available() else 'cpu')

        # Log device info
        if device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"✓ Batch size: {self.config.sapbert_batch_size}")
        else:
            logger.warning(f"⚠️  Using CPU (no GPU available)")
            logger.warning(f"⚠️  Encoding will be VERY slow (~4 hours for 8M names)")

        self.sapbert_tokenizer = AutoTokenizer.from_pretrained(self.config.sapbert_model)
        self.sapbert_model = AutoModel.from_pretrained(self.config.sapbert_model).to(device)
        self.sapbert_model.eval()

        # Load or compute embeddings
        if self.sapbert_cache.exists() and not self.config.force_recompute:
            logger.info(f"Loading precomputed SapBERT embeddings from {self.sapbert_cache}")
            with open(self.sapbert_cache, 'rb') as f:
                cached = pickle.load(f)
                self.sapbert_embeddings = cached['embeddings']
                self.umls_names = cached['names']
                self.name_to_cui = cached['name_to_cui']
        else:
            logger.info("Computing SapBERT embeddings for all UMLS concepts...")
            self._precompute_sapbert_embeddings()

    def _precompute_sapbert_embeddings(self):
        """Precompute SapBERT embeddings for all UMLS names"""

        # Get all UMLS names
        self.umls_names = self.umls_loader.get_all_names()
        self.name_to_cui = {}
        for name in self.umls_names:
            cuis = self.umls_loader.lookup_by_name(name)
            if cuis:
                self.name_to_cui[name] = cuis[0]  # Take first CUI

        # Encode in batches
        logger.info(f"Encoding {len(self.umls_names)} UMLS names with SapBERT...")
        logger.info(f"Estimated time: ~30-60 min with GPU, ~4 hours with CPU")
        logger.info(f"This is ONE-TIME only. Subsequent runs will use cache (~1 min).")

        self.sapbert_embeddings = self._encode_sapbert(self.umls_names)

        # Save to cache
        logger.info(f"Saving SapBERT embeddings to {self.sapbert_cache}")
        with open(self.sapbert_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.sapbert_embeddings,
                'names': self.umls_names,
                'name_to_cui': self.name_to_cui
            }, f)

    def _encode_sapbert_with_checkpointing(
        self,
        texts: List[str],
        start_idx: int = 0,
        existing_embeddings: List = None,
        checkpoint_path: Path = None,
        checkpoint_every: int = 1000
    ) -> np.ndarray:
        """Encode texts with checkpointing to prevent data loss on crashes"""
        device = self.sapbert_model.device
        batch_size = self.config.sapbert_batch_size

        # Log encoding start with GPU info
        logger.info(f"🚀 Starting encoding on {device.type.upper()}")
        if device.type == 'cuda':
            logger.info(f"   GPU Memory before: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

        all_embeddings = existing_embeddings if existing_embeddings else []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_batch = start_idx // batch_size

        # Adjust texts if resuming
        if start_idx > 0:
            texts = texts[start_idx:]

        for i in tqdm(range(0, len(texts), batch_size),
                     desc=f"🔥 Encoding with SapBERT on {device.type.upper()}",
                     total=total_batches - start_batch,
                     unit="batch"):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = self.sapbert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            # Encode
            with torch.no_grad():
                outputs = self.sapbert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())

            # Checkpoint every N batches
            current_batch = start_batch + (i // batch_size)
            if checkpoint_path and current_batch > 0 and current_batch % checkpoint_every == 0:
                try:
                    # Save checkpoint
                    checkpoint_embeddings = np.vstack(all_embeddings)
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump({
                            'embeddings': all_embeddings,
                            'last_index': start_idx + i + batch_size,
                            'last_batch': current_batch
                        }, f)
                    logger.info(f"   💾 Checkpoint saved at batch {current_batch}/{total_batches}")

                    # Free GPU memory
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                        logger.info(f"   📊 GPU Memory: {gpu_mem:.2f}GB")

                except Exception as e:
                    logger.warning(f"   Failed to save checkpoint: {e}")

            # Log GPU memory every 1000 batches
            elif device.type == 'cuda' and current_batch % 1000 == 0 and current_batch > 0:
                gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"   📊 Batch {current_batch}/{total_batches} - GPU Memory: {gpu_mem:.2f}GB")

        # Log completion with GPU stats
        if device.type == 'cuda':
            logger.info(f"✅ Encoding complete!")
            logger.info(f"   GPU Memory after: {torch.cuda.memory_allocated(0)/1e9:.2f}GB")
            logger.info(f"   Peak GPU Memory: {torch.cuda.max_memory_allocated(0)/1e9:.2f}GB")

        return np.vstack(all_embeddings)

    def _load_tfidf(self):
        """Load TF-IDF vectorizer and precomputed matrix"""
        logger.info("Loading TF-IDF vectorizer and matrix...")

        # Load or build TF-IDF
        if self.tfidf_cache.exists() and not self.config.force_recompute:
            logger.info(f"Loading precomputed TF-IDF from {self.tfidf_cache}")
            with open(self.tfidf_cache, 'rb') as f:
                cached = pickle.load(f)
                self.tfidf_vectorizer = cached['vectorizer']
                self.tfidf_matrix = cached['matrix']
                self.umls_names = cached['names']
                self.name_to_cui = cached['name_to_cui']
        else:
            logger.info("Building TF-IDF index for all UMLS concepts...")
            self._precompute_tfidf()

    def _precompute_tfidf(self):
        """Precompute TF-IDF matrix for all UMLS names"""

        # Get all UMLS names (reuse if already loaded)
        if self.umls_names is None:
            self.umls_names = self.umls_loader.get_all_names()
            self.name_to_cui = {}
            for name in self.umls_names:
                cuis = self.umls_loader.lookup_by_name(name)
                if cuis:
                    self.name_to_cui[name] = cuis[0]

        # Build TF-IDF vectorizer
        logger.info(f"Building TF-IDF index for {len(self.umls_names)} UMLS names...")
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.config.tfidf_ngram_range,
            min_df=2,
            lowercase=True
        )

        # Fit and transform
        texts_lower = [name.lower() for name in self.umls_names]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_lower)

        # Save to cache
        logger.info(f"Saving TF-IDF index to {self.tfidf_cache}")
        with open(self.tfidf_cache, 'wb') as f:
            pickle.dump({
                'vectorizer': self.tfidf_vectorizer,
                'matrix': self.tfidf_matrix,
                'names': self.umls_names,
                'name_to_cui': self.name_to_cui
            }, f)
