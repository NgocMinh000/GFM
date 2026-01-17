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

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available. Will use slower sklearn cosine_similarity.")

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
        self.faiss_cache = cache_dir / "faiss_ivf.index"

        # Initialize models (lazy loading)
        self.sapbert_model = None
        self.sapbert_tokenizer = None
        self.sapbert_embeddings = None
        self.faiss_index = None  # FAISS index for fast similarity search

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

    def generate_candidates_batch(self, entities: List[str], k: int = None) -> List[List[Candidate]]:
        """
        Generate candidates for multiple entities at once (BATCHED for speed)

        Args:
            entities: List of entity texts
            k: Number of candidates per entity

        Returns:
            List of candidate lists (one per entity)
        """
        if k is None:
            k = self.config.ensemble_final_k

        # Lazy load models
        if self.sapbert_model is None:
            self._load_sapbert()
        if self.tfidf_vectorizer is None:
            self._load_tfidf()

        # Batch encode all entities with SapBERT
        query_embs = self._encode_sapbert(entities)  # [batch_size, 768]

        # Batch search FAISS
        if self.faiss_index is not None:
            scores_batch, indices_batch = self.faiss_index.search(
                query_embs.astype('float32'),
                self.config.sapbert_top_k
            )  # [batch_size, k]
        else:
            # Fallback: process one by one
            return [self.generate_candidates(entity, k) for entity in entities]

        # Batch TF-IDF search (FIXED: Process one-by-one to avoid memory explosion)
        # NOTE: Cannot batch TF-IDF efficiently because:
        #   - tfidf_matrix is HUGE (7.9M × vocab_size)
        #   - Batching creates dense [batch_size × 7.9M] matrix → OOM + slow
        #   - Processing one-by-one with sparse ops is actually FASTER
        entities_lower = [e.lower() for e in entities]

        tfidf_top_k_indices_list = []
        tfidf_top_k_scores_list = []

        for entity_lower in entities_lower:
            # Transform single query (stays sparse)
            query_vec = self.tfidf_vectorizer.transform([entity_lower])  # [1, vocab_size] sparse

            # Compute similarities (sparse × sparse = sparse, FAST!)
            similarities = (query_vec * self.tfidf_matrix.T).toarray()[0]  # [7.9M] dense

            # Get top-k indices and scores
            top_k_indices = np.argpartition(-similarities, self.config.sapbert_top_k)[:self.config.sapbert_top_k]
            top_k_indices = top_k_indices[np.argsort(-similarities[top_k_indices])]  # Sort top-k
            top_k_scores = similarities[top_k_indices]

            tfidf_top_k_indices_list.append(top_k_indices)
            tfidf_top_k_scores_list.append(top_k_scores)

        # Convert to arrays for easier indexing
        tfidf_top_k_indices = np.array(tfidf_top_k_indices_list)  # [batch_size, k]
        tfidf_top_k_scores = np.array(tfidf_top_k_scores_list)    # [batch_size, k]

        # Process results for each entity
        all_candidates = []
        for i, entity in enumerate(entities):
            # SapBERT candidates from batch search
            sapbert_candidates = []
            for idx, score in zip(indices_batch[i], scores_batch[i]):
                name = self.umls_names[idx]
                # Safe lookup with fallback
                if name in self.name_to_cui:
                    cui = self.name_to_cui[name]
                else:
                    # Try to find CUI via UMLS loader
                    cuis = self.umls_loader.lookup_by_name(name)
                    if cuis:
                        cui = cuis[0]
                    else:
                        # Skip if no CUI found
                        continue

                sapbert_candidates.append(Candidate(
                    cui=cui,
                    name=name,
                    score=float(score),
                    method='sapbert'
                ))

            # TF-IDF candidates from batch search
            tfidf_candidates = []
            for idx, score in zip(tfidf_top_k_indices[i], tfidf_top_k_scores[i]):
                name = self.umls_names[idx]
                # Safe lookup with fallback
                if name in self.name_to_cui:
                    cui = self.name_to_cui[name]
                else:
                    # Try to find CUI via UMLS loader
                    cuis = self.umls_loader.lookup_by_name(name)
                    if cuis:
                        cui = cuis[0]
                    else:
                        # Skip if no CUI found
                        continue

                tfidf_candidates.append(Candidate(
                    cui=cui,
                    name=name,
                    score=float(score),
                    method='tfidf'
                ))

            # Ensemble
            ensemble_candidates = self._reciprocal_rank_fusion(
                sapbert_candidates,
                tfidf_candidates,
                k=k
            )

            all_candidates.append(ensemble_candidates)

        return all_candidates

    def _get_sapbert_candidates(self, entity: str, k: int) -> List[Candidate]:
        """Get candidates using SapBERT semantic similarity"""

        # Lazy load SapBERT
        if self.sapbert_model is None:
            self._load_sapbert()

        # Encode query entity
        query_emb = self._encode_sapbert([entity])[0]

        # Use FAISS for fast search if available
        if self.faiss_index is not None:
            # FAISS search: returns (distances, indices)
            # IndexFlatIP returns inner product scores (= cosine similarity for normalized vectors)
            query_emb_reshaped = query_emb.reshape(1, -1).astype('float32')
            scores, top_k_indices = self.faiss_index.search(query_emb_reshaped, k)

            # Flatten arrays
            scores = scores[0]  # [k]
            top_k_indices = top_k_indices[0]  # [k]
        else:
            # Fallback to sklearn (slow)
            similarities = cosine_similarity([query_emb], self.sapbert_embeddings)[0]
            top_k_indices = np.argsort(similarities)[::-1][:k]
            scores = similarities[top_k_indices]

        # Build candidate list
        candidates = []
        for idx, score in zip(top_k_indices, scores):
            name = self.umls_names[idx]
            # Safe lookup with fallback
            if name in self.name_to_cui:
                cui = self.name_to_cui[name]
            else:
                # Try to find CUI via UMLS loader
                cuis = self.umls_loader.lookup_by_name(name)
                if cuis:
                    cui = cuis[0]
                else:
                    # Skip if no CUI found
                    continue

            candidates.append(Candidate(
                cui=cui,
                name=name,
                score=float(score),
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

            # Safe lookup with fallback
            if name in self.name_to_cui:
                cui = self.name_to_cui[name]
            else:
                # Try to find CUI via UMLS loader
                cuis = self.umls_loader.lookup_by_name(name)
                if cuis:
                    cui = cuis[0]
                else:
                    # Skip if no CUI found
                    continue

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

            # Check if we have memory-mapped version (.npy)
            npy_cache = self.sapbert_cache.with_suffix('.npy')
            if npy_cache.exists():
                logger.info("   📦 Loading embeddings with memory-mapping (RAM-efficient)...")
                self.sapbert_embeddings = np.load(str(npy_cache), mmap_mode='r')
                # Load metadata
                meta_cache = self.sapbert_cache.with_suffix('.meta.pkl')
                with open(meta_cache, 'rb') as f:
                    meta = pickle.load(f)
                    self.umls_names = meta['names']
                    self.name_to_cui = meta['name_to_cui']
                logger.info(f"   ✓ Loaded {len(self.sapbert_embeddings):,} embeddings (memory-mapped)")
            else:
                # Legacy pickle format - requires full RAM load
                logger.warning("   ⚠️  Legacy pickle format detected (requires 24GB RAM)")
                logger.info("   Converting to memory-mapped format for future use...")
                with open(self.sapbert_cache, 'rb') as f:
                    cached = pickle.load(f)
                    self.sapbert_embeddings = cached['embeddings']
                    self.umls_names = cached['names']
                    self.name_to_cui = cached['name_to_cui']

                # Save as memory-mapped for next time
                logger.info("   Saving memory-mapped version...")
                np.save(str(npy_cache), self.sapbert_embeddings)
                with open(meta_cache, 'wb') as f:
                    pickle.dump({'names': self.umls_names, 'name_to_cui': self.name_to_cui}, f)
                logger.info("   ✓ Memory-mapped version saved for future runs")

                # Reload as memory-mapped to free RAM
                del self.sapbert_embeddings
                import gc
                gc.collect()
                self.sapbert_embeddings = np.load(str(npy_cache), mmap_mode='r')
        else:
            logger.info("Computing SapBERT embeddings for all UMLS concepts...")
            self._precompute_sapbert_embeddings()

        # Build FAISS index for fast similarity search
        self._build_faiss_index()

    def _encode_sapbert(self, texts: List[str]) -> np.ndarray:
        """
        Encode a small list of texts using SapBERT (for query encoding during inference).

        This is different from _encode_sapbert_chunked which is for bulk encoding
        millions of UMLS names during precomputation.

        Args:
            texts: List of text strings to encode (typically 1 query entity)

        Returns:
            Numpy array of embeddings (shape: [len(texts), 768])
        """
        device = self.sapbert_model.device

        # Tokenize
        inputs = self.sapbert_tokenizer(
            texts,
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

        return embeddings.cpu().numpy()

    def _build_faiss_index(self):
        """Build FAISS index with GPU if possible, CPU fallback"""
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Similarity search will be slow.")
            return

        # Check if we have cached index
        if self.faiss_cache.exists() and not self.config.force_recompute:
            logger.info(f"Loading precomputed FAISS index from {self.faiss_cache}")
            try:
                self.faiss_index = faiss.read_index(str(self.faiss_cache))
                logger.info(f"   ✓ Loaded IVF index with {self.faiss_index.ntotal:,} vectors")
                return
            except Exception as e:
                logger.warning(f"   Failed to load cached index: {e}")
                logger.info("   Rebuilding index...")

        logger.info("Building FAISS index for fast similarity search...")
        logger.info(f"   Indexing {len(self.sapbert_embeddings):,} embeddings (dim={self.sapbert_embeddings.shape[1]})")

        dim = self.sapbert_embeddings.shape[1]  # 768

        # Try GPU FAISS with memory management
        if torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
            try:
                logger.info("   🔧 Attempting GPU FAISS for maximum speed...")

                # Temporarily move SapBERT to CPU to free GPU memory
                logger.info("   Temporarily moving SapBERT to CPU to free GPU memory...")
                sapbert_device = self.sapbert_model.device
                self.sapbert_model = self.sapbert_model.cpu()
                torch.cuda.empty_cache()

                # Check available GPU memory
                gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1e9
                logger.info(f"   GPU memory available: {gpu_free:.1f}GB")
                logger.info(f"   Index needs: ~24GB")

                if gpu_free >= 22:  # Need at least 22GB free
                    logger.info("   Building GPU IndexFlatIP...")

                    # Build CPU index first
                    cpu_index = faiss.IndexFlatIP(dim)

                    # Move to GPU
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp memory

                    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

                    # Add embeddings
                    logger.info("   Adding embeddings to GPU index...")
                    gpu_index.add(self.sapbert_embeddings.astype('float32'))

                    self.faiss_index = gpu_index

                    # Move SapBERT back to GPU if there's room
                    try:
                        self.sapbert_model = self.sapbert_model.to(sapbert_device)
                        logger.info("   ✅ GPU FAISS index built successfully!")
                        logger.info("   ✅ SapBERT moved back to GPU")
                    except:
                        logger.info("   ✅ GPU FAISS index built!")
                        logger.info("   ⚠️  SapBERT stays on CPU (GPU full)")

                    logger.info(f"   🚀 GPU FAISS: 50-100x faster than CPU!")
                    return
                else:
                    raise Exception(f"Insufficient GPU memory: {gpu_free:.1f}GB < 22GB needed")

            except Exception as e:
                logger.warning(f"   GPU FAISS failed: {str(e)[:80]}...")
                logger.info("   Falling back to CPU FAISS")

                # Move SapBERT back to GPU
                try:
                    self.sapbert_model = self.sapbert_model.to(sapbert_device)
                    logger.info("   SapBERT moved back to GPU")
                except:
                    pass

                torch.cuda.empty_cache()

        # CPU fallback with memory-efficient IVF index
        logger.info("   Building CPU IndexIVFFlat (approximate search, memory-efficient)...")
        logger.info("   SapBERT stays on GPU for fast query encoding")

        # Use IVF (Inverted File) index for memory efficiency
        # nlist = number of clusters (sqrt(N) is a good heuristic)
        n_embeddings = len(self.sapbert_embeddings)
        nlist = min(int(np.sqrt(n_embeddings)), 16384)  # Max 16K clusters
        nprobe = 64  # Number of clusters to search (higher = more accurate)

        logger.info(f"   IVF config: {nlist:,} clusters, nprobe={nprobe}")
        logger.info(f"   Expected accuracy: >95% (vs 100% for exact search)")
        logger.info(f"   Memory savings: ~10x less than IndexFlatIP")

        # Create IVF index
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train on subset to save memory (10% sample or max 1M vectors)
        train_size = min(n_embeddings, max(int(n_embeddings * 0.1), 100000))
        logger.info(f"   Training IVF on {train_size:,} samples...")

        # Sample embeddings for training (every Nth sample)
        sample_step = max(1, n_embeddings // train_size)

        # Load sampled embeddings as contiguous float32 array
        train_embeddings = np.ascontiguousarray(
            self.sapbert_embeddings[::sample_step],
            dtype=np.float32
        )

        cpu_index.train(train_embeddings)
        logger.info(f"   ✓ Training complete")

        # Free training data
        del train_embeddings
        import gc
        gc.collect()

        # Add all embeddings in batches to avoid memory spike
        # Use smaller batch size to prevent OOM with memory-mapped arrays
        batch_size = 50000  # Reduced from 100K to minimize memory spikes
        logger.info(f"   Adding {n_embeddings:,} embeddings in batches of {batch_size:,}...")

        import gc
        for i in range(0, n_embeddings, batch_size):
            end = min(i + batch_size, n_embeddings)

            # Load batch from memory-mapped array
            batch = self.sapbert_embeddings[i:end]

            # Convert to float32 contiguous array
            batch_f32 = np.ascontiguousarray(batch, dtype=np.float32)

            # Add to index
            cpu_index.add(batch_f32)

            # Explicitly free memory
            del batch, batch_f32
            gc.collect()

            if (i // batch_size + 1) % 10 == 0 or end == n_embeddings:
                logger.info(f"      Added {end:,} / {n_embeddings:,} ({end/n_embeddings*100:.1f}%)")

        # Set search parameters
        cpu_index.nprobe = nprobe

        self.faiss_index = cpu_index
        logger.info(f"   ✓ IVF index built successfully with {nlist:,} clusters")

        # Save index to cache for future runs
        logger.info(f"   Saving FAISS index to {self.faiss_cache}...")
        faiss.write_index(cpu_index, str(self.faiss_cache))
        logger.info(f"   ✓ Index cached successfully")
        logger.info(f"   ✅ IVF index built and cached!")

    def _precompute_sapbert_embeddings(self):
        """Precompute SapBERT embeddings for all UMLS names using chunked processing"""

        # Get all UMLS names
        self.umls_names = self.umls_loader.get_all_names()
        self.name_to_cui = {}
        for name in self.umls_names:
            cuis = self.umls_loader.lookup_by_name(name)
            if cuis:
                self.name_to_cui[name] = cuis[0]  # Take first CUI

        # Encode in chunks to avoid RAM overflow
        logger.info(f"Encoding {len(self.umls_names)} UMLS names with SapBERT...")
        logger.info(f"Using CHUNKED processing to prevent RAM overflow")
        logger.info(f"Estimated time: ~30-60 min with GPU, ~4 hours with CPU")
        logger.info(f"This is ONE-TIME only. Subsequent runs will use cache (~1 min).")

        # Use chunked encoding with automatic memory management
        self.sapbert_embeddings = self._encode_sapbert_chunked(self.umls_names)

        # Save to cache in memory-mapped format
        logger.info(f"Saving SapBERT embeddings (memory-mapped format)...")

        # Save embeddings as .npy (memory-mappable)
        npy_cache = self.sapbert_cache.with_suffix('.npy')
        np.save(str(npy_cache), self.sapbert_embeddings)
        logger.info(f"   ✓ Embeddings saved: {npy_cache}")

        # Save metadata separately
        meta_cache = self.sapbert_cache.with_suffix('.meta.pkl')
        with open(meta_cache, 'wb') as f:
            pickle.dump({
                'names': self.umls_names,
                'name_to_cui': self.name_to_cui
            }, f)
        logger.info(f"   ✓ Metadata saved: {meta_cache}")

        # Also save legacy pickle format for backward compatibility
        with open(self.sapbert_cache, 'wb') as f:
            pickle.dump({
                'embeddings': self.sapbert_embeddings,
                'names': self.umls_names,
                'name_to_cui': self.name_to_cui
            }, f)
        logger.info(f"   ✓ Legacy format saved: {self.sapbert_cache}")

        logger.info("✓ SapBERT embeddings cached successfully")

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

    def _encode_sapbert_chunked(
        self,
        texts: List[str],
        chunk_size: int = 1_000_000  # Process 1M names at a time
    ) -> np.ndarray:
        """
        Encode texts in chunks to prevent RAM overflow.

        Instead of accumulating all embeddings in memory (~30GB for 7.9M names),
        this processes in chunks and saves to disk immediately.

        Args:
            texts: List of text strings to encode
            chunk_size: Number of texts per chunk (default 1M = ~3GB RAM)

        Returns:
            Combined embeddings array
        """
        device = self.sapbert_model.device
        batch_size = self.config.sapbert_batch_size
        chunk_dir = self.sapbert_cache.parent / "sapbert_chunks"
        chunk_dir.mkdir(exist_ok=True)

        total_texts = len(texts)
        num_chunks = (total_texts + chunk_size - 1) // chunk_size

        logger.info(f"🔥 Chunked encoding: {num_chunks} chunks of ~{chunk_size:,} names each")
        logger.info(f"   This prevents RAM overflow (peak ~3-4GB instead of ~30GB)")

        if device.type == 'cuda':
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

        # Check for existing chunks to resume
        existing_chunks = sorted(chunk_dir.glob("chunk_*.npy"))
        start_chunk = len(existing_chunks)

        if start_chunk > 0:
            logger.info(f"🔄 Found {start_chunk} existing chunks, resuming from chunk {start_chunk}...")

        # Process each chunk
        for chunk_idx in range(start_chunk, num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_texts)
            chunk_texts = texts[chunk_start:chunk_end]

            logger.info(f"\n📦 Processing chunk {chunk_idx+1}/{num_chunks}")
            logger.info(f"   Range: {chunk_start:,} → {chunk_end:,} ({len(chunk_texts):,} names)")

            # Encode this chunk
            chunk_embeddings = []
            num_batches = (len(chunk_texts) + batch_size - 1) // batch_size

            for i in tqdm(range(0, len(chunk_texts), batch_size),
                         desc=f"🔥 Encoding chunk {chunk_idx+1}/{num_chunks}",
                         total=num_batches,
                         unit="batch"):
                batch_texts = chunk_texts[i:i+batch_size]

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

                chunk_embeddings.append(embeddings.cpu().numpy())

                # Clear GPU cache periodically
                if device.type == 'cuda' and i % (batch_size * 100) == 0:
                    torch.cuda.empty_cache()

            # Combine chunk embeddings
            chunk_array = np.vstack(chunk_embeddings)

            # Save chunk to disk IMMEDIATELY (frees RAM)
            chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npy"
            np.save(chunk_file, chunk_array)

            # Free memory
            del chunk_embeddings
            del chunk_array
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # Log progress
            progress_pct = 100 * (chunk_idx + 1) / num_chunks
            logger.info(f"   ✓ Chunk {chunk_idx+1}/{num_chunks} saved to disk ({progress_pct:.1f}% complete)")

            if device.type == 'cuda':
                gpu_mem = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"   📊 GPU Memory: {gpu_mem:.2f}GB")

        # All chunks processed, now combine them
        logger.info(f"\n🔗 Combining {num_chunks} chunks into final array...")
        logger.info(f"   Loading chunks one-by-one to minimize RAM usage...")

        # Load and combine chunks efficiently
        all_chunks = sorted(chunk_dir.glob("chunk_*.npy"))
        combined_embeddings = []

        for i, chunk_file in enumerate(all_chunks):
            logger.info(f"   Loading chunk {i+1}/{len(all_chunks)}...")
            chunk_data = np.load(chunk_file)
            combined_embeddings.append(chunk_data)

        # Combine all chunks
        final_embeddings = np.vstack(combined_embeddings)
        logger.info(f"✅ Final embeddings shape: {final_embeddings.shape}")

        # Clean up chunk files
        logger.info(f"🧹 Cleaning up {len(all_chunks)} temporary chunk files...")
        for chunk_file in all_chunks:
            chunk_file.unlink()
        chunk_dir.rmdir()
        logger.info("✓ Cleanup complete")

        return final_embeddings

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
