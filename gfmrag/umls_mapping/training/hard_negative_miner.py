"""
Hard Negative Mining for Cross-Encoder Training

Generates hard negatives for contrastive learning using FAISS similarity search.

Strategy:
1. Semantic negatives: High similarity to correct CUI, but wrong CUI
2. Type negatives: Different semantic type from correct CUI
3. Random negatives: Random CUIs for training stability

Reference: Xiong et al. (2020) "Approximate Nearest Neighbor Negative
Contrastive Learning for Dense Text Retrieval"
"""

import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Mines hard negatives for UMLS entity linking training.

    Uses FAISS index to find confusingly similar (but incorrect) CUIs
    for each positive training example.
    """

    def __init__(
        self,
        umls_loader,  # UMLSLoader instance
        faiss_index_path: str = "tmp/umls_faiss_index",
        similarity_threshold: float = 0.85,
        top_k_candidates: int = 20,
        num_semantic_negatives: int = 5,
        num_type_negatives: int = 2,
        num_random_negatives: int = 2,
        cache_path: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Initialize hard negative miner.

        Args:
            umls_loader: UMLSLoader instance with concepts and FAISS index
            faiss_index_path: Path to FAISS index directory
            similarity_threshold: Minimum similarity for semantic hard negatives
            top_k_candidates: Number of top candidates to retrieve from FAISS
            num_semantic_negatives: Number of semantic hard negatives per positive
            num_type_negatives: Number of type negatives per positive
            num_random_negatives: Number of random negatives per positive
            cache_path: Path to cache mined negatives (speeds up training)
            random_state: Random seed for reproducibility
        """
        self.umls_loader = umls_loader
        self.faiss_index_path = Path(faiss_index_path)
        self.similarity_threshold = similarity_threshold
        self.top_k_candidates = top_k_candidates
        self.num_semantic_negatives = num_semantic_negatives
        self.num_type_negatives = num_type_negatives
        self.num_random_negatives = num_random_negatives
        self.cache_path = Path(cache_path) if cache_path else None
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)

        # Load FAISS index
        self.faiss_index = None
        self.cui_list = None  # Ordered list of CUIs matching FAISS index
        self._load_faiss_index()

        # Group CUIs by semantic type for type negative sampling
        self.type_to_cuis = defaultdict(list)
        self._group_cuis_by_type()

        # All CUIs for random sampling
        self.all_cuis = list(umls_loader.concepts.keys())

        # Cache for mined negatives
        self.negative_cache: Dict[str, Dict] = {}
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_faiss_index(self):
        """Load FAISS index and CUI list."""
        index_file = self.faiss_index_path / "umls.index"
        cui_list_file = self.faiss_index_path / "cui_list.txt"

        if not index_file.exists():
            logger.error(f"FAISS index not found: {index_file}")
            logger.error("Please run Stage 3 preprocessing to build FAISS index first")
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        logger.info(f"Loading FAISS index from {index_file}")
        self.faiss_index = faiss.read_index(str(index_file))

        # Load CUI list (maps FAISS index positions to CUIs)
        if cui_list_file.exists():
            with open(cui_list_file) as f:
                self.cui_list = [line.strip() for line in f]
        else:
            logger.warning("CUI list not found, will use index positions")
            self.cui_list = None

        logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")

    def _group_cuis_by_type(self):
        """Group CUIs by semantic type for type negative sampling."""
        logger.info("Grouping CUIs by semantic type")

        for cui, concept in self.umls_loader.concepts.items():
            if not concept.semantic_types:
                continue

            # Use first semantic type (most concepts have only one)
            primary_type = concept.semantic_types[0]
            self.type_to_cuis[primary_type].append(cui)

        logger.info(
            f"Grouped {len(self.umls_loader.concepts)} CUIs into "
            f"{len(self.type_to_cuis)} semantic types"
        )

    def _load_cache(self):
        """Load cached hard negatives."""
        logger.info(f"Loading cached hard negatives from {self.cache_path}")

        with open(self.cache_path, "rb") as f:
            self.negative_cache = pickle.load(f)

        logger.info(f"Loaded cache for {len(self.negative_cache)} CUIs")

    def save_cache(self):
        """Save mined hard negatives to cache."""
        if not self.cache_path:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving hard negative cache to {self.cache_path}")

        with open(self.cache_path, "wb") as f:
            pickle.dump(self.negative_cache, f)

        logger.info(f"Saved cache for {len(self.negative_cache)} CUIs")

    def mine_semantic_hard_negatives(
        self,
        cui: str,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Mine semantic hard negatives: high similarity, wrong CUI.

        Uses FAISS to find top-K similar CUIs, then filters by:
        - Similarity >= threshold (e.g., 0.85)
        - CUI != correct CUI
        - CUI != deprecated CUI

        Args:
            cui: Correct CUI
            k: Number of negatives to return (default: self.num_semantic_negatives)

        Returns:
            List of (negative_cui, similarity_score) tuples
        """
        if k is None:
            k = self.num_semantic_negatives

        # Get CUI embedding
        concept = self.umls_loader.concepts.get(cui)
        if not concept or not hasattr(concept, 'embedding'):
            logger.warning(f"CUI {cui} not found or has no embedding")
            return []

        embedding = concept.embedding.reshape(1, -1)

        # Search FAISS index
        similarities, indices = self.faiss_index.search(
            embedding, self.top_k_candidates
        )

        similarities = similarities[0]  # Shape: (top_k,)
        indices = indices[0]  # Shape: (top_k,)

        # Filter candidates
        hard_negatives = []
        for idx, sim in zip(indices, similarities):
            if len(hard_negatives) >= k:
                break

            # Map index to CUI
            if self.cui_list:
                candidate_cui = self.cui_list[idx]
            else:
                candidate_cui = f"C{idx:07d}"  # Fallback

            # Skip if same as correct CUI
            if candidate_cui == cui:
                continue

            # Check similarity threshold
            if sim < self.similarity_threshold:
                continue

            hard_negatives.append((candidate_cui, float(sim)))

        return hard_negatives

    def mine_type_negatives(
        self,
        cui: str,
        k: Optional[int] = None,
    ) -> List[str]:
        """
        Mine type negatives: different semantic type from correct CUI.

        Randomly samples CUIs from semantic types different from the
        correct CUI's type.

        Args:
            cui: Correct CUI
            k: Number of negatives to return (default: self.num_type_negatives)

        Returns:
            List of negative CUIs
        """
        if k is None:
            k = self.num_type_negatives

        # Get correct CUI's semantic type
        concept = self.umls_loader.concepts.get(cui)
        if not concept or not concept.semantic_types:
            logger.warning(f"CUI {cui} has no semantic type")
            return []

        correct_type = concept.semantic_types[0]

        # Collect all CUIs with different types
        different_type_cuis = []
        for sem_type, cuis in self.type_to_cuis.items():
            if sem_type != correct_type:
                different_type_cuis.extend(cuis)

        if not different_type_cuis:
            return []

        # Random sample
        k = min(k, len(different_type_cuis))
        return random.sample(different_type_cuis, k)

    def mine_random_negatives(self, cui: str, k: Optional[int] = None) -> List[str]:
        """
        Mine random negatives: random CUIs (easy negatives for stability).

        Args:
            cui: Correct CUI (to exclude)
            k: Number of negatives to return (default: self.num_random_negatives)

        Returns:
            List of random negative CUIs
        """
        if k is None:
            k = self.num_random_negatives

        # Sample from all CUIs, excluding correct CUI
        available_cuis = [c for c in self.all_cuis if c != cui]
        k = min(k, len(available_cuis))

        return random.sample(available_cuis, k)

    def mine_negatives_for_cui(
        self,
        cui: str,
        use_cache: bool = True,
    ) -> Dict[str, List]:
        """
        Mine all types of negatives for a given CUI.

        Args:
            cui: Correct CUI
            use_cache: Whether to use cached negatives if available

        Returns:
            Dictionary with keys:
            - semantic_negatives: List of (cui, similarity) tuples
            - type_negatives: List of CUIs
            - random_negatives: List of CUIs
        """
        # Check cache
        if use_cache and cui in self.negative_cache:
            return self.negative_cache[cui]

        # Mine fresh negatives
        semantic_negatives = self.mine_semantic_hard_negatives(cui)
        type_negatives = self.mine_type_negatives(cui)
        random_negatives = self.mine_random_negatives(cui)

        negatives = {
            "semantic_negatives": semantic_negatives,
            "type_negatives": type_negatives,
            "random_negatives": random_negatives,
        }

        # Cache for future use
        self.negative_cache[cui] = negatives

        return negatives

    def mine_negatives_batch(
        self,
        mentions: List[Dict],
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        """
        Mine negatives for a batch of mentions.

        Args:
            mentions: List of mention dictionaries (must have 'cui' key)
            show_progress: Show progress bar

        Returns:
            Dictionary mapping CUI -> negatives dict
        """
        # Get unique CUIs
        unique_cuis = list(set(m["cui"] for m in mentions))

        logger.info(f"Mining hard negatives for {len(unique_cuis)} unique CUIs")

        # Mine negatives
        cui_to_negatives = {}

        iterator = tqdm(unique_cuis, desc="Mining negatives") if show_progress else unique_cuis

        for cui in iterator:
            cui_to_negatives[cui] = self.mine_negatives_for_cui(cui)

        logger.info(f"Mined negatives for {len(cui_to_negatives)} CUIs")

        # Save cache
        if self.cache_path:
            self.save_cache()

        return cui_to_negatives

    def get_negative_statistics(
        self,
        cui_to_negatives: Optional[Dict[str, Dict]] = None
    ) -> Dict:
        """
        Compute statistics about mined negatives.

        Args:
            cui_to_negatives: Optional precomputed negatives dict
                            (uses cache if not provided)

        Returns:
            Dictionary with statistics
        """
        if cui_to_negatives is None:
            cui_to_negatives = self.negative_cache

        if not cui_to_negatives:
            return {}

        # Count negatives by type
        total_semantic = 0
        total_type = 0
        total_random = 0
        semantic_similarities = []

        for negatives in cui_to_negatives.values():
            semantic_negs = negatives.get("semantic_negatives", [])
            type_negs = negatives.get("type_negatives", [])
            random_negs = negatives.get("random_negatives", [])

            total_semantic += len(semantic_negs)
            total_type += len(type_negs)
            total_random += len(random_negs)

            # Collect similarity scores
            for _, sim in semantic_negs:
                semantic_similarities.append(sim)

        num_cuis = len(cui_to_negatives)

        return {
            "num_cuis": num_cuis,
            "total_negatives": total_semantic + total_type + total_random,
            "semantic_negatives": {
                "count": total_semantic,
                "avg_per_cui": total_semantic / num_cuis if num_cuis > 0 else 0,
                "avg_similarity": np.mean(semantic_similarities) if semantic_similarities else 0,
                "min_similarity": np.min(semantic_similarities) if semantic_similarities else 0,
                "max_similarity": np.max(semantic_similarities) if semantic_similarities else 0,
            },
            "type_negatives": {
                "count": total_type,
                "avg_per_cui": total_type / num_cuis if num_cuis > 0 else 0,
            },
            "random_negatives": {
                "count": total_random,
                "avg_per_cui": total_random / num_cuis if num_cuis > 0 else 0,
            },
        }

    def print_statistics(self):
        """Print statistics about mined negatives."""
        stats = self.get_negative_statistics()

        if not stats:
            print("No negatives mined yet")
            return

        print("=" * 80)
        print("HARD NEGATIVE MINING STATISTICS")
        print("=" * 80)
        print(f"CUIs with negatives:     {stats['num_cuis']:,}")
        print(f"Total negatives:         {stats['total_negatives']:,}")
        print()
        print("Semantic Hard Negatives (high similarity, wrong CUI):")
        print("-" * 80)
        print(f"  Count:                 {stats['semantic_negatives']['count']:,}")
        print(f"  Avg per CUI:           {stats['semantic_negatives']['avg_per_cui']:.2f}")
        print(f"  Avg similarity:        {stats['semantic_negatives']['avg_similarity']:.4f}")
        print(f"  Similarity range:      [{stats['semantic_negatives']['min_similarity']:.4f}, "
              f"{stats['semantic_negatives']['max_similarity']:.4f}]")
        print()
        print("Type Negatives (wrong semantic type):")
        print("-" * 80)
        print(f"  Count:                 {stats['type_negatives']['count']:,}")
        print(f"  Avg per CUI:           {stats['type_negatives']['avg_per_cui']:.2f}")
        print()
        print("Random Negatives (easy negatives for stability):")
        print("-" * 80)
        print(f"  Count:                 {stats['random_negatives']['count']:,}")
        print(f"  Avg per CUI:           {stats['random_negatives']['avg_per_cui']:.2f}")
        print("=" * 80)
