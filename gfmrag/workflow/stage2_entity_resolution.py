"""
Stage 2: Entity Resolution Pipeline
====================================

Replaces ColBERT entity linking with sophisticated multi-stage pipeline.

INPUT: kg.txt (triples from Stage 1)
OUTPUT: kg_clean.txt (triples + SYNONYM_OF edges)

Pipeline Architecture:
----------------------
[STAGE 0] Type Inference       → Classify entity types (drug/disease/symptom/etc.)
[STAGE 1] SapBERT Embedding    → Convert names to 768-dim vectors
[STAGE 2] FAISS Blocking       → Find ~150 candidates per entity
[STAGE 3] Multi-Feature Scoring → Calculate similarity (5 features)
[STAGE 4] Adaptive Thresholding → Type-specific decision thresholds
[STAGE 5] Clustering & Canon    → Group synonyms + canonical names

Each stage has:
- Input/output validation
- Evaluation metrics
- Intermediate file saving
- Error handling
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from gfmrag.kg_construction.utils import KG_DELIMITER

logger = logging.getLogger(__name__)


@dataclass
class EntityResolutionConfig:
    """Configuration for entity resolution pipeline"""

    # Input/Output paths
    kg_input_path: str  # Path to kg.txt from Stage 1
    output_dir: str     # Directory for all outputs

    # Stage 0: Type Inference
    type_inference_method: str = "hybrid"  # pattern/relationship/hybrid

    # Stage 1: SapBERT Embedding
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    embedding_batch_size: int = 256
    embedding_device: str = "cuda"  # cuda/cpu

    # Stage 2: FAISS Blocking
    faiss_k_neighbors: int = 150  # Candidates per entity
    faiss_similarity_threshold: float = 0.60
    faiss_index_type: str = "IndexHNSWFlat"  # Fast approximate search

    # Stage 3: Multi-Feature Scoring
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "sapbert": 0.50,      # SapBERT similarity
        "lexical": 0.25,      # String similarity
        "type_consistency": 0.15,  # Same type check
        "graph": 0.10,        # Shared neighbors
        "umls": 0.0,          # UMLS alignment (disabled)
    })

    # Stage 4: Adaptive Thresholding
    type_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "drug": 0.86,         # Strict - dosage sensitivity
        "disease": 0.82,      # Medium
        "symptom": 0.77,      # Lenient - high variation
        "procedure": 0.80,
        "gene": 0.91,         # Very strict
        "anatomy": 0.82,
        "other": 0.80,        # Default threshold
    })

    # Stage 5: Clustering
    canonical_selection_method: str = "frequency"  # frequency/length/umls

    # General
    num_processes: int = 10
    force: bool = False  # Force recompute all stages
    save_intermediate: bool = True  # Save outputs of each stage


class EntityResolutionPipeline:
    """
    Multi-stage entity resolution pipeline for medical knowledge graphs.

    Workflow:
    1. Load kg.txt triples
    2. Extract unique entities
    3. Run 6-stage resolution pipeline
    4. Output clean KG with SYNONYM_OF edges
    """

    def __init__(self, config: EntityResolutionConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage output paths
        self.stage_paths = {
            "stage0_types": self.output_dir / "stage0_entity_types.json",
            "stage1_embeddings": self.output_dir / "stage1_embeddings.npy",
            "stage1_entity_ids": self.output_dir / "stage1_entity_ids.json",
            "stage2_candidates": self.output_dir / "stage2_candidate_pairs.jsonl",
            "stage3_scores": self.output_dir / "stage3_scored_pairs.jsonl",
            "stage4_equivalents": self.output_dir / "stage4_equivalent_pairs.jsonl",
            "stage5_clusters": self.output_dir / "stage5_clusters.json",
            "stage5_canonical": self.output_dir / "stage5_canonical_names.json",
            "kg_clean": self.output_dir / "kg_clean.txt",
        }

        # Data storage
        self.entities: List[str] = []
        self.entity_to_id: Dict[str, int] = {}
        self.id_to_entity: Dict[int, str] = {}
        self.triples: List[Tuple[str, str, str]] = []

    def load_kg(self, kg_path: str) -> None:
        """
        Load kg.txt and extract unique entities.

        Args:
            kg_path: Path to kg.txt from Stage 1

        Format: head,relation,tail (using KG_DELIMITER from utils)
        """
        logger.info(f"Loading KG from: {kg_path}")

        entity_set = set()
        self.triples = []

        with open(kg_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(KG_DELIMITER)
                if len(parts) != 3:
                    logger.warning(f"Line {line_num}: Invalid triple format (expected 3 parts, got {len(parts)})")
                    continue

                head, relation, tail = parts
                self.triples.append((head, relation, tail))
                entity_set.add(head)
                entity_set.add(tail)

        # Create entity mappings
        self.entities = sorted(entity_set)
        self.entity_to_id = {e: i for i, e in enumerate(self.entities)}
        self.id_to_entity = {i: e for i, e in enumerate(self.entities)}

        logger.info(f"✅ Loaded {len(self.triples)} triples")
        logger.info(f"✅ Extracted {len(self.entities)} unique entities")

    def save_kg_clean(self, synonym_edges: List[Tuple[str, str, str]]) -> None:
        """
        Save clean KG with SYNONYM_OF edges.

        Args:
            synonym_edges: List of (entity1, "SYNONYM_OF", entity2) tuples
        """
        output_path = self.stage_paths["kg_clean"]
        logger.info(f"Saving clean KG to: {output_path}")

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write original triples
            for head, relation, tail in self.triples:
                f.write(f"{head}{KG_DELIMITER}{relation}{KG_DELIMITER}{tail}\n")

            # Write synonym edges
            for entity1, relation, entity2 in synonym_edges:
                f.write(f"{entity1}{KG_DELIMITER}{relation}{KG_DELIMITER}{entity2}\n")

        logger.info(f"✅ Saved {len(self.triples)} original + {len(synonym_edges)} synonym edges")

    def run(self) -> None:
        """Execute full 6-stage pipeline"""
        logger.info("="*80)
        logger.info("STAGE 2: ENTITY RESOLUTION PIPELINE")
        logger.info("="*80)

        # Load KG
        self.load_kg(self.config.kg_input_path)

        # Stage 0: Type Inference
        entity_types = self.stage0_type_inference()

        # Stage 1: SapBERT Embedding
        embeddings = self.stage1_sapbert_embedding()

        # Stage 2: FAISS Blocking
        candidate_pairs = self.stage2_faiss_blocking(embeddings, entity_types)

        # Stage 3: Multi-Feature Scoring
        scored_pairs = self.stage3_multifeature_scoring(candidate_pairs, embeddings, entity_types)

        # Stage 4: Adaptive Thresholding
        equivalent_pairs = self.stage4_adaptive_thresholding(scored_pairs, entity_types)

        # Stage 5: Clustering & Canonicalization
        synonym_edges = self.stage5_clustering_canonicalization(equivalent_pairs)

        # Save clean KG
        self.save_kg_clean(synonym_edges)

        logger.info("="*80)
        logger.info("✅ ENTITY RESOLUTION PIPELINE COMPLETED")
        logger.info("="*80)

    # ========================================================================
    # STAGE 0: TYPE INFERENCE
    # ========================================================================

    def stage0_type_inference(self) -> Dict[str, Dict]:
        """
        Classify entity types (drug, disease, symptom, etc.)

        Returns:
            dict: {entity_name: {"type": str, "confidence": float}}
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 0: TYPE INFERENCE")
        logger.info("="*80)

        output_path = self.stage_paths["stage0_types"]

        # Check cache
        if not self.config.force and output_path.exists():
            logger.info(f"Loading cached types from: {output_path}")
            with open(output_path, 'r') as f:
                entity_types = json.load(f)
            logger.info(f"✅ Loaded types for {len(entity_types)} entities")
            return entity_types

        logger.info(f"Method: {self.config.type_inference_method}")
        logger.info(f"Processing {len(self.entities)} entities...")

        entity_types = {}
        method = self.config.type_inference_method

        for entity in tqdm(self.entities, desc="Type inference"):
            if method == "pattern":
                entity_types[entity] = self._infer_type_pattern(entity)
            elif method == "relationship":
                entity_types[entity] = self._infer_type_relationship(entity)
            else:  # hybrid
                entity_types[entity] = self._infer_type_hybrid(entity)

        # Save
        if self.config.save_intermediate:
            with open(output_path, 'w') as f:
                json.dump(entity_types, f, indent=2)
            logger.info(f"✅ Saved to: {output_path}")

        # Evaluation
        self.evaluate_stage0(entity_types)

        return entity_types

    def _infer_type_pattern(self, entity: str) -> Dict:
        """Infer entity type using pattern matching (regex)"""
        import re

        entity_lower = entity.lower()

        # Medical suffix patterns
        disease_patterns = [
            r'.*itis$',  # inflammation: arthritis, hepatitis
            r'.*osis$',  # condition: necrosis, psychosis
            r'.*oma$',   # tumor: carcinoma, melanoma
            r'.*pathy$', # disease: neuropathy, myopathy
            r'.*syndrome$',
            r'.*disease$',
            r'.*infection$',
            r'.*cancer$',
            r'.*tumor$',
        ]

        drug_patterns = [
            r'.*cin$',    # antibiotics: penicillin, streptomycin
            r'.*ril$',    # ACE inhibitors: lisinopril, enalapril
            r'.*olol$',   # beta blockers: propranolol, atenolol
            r'.*pam$',    # benzodiazepines: diazepam, lorazepam
            r'.*ide$',    # diuretics: furosemide, hydrochlorothiazide
            r'.*statin$', # statins: atorvastatin, simvastatin
            r'.*mab$',    # monoclonal antibodies: rituximab
        ]

        symptom_patterns = [
            r'.*pain$',
            r'.*ache$',
            r'.*fever$',
            r'.*cough$',
            r'.*nausea$',
            r'.*fatigue$',
        ]

        gene_patterns = [
            r'^[A-Z]{2,6}\d+$',  # Gene symbols: TP53, BRCA1
            r'.*gene$',
        ]

        procedure_patterns = [
            r'.*ectomy$',  # removal: appendectomy
            r'.*otomy$',   # cutting: laparotomy
            r'.*plasty$',  # surgical repair: angioplasty
            r'.*scopy$',   # examination: endoscopy
            r'.*therapy$',
            r'.*surgery$',
        ]

        # Check patterns
        for pattern in disease_patterns:
            if re.match(pattern, entity_lower):
                return {"type": "disease", "confidence": 0.85}

        for pattern in drug_patterns:
            if re.match(pattern, entity_lower):
                return {"type": "drug", "confidence": 0.85}

        for pattern in symptom_patterns:
            if re.match(pattern, entity_lower):
                return {"type": "symptom", "confidence": 0.85}

        for pattern in gene_patterns:
            if re.match(pattern, entity_lower):
                return {"type": "gene", "confidence": 0.85}

        for pattern in procedure_patterns:
            if re.match(pattern, entity_lower):
                return {"type": "procedure", "confidence": 0.85}

        # Anatomy keywords
        anatomy_keywords = ['nerve', 'artery', 'vein', 'muscle', 'bone', 'organ',
                           'tissue', 'cell', 'gland', 'membrane']
        for keyword in anatomy_keywords:
            if keyword in entity_lower:
                return {"type": "anatomy", "confidence": 0.75}

        return {"type": "other", "confidence": 0.5}

    def _infer_type_relationship(self, entity: str) -> Dict:
        """Infer entity type using graph relationships"""
        # Build relationship profile for this entity
        incoming_relations = []
        outgoing_relations = []

        for head, rel, tail in self.triples:
            if head == entity:
                outgoing_relations.append(rel.lower())
            if tail == entity:
                incoming_relations.append(rel.lower())

        # Relationship-based rules
        drug_relations = ['treats', 'cures', 'prevents', 'inhibits', 'blocks',
                         'prescribed for', 'used to treat']
        disease_relations = ['caused by', 'treated by', 'symptom of', 'diagnosed as']
        symptom_relations = ['symptom of', 'associated with', 'indicates']
        procedure_relations = ['performed on', 'used to treat', 'surgical']

        # Check outgoing relations (entity is subject)
        for rel in outgoing_relations:
            for drug_rel in drug_relations:
                if drug_rel in rel:
                    return {"type": "drug", "confidence": 0.80}

        # Check incoming relations (entity is object)
        for rel in incoming_relations:
            for disease_rel in disease_relations:
                if disease_rel in rel:
                    return {"type": "disease", "confidence": 0.80}
            for symptom_rel in symptom_relations:
                if symptom_rel in rel:
                    return {"type": "symptom", "confidence": 0.80}

        return {"type": "other", "confidence": 0.5}

    def _infer_type_hybrid(self, entity: str) -> Dict:
        """Combine pattern and relationship methods"""
        pattern_result = self._infer_type_pattern(entity)
        relationship_result = self._infer_type_relationship(entity)

        # If both agree, high confidence
        if pattern_result["type"] == relationship_result["type"] and pattern_result["type"] != "other":
            return {"type": pattern_result["type"], "confidence": 0.95}

        # If pattern is confident, use it
        if pattern_result["confidence"] >= 0.85:
            return pattern_result

        # If relationship is confident, use it
        if relationship_result["confidence"] >= 0.80:
            return relationship_result

        # Otherwise use pattern (more reliable than "other")
        if pattern_result["type"] != "other":
            return pattern_result

        if relationship_result["type"] != "other":
            return relationship_result

        return {"type": "other", "confidence": 0.5}

    def evaluate_stage0(self, entity_types: Dict) -> None:
        """Evaluate type inference quality"""
        from collections import Counter

        type_counts = Counter(e["type"] for e in entity_types.values())

        logger.info("\n📊 Stage 0 Evaluation:")
        logger.info(f"  Total entities: {len(entity_types)}")
        logger.info(f"  Type distribution:")
        for type_name, count in type_counts.most_common():
            percentage = 100 * count / len(entity_types)
            logger.info(f"    - {type_name}: {count} ({percentage:.1f}%)")

        avg_confidence = np.mean([e["confidence"] for e in entity_types.values()])
        logger.info(f"  Average confidence: {avg_confidence:.3f}")

    # ========================================================================
    # STAGE 1: SAPBERT EMBEDDING
    # ========================================================================

    def stage1_sapbert_embedding(self) -> np.ndarray:
        """
        Generate SapBERT embeddings for all entities.

        Returns:
            np.ndarray: (N, 768) embeddings matrix
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1: SAPBERT EMBEDDING")
        logger.info("="*80)

        embeddings_path = self.stage_paths["stage1_embeddings"]
        entity_ids_path = self.stage_paths["stage1_entity_ids"]

        # Check cache
        if not self.config.force and embeddings_path.exists():
            logger.info(f"Loading cached embeddings from: {embeddings_path}")
            embeddings = np.load(embeddings_path)
            logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")
            return embeddings

        logger.info(f"Model: {self.config.sapbert_model}")
        logger.info(f"Device: {self.config.embedding_device}")
        logger.info(f"Batch size: {self.config.embedding_batch_size}")
        logger.info(f"Processing {len(self.entities)} entities...")

        # Load SapBERT model
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SapBERT model...")
        model = SentenceTransformer(self.config.sapbert_model, device=self.config.embedding_device)

        # Encode entities in batches
        logger.info("Encoding entities...")
        embeddings = model.encode(
            self.entities,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        logger.info(f"✅ Generated embeddings shape: {embeddings.shape}")

        # Save
        if self.config.save_intermediate:
            np.save(embeddings_path, embeddings)
            with open(entity_ids_path, 'w') as f:
                json.dump(self.entity_to_id, f, indent=2)
            logger.info(f"✅ Saved embeddings to: {embeddings_path}")
            logger.info(f"✅ Saved entity IDs to: {entity_ids_path}")

        # Evaluation
        self.evaluate_stage1(embeddings)

        return embeddings

    def evaluate_stage1(self, embeddings: np.ndarray) -> None:
        """Evaluate embedding quality"""
        logger.info("\n📊 Stage 1 Evaluation:")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Embedding dim: {embeddings.shape[1]}")
        logger.info(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
        logger.info(f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.3f}")

    # ========================================================================
    # STAGE 2: FAISS BLOCKING
    # ========================================================================

    def stage2_faiss_blocking(self, embeddings: np.ndarray, entity_types: Dict) -> List[Tuple[int, int, float]]:
        """
        Generate candidate pairs using FAISS approximate nearest neighbor search.

        Args:
            embeddings: (N, 768) embeddings matrix
            entity_types: Entity type information

        Returns:
            List of (entity1_id, entity2_id, similarity) tuples
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 2: FAISS BLOCKING")
        logger.info("="*80)

        candidates_path = self.stage_paths["stage2_candidates"]

        # Check cache
        if not self.config.force and candidates_path.exists():
            logger.info(f"Loading cached candidates from: {candidates_path}")
            candidate_pairs = []
            with open(candidates_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    candidate_pairs.append((data["entity1_id"], data["entity2_id"], data["similarity"]))
            logger.info(f"✅ Loaded {len(candidate_pairs)} candidate pairs")
            return candidate_pairs

        logger.info(f"K neighbors: {self.config.faiss_k_neighbors}")
        logger.info(f"Similarity threshold: {self.config.faiss_similarity_threshold}")
        logger.info(f"Processing {len(embeddings)} entities...")

        # Build FAISS index per entity type to prevent cross-type comparisons
        # Try to use GPU version first (faiss-gpu-cu12), fallback to CPU
        try:
            import faiss
            # Check if GPU is available
            if faiss.get_num_gpus() > 0:
                logger.info(f"  Using FAISS GPU (found {faiss.get_num_gpus()} GPU(s))")
                use_gpu = True
            else:
                logger.info("  Using FAISS CPU (no GPU available)")
                use_gpu = False
        except Exception as e:
            logger.warning(f"  FAISS GPU not available ({e}), using CPU version")
            import faiss
            use_gpu = False

        candidate_pairs = []

        # Group entities by type
        type_to_entities = {}
        for entity_id, entity_name in enumerate(self.entities):
            entity_type = entity_types[entity_name]["type"]
            if entity_type not in type_to_entities:
                type_to_entities[entity_type] = []
            type_to_entities[entity_type].append(entity_id)

        # Process each type separately
        for entity_type, entity_ids in type_to_entities.items():
            if len(entity_ids) < 2:
                continue  # Need at least 2 entities to compare

            logger.info(f"  Processing type '{entity_type}': {len(entity_ids)} entities")

            # Get embeddings for this type
            type_embeddings = embeddings[entity_ids].astype('float32')

            # Build FAISS index
            dim = type_embeddings.shape[1]
            if len(entity_ids) < 100:
                # For small datasets, use flat index (exact search)
                index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity for normalized vectors)
            else:
                # For larger datasets, use HNSW (approximate search)
                index = faiss.IndexHNSWFlat(dim, 32)  # 32 = M (number of connections)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 64

            index.add(type_embeddings)

            # Search k nearest neighbors for each entity
            k = min(self.config.faiss_k_neighbors, len(entity_ids))
            distances, indices = index.search(type_embeddings, k + 1)  # +1 to exclude self

            # Convert to candidate pairs
            for i, (dists, neighs) in enumerate(zip(distances, indices)):
                entity1_id = entity_ids[i]

                for dist, neigh_idx in zip(dists, neighs):
                    if neigh_idx == i:  # Skip self
                        continue

                    entity2_id = entity_ids[neigh_idx]

                    # Similarity = inner product (for normalized vectors, this is cosine similarity)
                    similarity = float(dist)

                    # Filter by threshold
                    if similarity >= self.config.faiss_similarity_threshold:
                        # Ensure consistent ordering (smaller ID first)
                        if entity1_id < entity2_id:
                            candidate_pairs.append((entity1_id, entity2_id, similarity))

        logger.info(f"✅ Generated {len(candidate_pairs)} candidate pairs")

        # Save
        if self.config.save_intermediate:
            with open(candidates_path, 'w') as f:
                for e1_id, e2_id, sim in candidate_pairs:
                    f.write(json.dumps({
                        "entity1_id": e1_id,
                        "entity2_id": e2_id,
                        "entity1_name": self.id_to_entity[e1_id],
                        "entity2_name": self.id_to_entity[e2_id],
                        "similarity": sim
                    }) + '\n')
            logger.info(f"✅ Saved to: {candidates_path}")

        # Evaluation
        self.evaluate_stage2(candidate_pairs, entity_types)

        return candidate_pairs

    def evaluate_stage2(self, candidate_pairs: List, entity_types: Dict) -> None:
        """Evaluate blocking quality"""
        logger.info("\n📊 Stage 2 Evaluation:")
        logger.info(f"  Candidate pairs: {len(candidate_pairs)}")

        if len(candidate_pairs) > 0:
            similarities = [sim for _, _, sim in candidate_pairs]
            logger.info(f"  Similarity range: [{min(similarities):.3f}, {max(similarities):.3f}]")
            logger.info(f"  Mean similarity: {np.mean(similarities):.3f}")

        # Reduction ratio
        n = len(self.entities)
        max_pairs = n * (n - 1) // 2
        reduction = (1 - len(candidate_pairs) / max_pairs) * 100 if max_pairs > 0 else 0
        logger.info(f"  Reduction: {reduction:.1f}% (from {max_pairs} to {len(candidate_pairs)})")

    # ========================================================================
    # STAGE 3: MULTI-FEATURE SCORING
    # ========================================================================

    def stage3_multifeature_scoring(self, candidate_pairs: List, embeddings: np.ndarray, entity_types: Dict) -> List[Dict]:
        """
        Calculate comprehensive similarity scores using 5 features.

        Args:
            candidate_pairs: List of (entity1_id, entity2_id, sapbert_sim)
            embeddings: Entity embeddings
            entity_types: Entity type information

        Returns:
            List of scored pairs with feature breakdown
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 3: MULTI-FEATURE SCORING")
        logger.info("="*80)

        scores_path = self.stage_paths["stage3_scores"]

        # Check cache
        if not self.config.force and scores_path.exists():
            logger.info(f"Loading cached scores from: {scores_path}")
            scored_pairs = []
            with open(scores_path, 'r') as f:
                for line in f:
                    scored_pairs.append(json.loads(line))
            logger.info(f"✅ Loaded {len(scored_pairs)} scored pairs")
            return scored_pairs

        logger.info(f"Feature weights: {self.config.feature_weights}")
        logger.info(f"Processing {len(candidate_pairs)} pairs...")

        # TODO: Implement multi-feature scoring
        # Features:
        # 1. SapBERT similarity (from blocking)
        # 2. Lexical similarity (edit distance, Jaccard, etc.)
        # 3. Type consistency (same type = 1.0, else 0.0)
        # 4. Graph similarity (shared neighbors)
        # 5. UMLS alignment (disabled = 0.0)

        scored_pairs = []

        # Save
        if self.config.save_intermediate:
            with open(scores_path, 'w') as f:
                for pair in scored_pairs:
                    f.write(json.dumps(pair) + '\n')
            logger.info(f"✅ Saved to: {scores_path}")

        # Evaluation
        self.evaluate_stage3(scored_pairs)

        return scored_pairs

    def evaluate_stage3(self, scored_pairs: List[Dict]) -> None:
        """Evaluate scoring quality"""
        logger.info("\n📊 Stage 3 Evaluation:")
        logger.info(f"  Scored pairs: {len(scored_pairs)}")

        if len(scored_pairs) > 0:
            final_scores = [p["final_score"] for p in scored_pairs]
            logger.info(f"  Score range: [{min(final_scores):.3f}, {max(final_scores):.3f}]")
            logger.info(f"  Mean score: {np.mean(final_scores):.3f}")
            logger.info(f"  Median score: {np.median(final_scores):.3f}")

    # ========================================================================
    # STAGE 4: ADAPTIVE THRESHOLDING
    # ========================================================================

    def stage4_adaptive_thresholding(self, scored_pairs: List[Dict], entity_types: Dict) -> List[Tuple[int, int]]:
        """
        Apply type-specific thresholds to decide equivalence.

        Args:
            scored_pairs: Scored pairs with features
            entity_types: Entity type information

        Returns:
            List of (entity1_id, entity2_id) equivalent pairs
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 4: ADAPTIVE THRESHOLDING")
        logger.info("="*80)

        equivalents_path = self.stage_paths["stage4_equivalents"]

        # Check cache
        if not self.config.force and equivalents_path.exists():
            logger.info(f"Loading cached equivalents from: {equivalents_path}")
            equivalent_pairs = []
            with open(equivalents_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    equivalent_pairs.append((data["entity1_id"], data["entity2_id"]))
            logger.info(f"✅ Loaded {len(equivalent_pairs)} equivalent pairs")
            return equivalent_pairs

        logger.info(f"Type-specific thresholds: {self.config.type_thresholds}")
        logger.info(f"Processing {len(scored_pairs)} scored pairs...")

        # TODO: Implement adaptive thresholding
        # 1. Get entity type for each pair
        # 2. Apply type-specific threshold
        # 3. Keep pairs above threshold

        equivalent_pairs = []

        # Save
        if self.config.save_intermediate:
            with open(equivalents_path, 'w') as f:
                for e1_id, e2_id in equivalent_pairs:
                    f.write(json.dumps({
                        "entity1_id": e1_id,
                        "entity2_id": e2_id,
                        "entity1_name": self.id_to_entity[e1_id],
                        "entity2_name": self.id_to_entity[e2_id]
                    }) + '\n')
            logger.info(f"✅ Saved to: {equivalents_path}")

        # Evaluation
        self.evaluate_stage4(equivalent_pairs, scored_pairs)

        return equivalent_pairs

    def evaluate_stage4(self, equivalent_pairs: List, scored_pairs: List) -> None:
        """Evaluate thresholding quality"""
        logger.info("\n📊 Stage 4 Evaluation:")
        logger.info(f"  Equivalent pairs: {len(equivalent_pairs)}")

        if len(scored_pairs) > 0:
            acceptance_rate = 100 * len(equivalent_pairs) / len(scored_pairs)
            logger.info(f"  Acceptance rate: {acceptance_rate:.1f}%")

    # ========================================================================
    # STAGE 5: CLUSTERING & CANONICALIZATION
    # ========================================================================

    def stage5_clustering_canonicalization(self, equivalent_pairs: List[Tuple[int, int]]) -> List[Tuple[str, str, str]]:
        """
        Group synonyms into clusters and select canonical names.

        Args:
            equivalent_pairs: List of (entity1_id, entity2_id)

        Returns:
            List of (entity, "SYNONYM_OF", canonical_entity) edges
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 5: CLUSTERING & CANONICALIZATION")
        logger.info("="*80)

        clusters_path = self.stage_paths["stage5_clusters"]
        canonical_path = self.stage_paths["stage5_canonical"]

        # Check cache
        if not self.config.force and clusters_path.exists():
            logger.info(f"Loading cached clusters from: {clusters_path}")
            with open(clusters_path, 'r') as f:
                clusters = json.load(f)
            with open(canonical_path, 'r') as f:
                canonical_names = json.load(f)
            logger.info(f"✅ Loaded {len(clusters)} clusters")

            # Reconstruct synonym edges
            synonym_edges = []
            for cluster_id, entity_ids in clusters.items():
                canonical_id = canonical_names[cluster_id]
                canonical_name = self.id_to_entity[canonical_id]
                for entity_id in entity_ids:
                    if entity_id != canonical_id:
                        entity_name = self.id_to_entity[entity_id]
                        synonym_edges.append((entity_name, "SYNONYM_OF", canonical_name))

            return synonym_edges

        logger.info(f"Method: {self.config.canonical_selection_method}")
        logger.info(f"Processing {len(equivalent_pairs)} equivalent pairs...")

        # TODO: Implement clustering & canonicalization
        # 1. Union-Find clustering
        # 2. Select canonical name per cluster
        # 3. Generate SYNONYM_OF edges

        clusters = {}
        canonical_names = {}
        synonym_edges = []

        # Save
        if self.config.save_intermediate:
            with open(clusters_path, 'w') as f:
                json.dump(clusters, f, indent=2)
            with open(canonical_path, 'w') as f:
                json.dump(canonical_names, f, indent=2)
            logger.info(f"✅ Saved clusters to: {clusters_path}")
            logger.info(f"✅ Saved canonical names to: {canonical_path}")

        # Evaluation
        self.evaluate_stage5(clusters, synonym_edges)

        return synonym_edges

    def evaluate_stage5(self, clusters: Dict, synonym_edges: List) -> None:
        """Evaluate clustering quality"""
        logger.info("\n📊 Stage 5 Evaluation:")
        logger.info(f"  Number of clusters: {len(clusters)}")
        logger.info(f"  SYNONYM_OF edges: {len(synonym_edges)}")

        if len(clusters) > 0:
            cluster_sizes = [len(entity_ids) for entity_ids in clusters.values()]
            logger.info(f"  Avg cluster size: {np.mean(cluster_sizes):.1f}")
            logger.info(f"  Max cluster size: {max(cluster_sizes)}")


@hydra.main(version_base=None, config_path="config", config_name="stage2_entity_resolution")
def main(cfg: DictConfig) -> None:
    """Main entry point for Stage 2 entity resolution"""

    logger.info("="*80)
    logger.info("STAGE 2: ENTITY RESOLUTION PIPELINE")
    logger.info("="*80)
    logger.info(f"\nConfig:\n{cfg}\n")

    # Create config object
    config = EntityResolutionConfig(
        kg_input_path=cfg.kg_input_path,
        output_dir=cfg.output_dir,
        type_inference_method=cfg.type_inference.method,
        sapbert_model=cfg.sapbert.model,
        embedding_batch_size=cfg.sapbert.batch_size,
        embedding_device=cfg.sapbert.device,
        faiss_k_neighbors=cfg.faiss.k_neighbors,
        faiss_similarity_threshold=cfg.faiss.similarity_threshold,
        feature_weights=dict(cfg.scoring.feature_weights),
        type_thresholds=dict(cfg.thresholding.type_thresholds),
        canonical_selection_method=cfg.clustering.canonical_method,
        num_processes=cfg.num_processes,
        force=cfg.force,
        save_intermediate=cfg.save_intermediate,
    )

    # Run pipeline
    pipeline = EntityResolutionPipeline(config)
    pipeline.run()

    logger.info("\n✅ Stage 2 completed successfully!")


if __name__ == "__main__":
    main()
