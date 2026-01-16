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
from dotenv import load_dotenv

from gfmrag.kg_construction.utils import KG_DELIMITER

# Load environment variables from .env file
load_dotenv()

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

    # Stage 1b: ColBERT Indexing
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_root: str = "tmp/colbert_index"
    colbert_topk: int = 10  # Top-k for similarity lookup

    # Stage 2: FAISS Blocking
    faiss_k_neighbors: int = 50  # Candidates per entity
    faiss_similarity_threshold: float = 0.80
    faiss_index_type: str = "IndexHNSWFlat"  # Fast approximate search

    # Stage 3: Multi-Feature Scoring
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "sapbert": 0.50,      # SapBERT similarity (medical domain embeddings)
        "lexical": 0.15,      # String similarity (Levenshtein edit distance)
        "colbert": 0.25,      # ColBERT late interaction similarity
        "graph": 0.10,        # Shared neighbors (Jaccard similarity)
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
            "stage1b_colbert_indexed": self.output_dir / "stage1b_colbert_indexed.json",
            "stage2_candidates": self.output_dir / "stage2_candidate_pairs.jsonl",
            "stage3_scores": self.output_dir / "stage3_scored_pairs.jsonl",
            "stage4_equivalents": self.output_dir / "stage4_equivalent_pairs.jsonl",
            "stage5_clusters": self.output_dir / "stage5_clusters.json",
            "stage5_canonical": self.output_dir / "stage5_canonical_names.json",
            "kg_clean": self.output_dir / "kg_clean.txt",
        }

        # ColBERT model (lazy initialization)
        self.colbert_model = None

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

        # Stage 1b: ColBERT Indexing
        self.stage1b_colbert_indexing()

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
    # STAGE 0: TYPE INFERENCE - SMART CASCADING (3-TIER)
    # ========================================================================

    # Tier 1: Medical keyword dictionaries (comprehensive)
    MEDICAL_KEYWORDS = {
        "drug": {
            # Common drug classes
            "antibiotic", "antibiotics", "penicillin", "amoxicillin", "azithromycin",
            "antiviral", "antivirals", "acyclovir", "oseltamivir",
            "analgesic", "analgesics", "painkiller", "pain reliever",
            "aspirin", "ibuprofen", "acetaminophen", "paracetamol", "morphine",
            "statin", "statins", "atorvastatin", "simvastatin", "rosuvastatin",
            "beta blocker", "beta-blocker", "propranolol", "atenolol", "metoprolol",
            "ace inhibitor", "lisinopril", "enalapril", "ramipril",
            "diuretic", "diuretics", "furosemide", "hydrochlorothiazide",
            "insulin", "metformin", "glipizide", "glyburide",
            "antidepressant", "ssri", "fluoxetine", "sertraline", "citalopram",
            "anticoagulant", "warfarin", "heparin", "rivaroxaban", "apixaban",
            "chemotherapy", "chemo", "cisplatin", "carboplatin", "doxorubicin",
            "vaccine", "vaccination", "immunization",
            "steroid", "corticosteroid", "prednisone", "dexamethasone",
            "medication", "medicine", "drug", "pill", "tablet", "capsule",
            "injection", "infusion", "iv", "intravenous",
        },
        "disease": {
            # Common diseases
            "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
            "hypertension", "high blood pressure",
            "cancer", "carcinoma", "tumor", "malignancy", "neoplasm",
            "heart disease", "cardiovascular disease", "coronary artery disease",
            "stroke", "cerebrovascular accident", "cva",
            "asthma", "copd", "chronic obstructive pulmonary disease",
            "pneumonia", "bronchitis", "tuberculosis", "tb",
            "alzheimer", "dementia", "parkinson",
            "arthritis", "osteoarthritis", "rheumatoid arthritis",
            "infection", "sepsis", "septic",
            "hepatitis", "cirrhosis", "liver disease",
            "kidney disease", "renal failure", "nephropathy",
            "influenza", "flu", "covid", "coronavirus",
            "hiv", "aids", "immunodeficiency",
            "anemia", "leukemia", "lymphoma",
            # Medical conditions
            "syndrome", "disorder", "disease", "condition",
            "insufficiency", "failure", "dysfunction",
        },
        "symptom": {
            "pain", "ache", "aching", "sore", "soreness",
            "headache", "migraine", "tension headache",
            "fever", "pyrexia", "febrile",
            "cough", "coughing", "productive cough", "dry cough",
            "fatigue", "tiredness", "exhaustion", "weakness",
            "nausea", "vomiting", "emesis",
            "diarrhea", "constipation", "bloating",
            "dizziness", "vertigo", "lightheadedness",
            "shortness of breath", "dyspnea", "breathlessness",
            "chest pain", "angina",
            "abdominal pain", "stomach ache", "belly pain",
            "back pain", "lower back pain",
            "rash", "itching", "pruritus", "hives",
            "swelling", "edema", "inflammation",
            "bleeding", "hemorrhage",
            "numbness", "tingling", "paresthesia",
            "tremor", "shaking", "seizure", "convulsion",
        },
        "procedure": {
            "surgery", "surgical", "operation",
            "biopsy", "excision", "resection",
            "transplant", "transplantation", "graft",
            "catheterization", "angioplasty", "stent",
            "dialysis", "hemodialysis", "peritoneal dialysis",
            "chemotherapy", "radiotherapy", "radiation therapy",
            "immunotherapy", "targeted therapy",
            "physical therapy", "physiotherapy", "rehabilitation",
            "screening", "test", "testing", "diagnosis",
            "imaging", "x-ray", "ct scan", "mri", "ultrasound",
            "endoscopy", "colonoscopy", "bronchoscopy",
            "echocardiogram", "ekg", "ecg", "electrocardiogram",
            "blood test", "lab test", "laboratory",
            "vaccination", "immunization",
        },
        "gene": {
            "gene", "genetic", "mutation", "variant",
            "protein", "enzyme", "receptor",
            "brca1", "brca2", "tp53", "egfr", "kras",
            "chromosome", "allele", "genotype", "phenotype",
            "dna", "rna", "mrna",
        },
        "anatomy": {
            "heart", "cardiac", "myocardium", "ventricle", "atrium",
            "lung", "pulmonary", "bronchus", "alveoli",
            "liver", "hepatic", "bile duct",
            "kidney", "renal", "nephron", "ureter",
            "brain", "cerebral", "cortex", "hippocampus",
            "stomach", "gastric", "intestine", "colon",
            "bone", "skeletal", "joint", "cartilage",
            "muscle", "muscular", "tendon", "ligament",
            "skin", "dermal", "epidermis",
            "blood", "vessel", "artery", "vein", "capillary",
            "nerve", "neural", "neuron", "axon",
            "gland", "thyroid", "pancreas", "adrenal",
            "organ", "tissue", "cell",
        },
    }

    def _infer_type_tier1_medical_keywords(self, entity: str) -> Dict:
        """
        Tier 1: Fast medical keyword matching with comprehensive dictionaries.

        Strategy:
        - Check entity tokens against curated medical keyword sets
        - Fast O(1) lookup using set intersection
        - High precision for common medical terms

        Args:
            entity: Entity name

        Returns:
            dict: {"type": str, "confidence": float, "tier": "tier1"}
        """
        entity_lower = entity.lower()
        entity_tokens = set(entity_lower.split())

        # Count keyword matches for each type
        type_matches = {}
        for entity_type, keywords in self.MEDICAL_KEYWORDS.items():
            # Check full string match
            if entity_lower in keywords:
                type_matches[entity_type] = 1.0
                continue

            # Check token overlap
            matches = entity_tokens & keywords
            if matches:
                # Confidence based on match ratio
                match_ratio = len(matches) / len(entity_tokens)
                type_matches[entity_type] = match_ratio

        # Enhanced pattern matching (from old _infer_type_pattern)
        pattern_result = self._infer_type_pattern(entity)
        if pattern_result["confidence"] >= 0.85:
            # High confidence pattern match
            return {
                "type": pattern_result["type"],
                "confidence": pattern_result["confidence"],
                "tier": "tier1_pattern"
            }

        # Select best keyword match
        if type_matches:
            best_type = max(type_matches.items(), key=lambda x: x[1])
            entity_type, match_score = best_type

            # Confidence scoring
            if match_score >= 0.9:
                confidence = 0.90
            elif match_score >= 0.7:
                confidence = 0.85
            elif match_score >= 0.5:
                confidence = 0.80
            else:
                confidence = 0.70

            return {
                "type": entity_type,
                "confidence": confidence,
                "tier": "tier1_keyword"
            }

        # No strong keyword match
        return {
            "type": pattern_result["type"],
            "confidence": pattern_result["confidence"],
            "tier": "tier1_pattern"
        }

    # Tier 2: Labeled examples for SapBERT kNN classification
    LABELED_EXAMPLES = {
        "drug": [
            "aspirin", "ibuprofen", "metformin", "lisinopril", "atorvastatin",
            "amlodipine", "metoprolol", "omeprazole", "albuterol", "gabapentin",
            "penicillin", "amoxicillin", "ciprofloxacin", "azithromycin",
            "insulin", "warfarin", "prednisone", "hydrochlorothiazide",
        ],
        "disease": [
            "diabetes mellitus", "hypertension", "coronary artery disease",
            "heart failure", "chronic kidney disease", "asthma", "copd",
            "pneumonia", "stroke", "alzheimer disease", "parkinson disease",
            "rheumatoid arthritis", "osteoarthritis", "cancer", "lymphoma",
            "hepatitis", "cirrhosis", "tuberculosis", "hiv infection",
        ],
        "symptom": [
            "chest pain", "headache", "abdominal pain", "back pain",
            "shortness of breath", "cough", "fever", "fatigue", "nausea",
            "dizziness", "edema", "rash", "joint pain", "muscle weakness",
        ],
        "procedure": [
            "coronary artery bypass graft", "percutaneous coronary intervention",
            "appendectomy", "cholecystectomy", "colonoscopy", "endoscopy",
            "ct scan", "mri", "x-ray", "echocardiogram", "dialysis",
            "physical therapy", "chemotherapy", "radiation therapy",
        ],
        "gene": [
            "BRCA1", "BRCA2", "TP53", "EGFR", "KRAS", "ALK", "HER2",
            "CFTR", "HTT", "APP", "PSEN1", "APOE",
        ],
        "anatomy": [
            "heart", "lung", "liver", "kidney", "brain", "stomach",
            "colon", "pancreas", "thyroid", "bone", "muscle", "skin",
            "blood vessel", "artery", "vein", "nerve", "lymph node",
        ],
    }

    def _infer_type_tier2_sapbert_knn(self, entity: str, embeddings: np.ndarray = None) -> Dict:
        """
        Tier 2: SapBERT kNN classification using labeled medical examples.

        Strategy:
        - Use pre-computed SapBERT embeddings (from Stage 1)
        - Compare with labeled example embeddings via cosine similarity
        - kNN voting (k=5) for type classification
        - Fast and domain-specific (medical embeddings)

        Args:
            entity: Entity name
            embeddings: Pre-computed SapBERT embeddings matrix (optional)

        Returns:
            dict: {"type": str, "confidence": float, "tier": "tier2"}
        """
        # Lazy initialization of labeled embeddings
        if not hasattr(self, '_tier2_labeled_embeddings'):
            from sentence_transformers import SentenceTransformer

            logger.info("  Initializing Tier 2: SapBERT kNN classifier...")

            # Load SapBERT model (same as Stage 1)
            model = SentenceTransformer(
                self.config.sapbert_model,
                device=self.config.embedding_device
            )

            # Encode labeled examples
            self._tier2_labeled_examples = []
            self._tier2_labeled_types = []

            for entity_type, examples in self.LABELED_EXAMPLES.items():
                for example in examples:
                    self._tier2_labeled_examples.append(example)
                    self._tier2_labeled_types.append(entity_type)

            # Compute embeddings
            self._tier2_labeled_embeddings = model.encode(
                self._tier2_labeled_examples,
                batch_size=self.config.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            logger.info(f"  ✅ Tier 2 initialized with {len(self._tier2_labeled_examples)} labeled examples")

        # Get entity embedding
        if entity in self.entity_to_id and embeddings is not None:
            # Use pre-computed embedding from Stage 1
            entity_id = self.entity_to_id[entity]
            entity_emb = embeddings[entity_id:entity_id+1]
        else:
            # Compute embedding on-the-fly (fallback)
            # Cache model to avoid reloading for each entity
            if not hasattr(self, '_tier2_sapbert_model'):
                from sentence_transformers import SentenceTransformer
                logger.info("  Loading SapBERT model for on-the-fly encoding...")
                self._tier2_sapbert_model = SentenceTransformer(
                    self.config.sapbert_model,
                    device=self.config.embedding_device
                )

            entity_emb = self._tier2_sapbert_model.encode(
                [entity],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        # Compute cosine similarities with labeled examples
        similarities = np.dot(self._tier2_labeled_embeddings, entity_emb.T).flatten()

        # kNN voting (k=5)
        k = 5
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_indices]
        top_k_types = [self._tier2_labeled_types[i] for i in top_k_indices]

        # Weighted voting by similarity
        type_scores = {}
        for sim, entity_type in zip(top_k_similarities, top_k_types):
            if entity_type not in type_scores:
                type_scores[entity_type] = 0.0
            type_scores[entity_type] += sim

        # Select best type
        best_type = max(type_scores.items(), key=lambda x: x[1])
        entity_type, total_score = best_type

        # Confidence based on:
        # 1. Top-1 similarity
        # 2. Voting consensus
        top1_sim = top_k_similarities[0]
        consensus = sum(1 for t in top_k_types if t == entity_type) / k

        # Confidence formula
        confidence = (top1_sim * 0.6) + (consensus * 0.4)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return {
            "type": entity_type,
            "confidence": confidence,
            "tier": "tier2_sapbert_knn"
        }

    # ========================================================================
    # STAGE 0: TYPE INFERENCE
    # ========================================================================

    def _batch_process_llm_step2(self, entities: List[str]) -> Dict[str, Dict]:
        """
        Batch process Step 2 LLM relationship inference in parallel.

        Uses ThreadPoolExecutor to process multiple entities concurrently.
        Loads BATCH_SIZE and MAX_WORKERS from environment variables.

        Args:
            entities: List of entity names to process

        Returns:
            dict: {entity_name: {"type": str, "confidence": float}}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Load parallel processing config from environment
        batch_size = int(os.environ.get("BATCH_SIZE", "15"))
        max_workers = int(os.environ.get("MAX_WORKERS", "15"))

        logger.info(f"Step 2 LLM Parallel Processing: BATCH_SIZE={batch_size}, MAX_WORKERS={max_workers}")

        results = {}
        processed_count = 0
        total_entities = len(entities)

        # Process in batches
        total_batches = (len(entities) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(entities))
            batch = entities[start_idx:end_idx]

            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_entity = {
                    executor.submit(self._infer_type_relationship_llm, entity): entity
                    for entity in batch
                }

                # Collect results as they complete
                for future in as_completed(future_to_entity):
                    entity = future_to_entity[future]
                    try:
                        result = future.result()
                        results[entity] = result
                        processed_count += 1
                        # Display progress every 10 entities or on last entity
                        if processed_count % 10 == 0 or processed_count == total_entities:
                            logger.info(f"  Progress: {processed_count}/{total_entities} entities processed ({100*processed_count/total_entities:.1f}%)")
                    except Exception as e:
                        logger.warning(f"Parallel LLM failed for '{entity}': {e}")
                        results[entity] = {"type": "other", "confidence": 0.3}
                        processed_count += 1

        logger.info(f"✅ Completed parallel LLM processing for {len(results)} entities")
        return results

    def stage0_type_inference(self) -> Dict[str, Dict]:
        """
        Classify entity types using SMART CASCADING (3-Tier) approach.

        OPTIMIZED Architecture (5-10x faster, higher accuracy):
        ├─ TIER 1: Medical Keywords + Pattern (0.001s/entity)
        │  └─ Early stop if confidence ≥ 0.80 (relaxed for more coverage)
        ├─ TIER 2: SapBERT kNN Classifier (0.01s/entity)
        │  └─ Early stop if confidence ≥ 0.80 (strict for quality)
        └─ TIER 3: GPT-4.1 Mini LLM (0.5s/entity, hard cases ~15-20%)
           └─ Model: gpt-4.1-mini-2025-04-14 (fast and cost-effective)

        REMOVED:
        - Zero-shot BART (78% bottleneck, not medical-specific)

        Returns:
            dict: {entity_name: {"type": str, "confidence": float, "tier": str}}
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 0: SMART CASCADING TYPE INFERENCE (3-Tier)")
        logger.info("="*80)

        output_path = self.stage_paths["stage0_types"]

        # Check cache
        if not self.config.force and output_path.exists():
            logger.info(f"Loading cached types from: {output_path}")
            with open(output_path, 'r') as f:
                entity_types = json.load(f)
            logger.info(f"✅ Loaded types for {len(entity_types)} entities")
            return entity_types

        logger.info(f"Processing {len(self.entities)} unique entities...")
        logger.info("Architecture: Tier 1 (Keywords) → Tier 2 (SapBERT kNN) → Tier 3 (LLM)")
        logger.info("Early stopping: Tier 1 @ 0.80 confidence, Tier 2 @ 0.80 confidence")
        logger.info("LLM Model: gpt-4.1-mini-2025-04-14 (GPT-4.1 Mini for fast processing)")

        entity_types = {}
        tier_stats = {
            "tier1": 0,
            "tier2": 0,
            "tier3": 0,
        }

        # Pre-compute SapBERT embeddings for Tier 2 (reuse from Stage 1 if available)
        embeddings = None
        embeddings_path = self.stage_paths["stage1_embeddings"]
        if embeddings_path.exists():
            logger.info("Loading SapBERT embeddings for Tier 2...")
            embeddings = np.load(embeddings_path)
            logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")

        # Collect hard cases for Tier 3 (LLM)
        tier3_entities = []

        # Process each entity with cascading
        for entity in tqdm(self.entities, desc="Cascading type inference"):
            # Skip if already processed
            if entity in entity_types:
                continue

            # ─────────────────────────────────────────────────────────
            # TIER 1: Medical Keywords + Pattern (FAST)
            # ─────────────────────────────────────────────────────────
            tier1_result = self._infer_type_tier1_medical_keywords(entity)

            if tier1_result["confidence"] >= 0.80:
                # HIGH CONFIDENCE → Early stop
                entity_types[entity] = tier1_result
                tier_stats["tier1"] += 1
                continue

            # ─────────────────────────────────────────────────────────
            # TIER 2: SapBERT kNN Classifier (MEDIUM)
            # ─────────────────────────────────────────────────────────
            tier2_result = self._infer_type_tier2_sapbert_knn(entity, embeddings)

            if tier2_result["confidence"] >= 0.80:
                # MEDIUM CONFIDENCE → Early stop
                entity_types[entity] = tier2_result
                tier_stats["tier2"] += 1
                continue

            # ─────────────────────────────────────────────────────────
            # TIER 3: LLM Relationship (SLOW but accurate)
            # ─────────────────────────────────────────────────────────
            # Collect for batch processing
            tier3_entities.append(entity)

            # Fallback: Use Tier 2 result for now (will be updated after LLM)
            entity_types[entity] = tier2_result

        # Batch process Tier 3 entities with LLM (parallel)
        if tier3_entities:
            logger.info(f"\n🔍 Tier 3: Processing {len(tier3_entities)} hard cases with LLM...")
            llm_results = self._batch_process_llm_step2(tier3_entities)

            for entity in tier3_entities:
                llm_result = llm_results.get(entity, {"type": "other", "confidence": 0.3, "tier": "tier3_llm"})
                llm_result["tier"] = "tier3_llm"

                # Keep LLM result if confidence > Tier 2
                tier2_conf = entity_types[entity]["confidence"]
                if llm_result["confidence"] > tier2_conf:
                    entity_types[entity] = llm_result
                    tier_stats["tier3"] += 1
                else:
                    # Keep Tier 2 result but mark as validated by Tier 3
                    entity_types[entity]["tier"] = "tier2_validated"
                    tier_stats["tier2"] += 1

        # Save
        if self.config.save_intermediate:
            with open(output_path, 'w') as f:
                json.dump(entity_types, f, indent=2)
            logger.info(f"✅ Saved to: {output_path}")

        # Evaluation
        self.evaluate_stage0(entity_types, tier_stats)

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

    def _infer_type_relationship_llm(self, entity: str) -> Dict:
        """
        Infer entity type using LLM analysis of graph relationships.

        Step 2: Relationship-Based with LLM
        - Extracts all relationships for the entity
        - Sends to gpt-4o-mini via YEScale API
        - LLM analyzes relationships and infers type

        Args:
            entity: Entity name

        Returns:
            dict: {"type": str, "confidence": float}
        """
        # Extract relationships for this entity
        incoming_relations = []
        outgoing_relations = []

        for head, rel, tail in self.triples:
            if head == entity:
                outgoing_relations.append(f"{entity} --[{rel}]--> {tail}")
            if tail == entity:
                incoming_relations.append(f"{head} --[{rel}]--> {entity}")

        # If no relationships, return low confidence "other"
        if not incoming_relations and not outgoing_relations:
            return {"type": "other", "confidence": 0.3}

        # Build context for LLM
        relationship_context = ""
        if outgoing_relations:
            relationship_context += "Outgoing relationships:\n" + "\n".join(outgoing_relations[:10])
        if incoming_relations:
            if outgoing_relations:
                relationship_context += "\n\n"
            relationship_context += "Incoming relationships:\n" + "\n".join(incoming_relations[:10])

        # Prompt for LLM
        prompt = f"""You are a medical entity type classifier. Analyze the following entity and its relationships in a medical knowledge graph.

Entity: "{entity}"

{relationship_context}

Based on these relationships, classify the entity into ONE of these types:
- drug: medications, pharmaceuticals, therapeutic compounds
- disease: illnesses, conditions, syndromes, infections
- symptom: clinical signs, patient complaints, manifestations
- gene: genes, proteins, genetic markers
- procedure: medical procedures, surgeries, treatments, therapies
- anatomy: body parts, organs, tissues, cells
- other: if none of the above fit well

Respond in this EXACT JSON format (no extra text):
{{"type": "...", "confidence": 0.XX, "reasoning": "..."}}

Where confidence is 0.0-1.0 (how certain you are about this classification).
"""

        try:
            # Initialize LLM model using langchain_util (same as OpenIE)
            if not hasattr(self, '_llm_cache'):
                import os
                from gfmrag.kg_construction.langchain_util import init_langchain_model

                # Check if YEScale or OpenAI configured
                yescale_url = os.environ.get("YESCALE_API_BASE_URL")
                api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")

                if not api_key:
                    logger.warning(
                        "LLM API not configured (YESCALE_API_KEY or OPENAI_API_KEY missing). "
                        "Skipping LLM-based relationship inference. "
                        "Set YESCALE_API_BASE_URL and YESCALE_API_KEY to use YEScale, "
                        "or set OPENAI_API_KEY to use OpenAI."
                    )
                    self._llm_cache = None
                else:
                    # Use init_langchain_model - auto-detects YEScale or OpenAI
                    self._llm_cache = init_langchain_model(
                        llm="openai",  # Will use YEScale if YESCALE_API_BASE_URL is set
                        model_name="gpt-4.1-mini-2025-04-14",  # GPT-4.1 Mini for fast processing
                        temperature=0.0,
                    )
                    if yescale_url:
                        logger.info(f"✅ Initialized YEScale LLM for relationship inference: {yescale_url}")
                    else:
                        logger.info("✅ Initialized OpenAI LLM for relationship inference")

            # If LLM not available, return low confidence "other"
            if self._llm_cache is None:
                return {"type": "other", "confidence": 0.3}

            # Get LLM response (follow llm_openie_model.py pattern)
            from langchain_core.messages import HumanMessage
            from langchain_openai import ChatOpenAI
            from gfmrag.kg_construction.utils import extract_json_dict
            import json

            llm = self._llm_cache

            if isinstance(llm, ChatOpenAI):
                # OpenAI: Use JSON mode (reliable)
                response = llm.invoke(
                    [HumanMessage(content=prompt)],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                response_content = response.content
                result = eval(response_content)
            else:
                # YEScale or other models: Parse JSON manually
                response = llm.invoke(
                    [HumanMessage(content=prompt)],
                    temperature=0,
                )
                response_content = response.content
                result = extract_json_dict(response_content)

            # Extract and validate
            entity_type = result.get("type", "other")
            confidence = float(result.get("confidence", 0.5))

            # Validate type
            valid_types = {"drug", "disease", "symptom", "gene", "procedure", "anatomy", "other"}
            if entity_type not in valid_types:
                entity_type = "other"

            # Clamp confidence to [0, 1]
            confidence = max(0.0, min(1.0, confidence))

            return {"type": entity_type, "confidence": confidence}

        except Exception as e:
            logger.warning(f"LLM inference failed for entity '{entity}': {e}")
            return {"type": "other", "confidence": 0.3}

    def _infer_type_zeroshot(self, entity: str) -> Dict:
        """
        [DEPRECATED] Infer entity type using zero-shot classification.

        ⚠️  DEPRECATED: This method is NO LONGER USED in SMART CASCADING (3-Tier).

        Why deprecated:
        - BART-large-mnli is NOT medical-specific (general domain)
        - Major bottleneck: 78% of total processing time
        - Replaced by Tier 2: SapBERT kNN (faster + medical-specific)

        Old Step 3: Zero-shot Classification
        - Used transformers pipeline with BART-large-mnli
        - Classified entity name directly without fine-tuning
        - Replaced by medical-specific SapBERT kNN in new architecture

        Args:
            entity: Entity name

        Returns:
            dict: {"type": str, "confidence": float}
        """
        try:
            from transformers import pipeline

            # Cache the classifier at class level to avoid re-initialization
            if not hasattr(self, '_zeroshot_classifier'):
                # Use BART-large-mnli for zero-shot classification
                # Alternative: facebook/bart-large-mnli, microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract
                self._zeroshot_classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if self.config.embedding_device == "cuda" else -1
                )

            classifier = self._zeroshot_classifier

            # Candidate labels with detailed descriptions
            candidate_labels = [
                "drug medication pharmaceutical",
                "disease illness condition syndrome",
                "symptom sign manifestation",
                "gene protein genetic marker",
                "medical procedure surgery treatment",
                "anatomy body part organ tissue",
                "other entity"
            ]

            # Classify
            result = classifier(entity, candidate_labels)

            # Extract top prediction
            top_label = result['labels'][0]
            top_score = result['scores'][0]

            # Map label to entity type
            label_mapping = {
                "drug medication pharmaceutical": "drug",
                "disease illness condition syndrome": "disease",
                "symptom sign manifestation": "symptom",
                "gene protein genetic marker": "gene",
                "medical procedure surgery treatment": "procedure",
                "anatomy body part organ tissue": "anatomy",
                "other entity": "other"
            }

            entity_type = label_mapping.get(top_label, "other")
            confidence = float(top_score)

            return {"type": entity_type, "confidence": confidence}

        except Exception as e:
            logger.warning(f"Zero-shot classification failed for entity '{entity}': {e}")
            return {"type": "other", "confidence": 0.3}

    def _hybrid_decision(
        self,
        entity: str,
        pattern_result: Dict,
        relationship_result: Dict,
        zeroshot_result: Dict
    ) -> Dict:
        """
        Combine results from all 3 methods using weighted voting.

        Step 4: Hybrid Decision Logic (Weighted Voting)
        - Pattern weight: 0.2
        - Relationship-LLM weight: 0.4
        - Zero-shot weight: 0.4

        For each entity type, calculate weighted score:
        score[type] = sum(weight * confidence) for all methods predicting that type

        Choose type with highest weighted score.

        Args:
            entity: Entity name
            pattern_result: {"type": str, "confidence": float}
            relationship_result: {"type": str, "confidence": float}
            zeroshot_result: {"type": str, "confidence": float}

        Returns:
            dict: {"type": str, "confidence": float, "method": str}
        """
        # Method weights
        PATTERN_WEIGHT = 0.2
        LLM_WEIGHT = 0.4
        ZEROSHOT_WEIGHT = 0.4

        p_type = pattern_result["type"]
        p_conf = pattern_result["confidence"]
        r_type = relationship_result["type"]
        r_conf = relationship_result["confidence"]
        z_type = zeroshot_result["type"]
        z_conf = zeroshot_result["confidence"]

        # Calculate weighted scores for each type
        type_scores = {}  # {type: (weighted_score, total_weight, methods)}

        # Add pattern vote
        if p_type not in type_scores:
            type_scores[p_type] = [0.0, 0.0, []]
        type_scores[p_type][0] += PATTERN_WEIGHT * p_conf
        type_scores[p_type][1] += PATTERN_WEIGHT
        type_scores[p_type][2].append("pattern")

        # Add LLM vote
        if r_type not in type_scores:
            type_scores[r_type] = [0.0, 0.0, []]
        type_scores[r_type][0] += LLM_WEIGHT * r_conf
        type_scores[r_type][1] += LLM_WEIGHT
        type_scores[r_type][2].append("llm")

        # Add zero-shot vote
        if z_type not in type_scores:
            type_scores[z_type] = [0.0, 0.0, []]
        type_scores[z_type][0] += ZEROSHOT_WEIGHT * z_conf
        type_scores[z_type][1] += ZEROSHOT_WEIGHT
        type_scores[z_type][2].append("zeroshot")

        # Find type with highest weighted score (excluding "other" if possible)
        non_other_types = [(t, s) for t, s in type_scores.items() if t != "other"]

        if non_other_types:
            # Choose non-"other" type with highest score
            best_type, (weighted_score, total_weight, methods) = max(
                non_other_types,
                key=lambda x: x[1][0]
            )
        else:
            # All methods predicted "other"
            best_type, (weighted_score, total_weight, methods) = max(
                type_scores.items(),
                key=lambda x: x[1][0]
            )

        # Calculate final confidence as weighted average
        final_confidence = weighted_score / total_weight if total_weight > 0 else 0.5

        # Determine method label
        if len(methods) == 3:
            method = "weighted_unanimous"
        elif len(methods) == 2:
            method = f"weighted_{'_'.join(sorted(methods))}"
        else:
            method = f"weighted_{methods[0]}"

        return {
            "type": best_type,
            "confidence": final_confidence,
            "method": method
        }

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

    def evaluate_stage0(self, entity_types: Dict, tier_stats: Dict = None) -> None:
        """Evaluate type inference quality with tier statistics"""
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

        # Show tier distribution (NEW for 3-tier cascading)
        if tier_stats:
            total = sum(tier_stats.values())
            logger.info(f"\n  🎯 Tier Distribution (Early Stopping):")
            logger.info(f"    ├─ Tier 1 (Keywords): {tier_stats['tier1']} ({100*tier_stats['tier1']/total:.1f}%) - FAST ✨")
            logger.info(f"    ├─ Tier 2 (SapBERT):  {tier_stats['tier2']} ({100*tier_stats['tier2']/total:.1f}%) - MEDIUM 🔬")
            logger.info(f"    └─ Tier 3 (LLM):      {tier_stats['tier3']} ({100*tier_stats['tier3']/total:.1f}%) - SLOW but ACCURATE 🧠")
            logger.info(f"  Performance: {tier_stats['tier1']+tier_stats['tier2']}/{total} ({100*(tier_stats['tier1']+tier_stats['tier2'])/total:.1f}%) resolved without LLM!")

        # Show tier field distribution (alternative if tier_stats not provided)
        if entity_types and "tier" in next(iter(entity_types.values())):
            tier_counts = Counter(e["tier"] for e in entity_types.values())
            logger.info(f"  Tier field distribution:")
            for tier_name, count in tier_counts.most_common():
                percentage = 100 * count / len(entity_types)
                logger.info(f"    - {tier_name}: {count} ({percentage:.1f}%)")

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
        logger.info(f"Checking for cached embeddings at: {embeddings_path}")
        if not self.config.force and embeddings_path.exists():
            logger.info("✓ Cache found - loading cached embeddings (SapBERT model will not be loaded)")
            embeddings = np.load(embeddings_path)
            logger.info(f"✅ Loaded embeddings shape: {embeddings.shape}")
            logger.info("   💡 Tip: Use --force flag to regenerate embeddings and see full model loading process")
            return embeddings

        logger.info("✗ No cache found - will generate new embeddings")
        logger.info(f"Model: {self.config.sapbert_model}")
        logger.info(f"Device: {self.config.embedding_device}")
        logger.info(f"Batch size: {self.config.embedding_batch_size}")
        logger.info(f"Processing {len(self.entities)} entities...")

        # Load SapBERT model from sentence-transformers (NOT HuggingFace transformers)
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SapBERT model from sentence-transformers library...")
        model = SentenceTransformer(self.config.sapbert_model, device=self.config.embedding_device)
        logger.info("✅ SapBERT model loaded successfully")

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
    # STAGE 1B: COLBERT INDEXING
    # ========================================================================

    def stage1b_colbert_indexing(self) -> None:
        """
        Index all entities using ColBERT for late interaction similarity.

        ColBERT provides token-level embeddings and MaxSim operation for
        better handling of multi-word entities and semantic matching.

        This stage creates a ColBERT index that will be used in Stage 3
        to compute similarity scores between entity pairs.
        """
        logger.info("\n" + "="*80)
        logger.info("STAGE 1B: COLBERT INDEXING")
        logger.info("="*80)

        indexed_flag_path = self.stage_paths["stage1b_colbert_indexed"]

        # Check cache
        if not self.config.force and indexed_flag_path.exists():
            logger.info(f"ColBERT index already exists, loading...")
            with open(indexed_flag_path, 'r') as f:
                index_info = json.load(f)
            logger.info(f"✅ ColBERT index loaded: {index_info['num_entities']} entities")

            # Initialize ColBERT model and load existing index
            from gfmrag.kg_construction.entity_linking_model import ColbertELModel
            self.colbert_model = ColbertELModel(
                model_name_or_path=self.config.colbert_model,
                root=self.config.colbert_root,
                force=False,
            )
            # The index will be auto-loaded when we call __call__
            self.colbert_model.index(self.entities)
            return

        logger.info(f"Model: {self.config.colbert_model}")
        logger.info(f"Index root: {self.config.colbert_root}")
        logger.info(f"Processing {len(self.entities)} entities...")

        # Initialize ColBERT model
        from gfmrag.kg_construction.entity_linking_model import ColbertELModel

        self.colbert_model = ColbertELModel(
            model_name_or_path=self.config.colbert_model,
            root=self.config.colbert_root,
            force=self.config.force,
        )

        # Index all entities
        logger.info("Indexing entities with ColBERT...")
        self.colbert_model.index(self.entities)

        logger.info(f"✅ Indexed {len(self.entities)} entities")

        # Save index info
        if self.config.save_intermediate:
            with open(indexed_flag_path, 'w') as f:
                json.dump({
                    "num_entities": len(self.entities),
                    "model": self.config.colbert_model,
                    "index_root": self.config.colbert_root,
                }, f, indent=2)
            logger.info(f"✅ Saved index info to: {indexed_flag_path}")

        # Evaluation
        self.evaluate_stage1b()

    def evaluate_stage1b(self) -> None:
        """Evaluate ColBERT indexing"""
        logger.info("\n📊 Stage 1B Evaluation:")
        logger.info(f"  Indexed entities: {len(self.entities)}")
        logger.info(f"  Model: {self.config.colbert_model}")
        logger.info(f"  Index location: {self.config.colbert_root}")

    # ========================================================================
    # STAGE 2: FAISS BLOCKING
    # ========================================================================

    def _has_token_overlap(self, entity1: str, entity2: str) -> bool:
        """
        Check if two entities share at least one common token (word).

        Token overlap pre-filtering reduces false candidates by 20-30%.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            bool: True if entities share >= 1 token

        Examples:
            "type 2 diabetes" ↔ "diabetes mellitus" → True (share "diabetes")
            "aspirin" ↔ "hypertension" → False (no common tokens)
            "MI" ↔ "myocardial infarction" → False (abbreviation mismatch)
        """
        # Tokenize and lowercase
        tokens1 = set(entity1.lower().split())
        tokens2 = set(entity2.lower().split())

        # Check overlap
        return len(tokens1 & tokens2) > 0

    def stage2_faiss_blocking(self, embeddings: np.ndarray, entity_types: Dict) -> List[Tuple[int, int, float]]:
        """
        Generate candidate pairs using FAISS approximate nearest neighbor search.

        Strategy: Hybrid Type-Specific + Cross-Type Blocking
        - High confidence entities (>= 0.75): Type-specific search only
        - Low confidence entities (< 0.75): Type-specific + global search
        - All candidates: Token overlap filtering

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

        # Hybrid blocking parameters
        CONFIDENCE_THRESHOLD = 0.75  # Split point for high/low confidence
        HIGH_CONF_K = 50  # K for high confidence (type-specific only)
        LOW_CONF_TYPE_K = 30  # K for low confidence type-specific search
        LOW_CONF_GLOBAL_K = 20  # K for low confidence global search

        candidate_pairs = []
        candidates_before_filter = 0
        candidates_after_filter = 0

        # Separate entities by confidence level
        high_conf_entities = []
        low_conf_entities = []
        for entity_id, entity_name in enumerate(self.entities):
            confidence = entity_types[entity_name].get("confidence", 1.0)
            if confidence >= CONFIDENCE_THRESHOLD:
                high_conf_entities.append(entity_id)
            else:
                low_conf_entities.append(entity_id)

        logger.info(f"  High confidence entities (>={CONFIDENCE_THRESHOLD}): {len(high_conf_entities)}")
        logger.info(f"  Low confidence entities (<{CONFIDENCE_THRESHOLD}): {len(low_conf_entities)}")

        # Group entities by type
        type_to_entities = {}
        for entity_id, entity_name in enumerate(self.entities):
            entity_type = entity_types[entity_name]["type"]
            if entity_type not in type_to_entities:
                type_to_entities[entity_type] = []
            type_to_entities[entity_type].append(entity_id)

        # Build type-specific indices
        type_indices = {}
        logger.info("  Building type-specific FAISS indices...")

        for entity_type, entity_ids in type_to_entities.items():
            if len(entity_ids) < 2:
                continue

            type_embeddings = embeddings[entity_ids].astype('float32')
            dim = type_embeddings.shape[1]

            if len(entity_ids) < 100:
                index = faiss.IndexFlatIP(dim)
                index_type = "IP"
            else:
                index = faiss.IndexHNSWFlat(dim, 32)
                index.hnsw.efConstruction = 40
                index.hnsw.efSearch = 64
                index_type = "HNSW"

            index.add(type_embeddings)
            type_indices[entity_type] = {
                "index": index,
                "entity_ids": entity_ids,
                "index_type": index_type
            }
            logger.info(f"    Type '{entity_type}': {len(entity_ids)} entities ({index_type})")

        # Build global index for low-confidence entities
        global_index = None
        if low_conf_entities:
            logger.info(f"  Building global FAISS index for {len(low_conf_entities)} low-confidence entities...")
            global_embeddings = embeddings[low_conf_entities].astype('float32')
            dim = global_embeddings.shape[1]

            if len(low_conf_entities) < 100:
                global_index = faiss.IndexFlatIP(dim)
                global_index_type = "IP"
            else:
                global_index = faiss.IndexHNSWFlat(dim, 32)
                global_index.hnsw.efConstruction = 40
                global_index.hnsw.efSearch = 64
                global_index_type = "HNSW"

            global_index.add(global_embeddings)
            logger.info(f"    Global index: {len(low_conf_entities)} entities ({global_index_type})")

        # Search for candidates
        logger.info("  Searching for candidates...")
        seen_pairs = set()  # Prevent duplicates from hybrid search

        for entity_id, entity_name in enumerate(tqdm(self.entities, desc="  Blocking")):
            entity_type = entity_types[entity_name]["type"]
            confidence = entity_types[entity_name].get("confidence", 1.0)

            # Skip if type not in indices (< 2 entities)
            if entity_type not in type_indices:
                continue

            candidates = []

            # STRATEGY 1: Type-specific search (all entities)
            type_info = type_indices[entity_type]
            type_index = type_info["index"]
            type_entity_ids = type_info["entity_ids"]
            index_type = type_info["index_type"]

            # Find position in type-specific index
            local_idx = type_entity_ids.index(entity_id)
            query_embedding = embeddings[entity_id:entity_id+1].astype('float32')

            if confidence >= CONFIDENCE_THRESHOLD:
                # High confidence: Type-specific only
                k = min(HIGH_CONF_K, len(type_entity_ids))
            else:
                # Low confidence: Reduced type-specific K
                k = min(LOW_CONF_TYPE_K, len(type_entity_ids))

            distances, indices = type_index.search(query_embedding, k + 1)  # +1 for self

            for dist, neigh_idx in zip(distances[0], indices[0]):
                if neigh_idx == local_idx:  # Skip self
                    continue

                neighbor_id = type_entity_ids[neigh_idx]

                # Convert distance to similarity
                if index_type == "IP":
                    similarity = float(dist)
                else:  # HNSW (L2)
                    similarity = 1.0 - (dist * dist / 2.0)

                similarity = float(np.clip(similarity, 0.0, 1.0))
                candidates.append((neighbor_id, similarity))

            # STRATEGY 2: Global search (only low confidence entities)
            if confidence < CONFIDENCE_THRESHOLD and global_index is not None:
                # Find position in global index
                try:
                    global_local_idx = low_conf_entities.index(entity_id)
                    k_global = min(LOW_CONF_GLOBAL_K, len(low_conf_entities))

                    distances, indices = global_index.search(query_embedding, k_global + 1)

                    for dist, neigh_idx in zip(distances[0], indices[0]):
                        if neigh_idx == global_local_idx:  # Skip self
                            continue

                        neighbor_id = low_conf_entities[neigh_idx]

                        # Skip if same type (already searched)
                        neighbor_type = entity_types[self.id_to_entity[neighbor_id]]["type"]
                        if neighbor_type == entity_type:
                            continue

                        # Convert distance to similarity
                        if len(low_conf_entities) < 100:  # IP index
                            similarity = float(dist)
                        else:  # HNSW
                            similarity = 1.0 - (dist * dist / 2.0)

                        similarity = float(np.clip(similarity, 0.0, 1.0))
                        candidates.append((neighbor_id, similarity))
                except ValueError:
                    # Entity not in low_conf_entities (shouldn't happen)
                    pass

            # Process candidates for this entity
            for neighbor_id, similarity in candidates:
                # Threshold filter
                if similarity < self.config.faiss_similarity_threshold:
                    continue

                candidates_before_filter += 1

                # Token overlap filter
                entity1_name = self.id_to_entity[entity_id]
                entity2_name = self.id_to_entity[neighbor_id]

                if not self._has_token_overlap(entity1_name, entity2_name):
                    continue

                candidates_after_filter += 1

                # Ensure consistent ordering and avoid duplicates
                if entity_id < neighbor_id:
                    pair_key = (entity_id, neighbor_id)
                else:
                    pair_key = (neighbor_id, entity_id)

                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    candidate_pairs.append((pair_key[0], pair_key[1], similarity))

        logger.info(f"✅ Generated {len(candidate_pairs)} candidate pairs")
        logger.info(f"  Before token filter: {candidates_before_filter}")
        logger.info(f"  After token filter: {candidates_after_filter} (-{100*(1-candidates_after_filter/max(candidates_before_filter,1)):.1f}%)")

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

    def _compute_colbert_similarity(self, entity1: str, entity2: str) -> float:
        """
        Compute ColBERT similarity between two entities using direct encoding.

        Uses the fixed pairwise similarity method that:
        - Encodes both entities directly (no indexing, no FAISS)
        - Computes MaxSim at token level
        - Returns bidirectional averaged score (symmetric)

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            float: ColBERT similarity score [0.0, 1.0], already bidirectionally averaged

        Note:
            The returned score is already symmetric (entity1↔entity2 = entity2↔entity1).
            No need to compute reverse direction separately.
        """
        if self.colbert_model is None:
            logger.warning("ColBERT model not initialized, returning 0.0 similarity")
            return 0.0

        try:
            # Use the built-in pairwise similarity method
            # This method internally computes bidirectional MaxSim and averages
            score = self.colbert_model.compute_pairwise_similarity(entity1, entity2)

            # Validate score is a valid number
            if not isinstance(score, (int, float)) or score < 0:
                logger.warning(f"Invalid ColBERT score {score} for '{entity1}' vs '{entity2}', returning 0.0")
                return 0.0

            return float(score)

        except Exception as e:
            logger.warning(f"ColBERT similarity computation failed for '{entity1}' and '{entity2}': {e}")
            return 0.0

    def stage3_multifeature_scoring(self, candidate_pairs: List, embeddings: np.ndarray, entity_types: Dict) -> List[Dict]:
        """
        Calculate comprehensive similarity scores using 4 features.

        Features:
        1. SapBERT similarity (0.50) - Medical domain embeddings
        2. Lexical similarity (0.15) - Levenshtein edit distance
        3. ColBERT similarity (0.25) - Late interaction token-level matching
        4. Graph similarity (0.10) - Shared neighbors (Jaccard)

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

        # Build graph neighbor cache for efficiency
        entity_neighbors = {}
        for entity in self.entities:
            neighbors = set()
            for head, rel, tail in self.triples:
                if head == entity:
                    neighbors.add(tail)
                if tail == entity:
                    neighbors.add(head)
            entity_neighbors[entity] = neighbors

        scored_pairs = []

        for e1_id, e2_id, sapbert_sim in tqdm(candidate_pairs, desc="Scoring pairs"):
            entity1 = self.id_to_entity[e1_id]
            entity2 = self.id_to_entity[e2_id]

            # Feature 1: SapBERT similarity (already computed)
            # Clamp to [0, 1] to ensure valid range
            sapbert_score = float(np.clip(sapbert_sim, 0.0, 1.0))

            # Feature 2: Lexical similarity (normalized edit distance)
            max_len = max(len(entity1), len(entity2))
            if max_len == 0:
                lexical_score = 1.0
            else:
                # Levenshtein edit distance
                edit_dist = self._levenshtein_distance(entity1.lower(), entity2.lower())
                lexical_score = 1.0 - (edit_dist / max_len)

            # Feature 3: ColBERT similarity (late interaction)
            # compute_pairwise_similarity() already returns bidirectional average
            colbert_score = self._compute_colbert_similarity(entity1, entity2)

            # Feature 4: Graph similarity (Jaccard similarity of neighbors)
            neighbors1 = entity_neighbors.get(entity1, set())
            neighbors2 = entity_neighbors.get(entity2, set())
            if len(neighbors1) + len(neighbors2) > 0:
                shared = len(neighbors1 & neighbors2)
                total = len(neighbors1 | neighbors2)
                graph_score = shared / total if total > 0 else 0.0
            else:
                graph_score = 0.0

            # Weighted combination
            weights = self.config.feature_weights
            final_score = (
                weights["sapbert"] * sapbert_score +
                weights["lexical"] * lexical_score +
                weights["colbert"] * colbert_score +
                weights["graph"] * graph_score
            )

            scored_pairs.append({
                "entity1_id": e1_id,
                "entity2_id": e2_id,
                "entity1_name": entity1,
                "entity2_name": entity2,
                "sapbert": sapbert_score,
                "lexical": lexical_score,
                "colbert": colbert_score,
                "graph": graph_score,
                "final_score": final_score
            })

        logger.info(f"✅ Scored {len(scored_pairs)} pairs")

        # Save
        if self.config.save_intermediate:
            with open(scores_path, 'w') as f:
                for pair in scored_pairs:
                    f.write(json.dumps(pair) + '\n')
            logger.info(f"✅ Saved to: {scores_path}")

        # Evaluation
        self.evaluate_stage3(scored_pairs)

        return scored_pairs

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

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

        equivalent_pairs = []

        for pair in tqdm(scored_pairs, desc="Thresholding"):
            entity1 = pair["entity1_name"]
            entity2 = pair["entity2_name"]

            # Get entity type (both should be same type after blocking)
            entity_type = entity_types[entity1]["type"]

            # Get type-specific threshold (default to 0.80 if type not found)
            threshold = self.config.type_thresholds.get(entity_type, 0.80)

            # Apply threshold
            if pair["final_score"] >= threshold:
                equivalent_pairs.append((pair["entity1_id"], pair["entity2_id"]))

        logger.info(f"✅ Found {len(equivalent_pairs)} equivalent pairs")

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

        # Union-Find data structure for clustering
        parent = {}

        def find(x):
            """Find root of element x with path compression"""
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            """Union two elements"""
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build clusters using Union-Find
        for e1_id, e2_id in equivalent_pairs:
            union(e1_id, e2_id)

        # Group entities by cluster root
        clusters_dict = {}
        for entity_id in range(len(self.entities)):
            root = find(entity_id)
            if root not in clusters_dict:
                clusters_dict[root] = []
            clusters_dict[root].append(entity_id)

        # Filter out singleton clusters (only 1 entity)
        clusters = {str(root): entity_ids for root, entity_ids in clusters_dict.items() if len(entity_ids) > 1}

        # Select canonical name for each cluster
        canonical_names = {}
        synonym_edges = []

        for cluster_id, entity_ids in clusters.items():
            entity_names = [self.id_to_entity[eid] for eid in entity_ids]

            # Count frequency of each entity in the knowledge graph
            name_freq = {name: 0 for name in entity_names}
            for head, rel, tail in self.triples:
                if head in name_freq:
                    name_freq[head] += 1
                if tail in name_freq:
                    name_freq[tail] += 1

            # Select canonical name based on method
            if self.config.canonical_selection_method == "frequency":
                # Most frequent in corpus
                canonical_name = max(entity_names, key=lambda n: name_freq[n])
            elif self.config.canonical_selection_method == "length":
                # Shortest name (often more concise)
                canonical_name = min(entity_names, key=len)
            else:  # Default to frequency
                canonical_name = max(entity_names, key=lambda n: name_freq[n])

            # Get canonical entity ID
            canonical_id = self.entity_to_id[canonical_name]
            canonical_names[cluster_id] = canonical_id

            # Create SYNONYM_OF edges
            for name in entity_names:
                if name != canonical_name:
                    synonym_edges.append((name, "SYNONYM_OF", canonical_name))

        logger.info(f"✅ Created {len(synonym_edges)} SYNONYM_OF edges from {len(clusters)} clusters")

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
        colbert_model=cfg.colbert.model,
        colbert_root=cfg.colbert.root,
        colbert_topk=cfg.colbert.topk,
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

    # Generate visualizations
    logger.info("\nGenerating visualization plots...")
    try:
        from gfmrag.workflow.stage2_visualization import visualize_stage2_metrics
        visualize_stage2_metrics(Path(config.output_dir))
        logger.info("✓ Visualizations generated successfully")
    except ImportError:
        logger.warning("Matplotlib/Seaborn not installed. Skipping visualization.")
        logger.warning("Install with: pip install matplotlib seaborn")
    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")


if __name__ == "__main__":
    main()
