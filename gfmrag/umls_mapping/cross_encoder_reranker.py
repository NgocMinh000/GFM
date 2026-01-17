"""
Stage 3.5: Cross-Encoder Reranking
Reranks candidates using PubMedBERT cross-encoder
"""

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import UMLSMappingConfig
from .hard_negative_filter import FilteredCandidate

logger = logging.getLogger(__name__)


@dataclass
class RerankedCandidate:
    """Candidate with cross-encoder score"""
    cui: str
    name: str
    score: float
    cross_encoder_score: float
    previous_score: float
    method: str = 'cross_encoder_reranked'


class CrossEncoderReranker:
    """
    Reranks candidates using cross-encoder (PubMedBERT)

    Cross-encoder directly scores (entity, candidate) pairs
    More accurate than bi-encoder but slower
    """

    def __init__(self, config: UMLSMappingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None

    def rerank(
        self,
        entity: str,
        candidates: List[FilteredCandidate]
    ) -> List[RerankedCandidate]:
        """
        Rerank candidates using cross-encoder

        Args:
            entity: Original entity text
            candidates: List of filtered candidates

        Returns:
            Reranked candidates
        """

        # Lazy load model
        if self.model is None:
            self._load_model()

        # Score all (entity, candidate) pairs
        cross_scores = self._score_pairs(entity, candidates)

        # Combine with previous scores + apply pre-filtering
        reranked = []
        filtered_count = 0

        # Get thresholds from config
        min_prev_score = getattr(self.config, 'cross_encoder_min_prev_score', 0.6)
        min_cross_score = getattr(self.config, 'cross_encoder_min_cross_score', 0.5)
        cross_weight = getattr(self.config, 'cross_encoder_weight', 0.4)
        prev_weight = 1.0 - cross_weight

        for candidate, cross_score in zip(candidates, cross_scores):
            # PRE-FILTERING: Remove low-quality candidates
            # Skip if previous_score < threshold OR cross_encoder_score < threshold
            if candidate.score < min_prev_score:
                filtered_count += 1
                logger.debug(f"Filtered (low previous_score): {candidate.name} (prev={candidate.score:.3f} < {min_prev_score})")
                continue

            if cross_score < min_cross_score:
                filtered_count += 1
                logger.debug(f"Filtered (low cross_encoder): {candidate.name} (cross={cross_score:.3f} < {min_cross_score})")
                continue

            # Weighted combination (rebalanced: configurable weights)
            # Default: 0.4 cross-encoder, 0.6 previous
            # Rationale: previous_score is aggregation of 4 stages (more reliable)
            #            cross_encoder is zero-shot (less reliable until fine-tuned)
            final_score = (
                cross_score * cross_weight +
                candidate.score * prev_weight
            )

            reranked.append(RerankedCandidate(
                cui=candidate.cui,
                name=candidate.name,
                score=final_score,
                cross_encoder_score=cross_score,
                previous_score=candidate.score
            ))

        # Log filtering stats
        if filtered_count > 0:
            logger.info(f"Pre-filtered {filtered_count}/{len(candidates)} candidates for '{entity}'")

        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)

        return reranked

    def _load_model(self):
        """Load cross-encoder model"""
        logger.info(f"Loading cross-encoder model: {self.config.cross_encoder_model}")

        device = torch.device(self.config.cross_encoder_device if torch.cuda.is_available() else 'cpu')

        # Check if fine-tuned model exists
        model_path = self.config.cross_encoder_model

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # For PubMedBERT, we need to add a classification head
        # If pretrained reranker exists, load it; otherwise use base model
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1  # Binary relevance scoring
            ).to(device)
        except:
            # Fallback: use base model (will need fine-tuning)
            logger.warning("Could not load fine-tuned cross-encoder. Using base model.")
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(model_path).to(device)

        self.model.eval()

    def _score_pairs(
        self,
        entity: str,
        candidates: List[FilteredCandidate]
    ) -> List[float]:
        """
        Score all (entity, candidate) pairs

        Args:
            entity: Query entity
            candidates: List of candidates

        Returns:
            List of cross-encoder scores
        """

        device = self.model.device
        batch_size = 32

        # Prepare pairs
        pairs = [(entity, candidate.name) for candidate in candidates]

        all_scores = []

        for i in tqdm(range(0, len(pairs), batch_size), desc="Cross-encoder scoring", leave=False):
            batch_pairs = pairs[i:i+batch_size]

            # Tokenize pairs
            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]

            inputs = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            ).to(device)

            # Score
            with torch.no_grad():
                try:
                    # If using classification head
                    outputs = self.model(**inputs)
                    scores = torch.sigmoid(outputs.logits).squeeze(-1)
                except:
                    # Fallback: use CLS token similarity
                    outputs = self.model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :]
                    # Simple scoring: use mean of CLS embedding
                    scores = torch.sigmoid(cls_emb.mean(dim=1))

            all_scores.extend(scores.cpu().numpy().tolist())

        return all_scores

    def fine_tune(self, training_data: List[Tuple[str, str, int]]):
        """
        Fine-tune cross-encoder on labeled data

        Args:
            training_data: List of (entity, candidate, label) tuples
                          label: 1 for correct, 0 for incorrect

        Note: This is a placeholder. Full implementation would include:
        - Training loop with optimizer
        - Hard negative mining
        - Validation
        - Model checkpointing
        """

        logger.warning("Cross-encoder fine-tuning not implemented yet.")
        logger.info("Using pre-trained PubMedBERT without task-specific fine-tuning.")

        # TODO: Implement training loop if needed
        # For now, we rely on pre-trained PubMedBERT's medical knowledge
        pass
