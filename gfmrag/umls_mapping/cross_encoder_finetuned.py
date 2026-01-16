"""
Enhanced Cross-Encoder Reranker with Fine-Tuning Integration

This module provides an enhanced version of CrossEncoderReranker that:
1. Uses fine-tuned binary cross-encoder (from Phase 2 training)
2. Applies Platt scaling for calibrated probabilities
3. Uses adaptive thresholds per entity type for better precision/recall tradeoff

Usage:
    # Use in Stage 3 pipeline
    from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker

    reranker = FineTunedCrossEncoderReranker(
        model_path="models/cross_encoder_finetuned/checkpoint-best",
        calibrator_path="models/calibration/platt_scaler.pkl",
        threshold_tuner_path="models/thresholds/adaptive_tuner.pkl",
        device="cuda"
    )

    reranked = reranker.rerank(entity, candidates, entity_type="Disease")
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from gfmrag.umls_mapping.config import UMLSMappingConfig
from gfmrag.umls_mapping.cross_encoder_reranker import RerankedCandidate
from gfmrag.umls_mapping.hard_negative_filter import FilteredCandidate
from gfmrag.umls_mapping.training.models.binary_cross_encoder import BinaryCrossEncoder
from gfmrag.umls_mapping.training.calibration import PlattScaling
from gfmrag.umls_mapping.training.adaptive_thresholds import AdaptiveThresholdTuner

logger = logging.getLogger(__name__)


class FineTunedCrossEncoderReranker:
    """
    Enhanced cross-encoder reranker using fine-tuned model.

    Improvements over baseline:
    - Fine-tuned on MedMentions entity linking task (~2.4M samples)
    - Calibrated probabilities via Platt scaling (better confidence estimates)
    - Adaptive thresholds per entity type (optimized for each semantic type)

    Expected improvements:
    - High confidence mappings: 5-10% → 40-60% (+30-50%)
    - Cross-encoder score: 0.58 → 0.85+ (+47%)
    - Score margin: 0.10-0.12 → 0.25+ (+100-150%)
    """

    def __init__(
        self,
        model_path: str,
        calibrator_path: Optional[str] = None,
        threshold_tuner_path: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 32,
        use_calibration: bool = True,
        use_adaptive_thresholds: bool = True,
    ):
        """
        Initialize fine-tuned cross-encoder reranker.

        Args:
            model_path: Path to fine-tuned model checkpoint
            calibrator_path: Path to Platt scaling calibrator (optional)
            threshold_tuner_path: Path to adaptive threshold tuner (optional)
            device: Computation device ("cuda" or "cpu")
            batch_size: Batch size for scoring
            use_calibration: Whether to apply calibration
            use_adaptive_thresholds: Whether to use adaptive thresholds
        """
        self.model_path = Path(model_path)
        self.calibrator_path = Path(calibrator_path) if calibrator_path else None
        self.threshold_tuner_path = Path(threshold_tuner_path) if threshold_tuner_path else None
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_calibration = use_calibration
        self.use_adaptive_thresholds = use_adaptive_thresholds

        # Components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.calibrator = None
        self.threshold_tuner = None

        logger.info(f"Initialized FineTunedCrossEncoderReranker")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Calibration: {use_calibration}")
        logger.info(f"  Adaptive thresholds: {use_adaptive_thresholds}")

    def _load_components(self):
        """Lazy load model and components."""
        if self.model is not None:
            return  # Already loaded

        logger.info("Loading fine-tuned cross-encoder components...")

        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        )

        # Load fine-tuned model
        logger.info(f"Loading fine-tuned model from {self.model_path}")
        self.model = BinaryCrossEncoder(
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            num_labels=2,
            dropout=0.1,
        )

        state_dict = torch.load(
            self.model_path / "pytorch_model.bin",
            map_location=self.device,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Model loaded with {self.model.count_parameters():,} parameters")

        # Load calibrator
        if self.use_calibration and self.calibrator_path and self.calibrator_path.exists():
            logger.info(f"Loading calibrator from {self.calibrator_path}")
            self.calibrator = PlattScaling.load(str(self.calibrator_path))
        elif self.use_calibration:
            logger.warning(f"Calibration enabled but calibrator not found at {self.calibrator_path}")
            logger.warning("Using uncalibrated probabilities")

        # Load threshold tuner
        if self.use_adaptive_thresholds and self.threshold_tuner_path and self.threshold_tuner_path.exists():
            logger.info(f"Loading threshold tuner from {self.threshold_tuner_path}")
            self.threshold_tuner = AdaptiveThresholdTuner.load(str(self.threshold_tuner_path))
        elif self.use_adaptive_thresholds:
            logger.warning(f"Adaptive thresholds enabled but tuner not found at {self.threshold_tuner_path}")
            logger.warning("Using default threshold 0.5")

    def rerank(
        self,
        entity: str,
        candidates: List[FilteredCandidate],
        entity_type: Optional[str] = None,
        return_all: bool = False,
    ) -> List[RerankedCandidate]:
        """
        Rerank candidates using fine-tuned cross-encoder.

        Args:
            entity: Original entity text
            candidates: List of filtered candidates
            entity_type: Entity semantic type (for adaptive thresholds)
            return_all: If False, only return candidates above threshold

        Returns:
            Reranked candidates with calibrated scores
        """
        # Lazy load components
        self._load_components()

        if not candidates:
            return []

        # Score all (entity, candidate) pairs
        cross_scores, calibrated_scores = self._score_pairs(entity, candidates)

        # Create reranked candidates
        reranked = []
        for candidate, cross_score, calibrated_score in zip(candidates, cross_scores, calibrated_scores):
            # Use calibrated score if available, otherwise raw cross-encoder score
            final_cross_score = calibrated_score if self.calibrator else cross_score

            # Weighted combination with previous score
            # Give more weight to fine-tuned cross-encoder (0.8 vs 0.2)
            final_score = (
                final_cross_score * 0.8 +
                candidate.score * 0.2
            )

            reranked.append(RerankedCandidate(
                cui=candidate.cui,
                name=candidate.name,
                score=final_score,
                cross_encoder_score=final_cross_score,
                previous_score=candidate.score,
                method='finetuned_cross_encoder',
            ))

        # Sort by final score
        reranked.sort(key=lambda x: x.score, reverse=True)

        # Apply adaptive threshold filtering
        if not return_all and self.use_adaptive_thresholds and self.threshold_tuner:
            threshold = self.threshold_tuner.get_threshold(entity_type or "default")
            reranked = [c for c in reranked if c.cross_encoder_score >= threshold]

            logger.debug(f"Applied threshold {threshold:.3f} for type '{entity_type}', "
                        f"kept {len(reranked)}/{len(candidates)} candidates")

        return reranked

    def _score_pairs(
        self,
        entity: str,
        candidates: List[FilteredCandidate],
    ) -> Tuple[List[float], List[float]]:
        """
        Score all (entity, candidate) pairs.

        Args:
            entity: Query entity
            candidates: List of candidates

        Returns:
            Tuple of (raw_scores, calibrated_scores)
        """
        # Prepare pairs
        pairs = [(entity, candidate.name) for candidate in candidates]

        all_raw_scores = []
        all_logits = []

        # Batch processing
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i+self.batch_size]

            # Tokenize pairs
            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]

            inputs = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)

            # Score with fine-tuned model
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )

                # Get probabilities for positive class
                probs = outputs["probabilities"][:, 1]  # P(correct match)
                logits = outputs["logits"]

                all_raw_scores.extend(probs.cpu().numpy().tolist())
                all_logits.append(logits.cpu().numpy())

        # Apply calibration if available
        if self.calibrator:
            all_logits = np.concatenate(all_logits, axis=0)
            calibrated_scores = self.calibrator.predict_proba(all_logits).tolist()
        else:
            calibrated_scores = all_raw_scores

        return all_raw_scores, calibrated_scores

    def predict_single(
        self,
        entity: str,
        cui_name: str,
        entity_type: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Predict score for a single (entity, CUI) pair.

        Args:
            entity: Entity text
            cui_name: CUI preferred name
            entity_type: Entity semantic type (for threshold)

        Returns:
            Tuple of (calibrated_score, is_correct_prediction)
        """
        self._load_components()

        # Tokenize
        inputs = self.tokenizer(
            entity,
            cui_name,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Score
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            prob = outputs["probabilities"][0, 1].item()  # P(correct match)
            logits = outputs["logits"][0].cpu().numpy()

        # Apply calibration
        if self.calibrator:
            calibrated_prob = self.calibrator.predict_proba(logits.reshape(1, -1))[0]
        else:
            calibrated_prob = prob

        # Apply threshold
        if self.use_adaptive_thresholds and self.threshold_tuner:
            threshold = self.threshold_tuner.get_threshold(entity_type or "default")
            is_correct = calibrated_prob >= threshold
        else:
            is_correct = calibrated_prob >= 0.5

        return calibrated_prob, is_correct

    @classmethod
    def from_config(
        cls,
        config: UMLSMappingConfig,
        use_finetuned: bool = True,
    ) -> "FineTunedCrossEncoderReranker":
        """
        Create reranker from config.

        Args:
            config: UMLS mapping config
            use_finetuned: Whether to use fine-tuned model (fallback to baseline if False)

        Returns:
            FineTunedCrossEncoderReranker instance
        """
        if not use_finetuned:
            # Fallback to baseline cross-encoder
            logger.warning("Fine-tuned model not requested, using baseline CrossEncoderReranker")
            from gfmrag.umls_mapping.cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker(config)

        # Paths from config
        model_path = config.finetuned_model_path if hasattr(config, 'finetuned_model_path') else \
                     "models/cross_encoder_finetuned/checkpoint-best"

        calibrator_path = config.calibrator_path if hasattr(config, 'calibrator_path') else \
                         "models/calibration/platt_scaler.pkl"

        threshold_tuner_path = config.threshold_tuner_path if hasattr(config, 'threshold_tuner_path') else \
                              "models/thresholds/adaptive_tuner.pkl"

        device = config.cross_encoder_device if hasattr(config, 'cross_encoder_device') else "cuda"

        return cls(
            model_path=model_path,
            calibrator_path=calibrator_path,
            threshold_tuner_path=threshold_tuner_path,
            device=device,
        )

    def get_confidence_tier(
        self,
        score: float,
        entity_type: Optional[str] = None,
    ) -> str:
        """
        Get confidence tier for a score.

        Args:
            score: Calibrated cross-encoder score
            entity_type: Entity type (for adaptive thresholding)

        Returns:
            Confidence tier: "high", "medium", or "low"
        """
        # Get threshold
        if self.use_adaptive_thresholds and self.threshold_tuner:
            threshold = self.threshold_tuner.get_threshold(entity_type or "default")
        else:
            threshold = 0.5

        # Tier based on distance from threshold
        if score >= threshold + 0.2:  # Far above threshold
            return "high"
        elif score >= threshold:  # Above threshold
            return "medium"
        else:  # Below threshold
            return "low"

    def compute_score_margin(
        self,
        reranked_candidates: List[RerankedCandidate],
    ) -> float:
        """
        Compute score margin between top 2 candidates.

        Large margin = high confidence in top candidate.

        Args:
            reranked_candidates: List of reranked candidates (sorted by score)

        Returns:
            Score margin (0.0 if <2 candidates)
        """
        if len(reranked_candidates) < 2:
            return 0.0

        return reranked_candidates[0].cross_encoder_score - reranked_candidates[1].cross_encoder_score

    def get_statistics(self) -> dict:
        """Get reranker statistics."""
        stats = {
            "model_loaded": self.model is not None,
            "calibration_enabled": self.use_calibration,
            "calibrator_loaded": self.calibrator is not None,
            "adaptive_thresholds_enabled": self.use_adaptive_thresholds,
            "threshold_tuner_loaded": self.threshold_tuner is not None,
            "device": str(self.device),
        }

        if self.threshold_tuner:
            stats["threshold_statistics"] = self.threshold_tuner.get_statistics()

        return stats
