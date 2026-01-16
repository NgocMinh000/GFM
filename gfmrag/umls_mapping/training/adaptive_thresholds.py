"""
Adaptive Thresholds for Entity Type-Specific Confidence

Different entity types may require different confidence thresholds:
- "Disease" entities might be easier to link (lower threshold)
- "Pharmacologic Substance" entities might be harder (higher threshold)

This module tunes thresholds per entity type to maximize F1 or other metrics.

Strategy:
1. Collect predictions per entity type on validation set
2. For each type, find optimal threshold that maximizes F1
3. Use type-specific thresholds at inference time
"""

import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class AdaptiveThresholdTuner:
    """
    Tunes confidence thresholds per entity type.

    Each entity type gets its own threshold optimized for:
    - F1 score (default)
    - Precision (if high precision required)
    - Recall (if high coverage required)
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        min_samples: int = 100,
        objective: str = "f1",
    ):
        """
        Initialize threshold tuner.

        Args:
            default_threshold: Fallback threshold for types with insufficient data
            min_samples: Minimum samples required to tune type-specific threshold
            objective: Optimization objective ("f1", "precision", or "recall")
        """
        self.default_threshold = default_threshold
        self.min_samples = min_samples
        self.objective = objective

        # Type-specific thresholds
        self.thresholds: Dict[str, float] = {}

        # Statistics
        self.type_stats: Dict[str, Dict] = {}

        self.is_fitted = False

    def fit(
        self,
        entity_types: List[str],
        probabilities: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Tune thresholds on validation data.

        Args:
            entity_types: Entity type for each sample [N]
            probabilities: Predicted probabilities [N]
            labels: True labels [N]
        """
        logger.info(f"Tuning adaptive thresholds (objective: {self.objective})...")

        # Group samples by entity type
        type_to_samples = defaultdict(list)

        for i, entity_type in enumerate(entity_types):
            type_to_samples[entity_type].append({
                "prob": probabilities[i],
                "label": labels[i],
            })

        # Tune threshold for each type
        for entity_type, samples in type_to_samples.items():
            if len(samples) < self.min_samples:
                # Not enough samples, use default threshold
                self.thresholds[entity_type] = self.default_threshold
                logger.debug(f"Type '{entity_type}': Using default threshold "
                            f"({len(samples)} < {self.min_samples} samples)")
                continue

            # Extract data
            type_probs = np.array([s["prob"] for s in samples])
            type_labels = np.array([s["label"] for s in samples])

            # Find optimal threshold
            optimal_threshold, metrics = self._find_optimal_threshold(
                type_probs, type_labels
            )

            self.thresholds[entity_type] = optimal_threshold
            self.type_stats[entity_type] = {
                "num_samples": len(samples),
                "optimal_threshold": optimal_threshold,
                **metrics,
            }

            logger.info(
                f"Type '{entity_type}': threshold={optimal_threshold:.3f}, "
                f"F1={metrics['f1']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}, "
                f"samples={len(samples)}"
            )

        self.is_fitted = True

        # Summary statistics
        self._print_summary()

    def _find_optimal_threshold(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        num_thresholds: int = 100,
    ) -> Tuple[float, Dict]:
        """
        Find optimal threshold for a single entity type.

        Args:
            probabilities: Predicted probabilities [N]
            labels: True labels [N]
            num_thresholds: Number of thresholds to try

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        # Try different thresholds
        thresholds = np.linspace(0.0, 1.0, num_thresholds)

        best_threshold = self.default_threshold
        best_score = 0.0
        best_metrics = {}

        for threshold in thresholds:
            # Make predictions
            preds = (probabilities >= threshold).astype(int)

            # Compute metrics
            f1 = f1_score(labels, preds, zero_division=0)
            precision = precision_score(labels, preds, zero_division=0)
            recall = recall_score(labels, preds, zero_division=0)

            # Select score based on objective
            if self.objective == "f1":
                score = f1
            elif self.objective == "precision":
                score = precision
            elif self.objective == "recall":
                score = recall
            else:
                score = f1

            # Update best
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                }

        return best_threshold, best_metrics

    def predict(
        self,
        entity_type: str,
        probability: float,
    ) -> bool:
        """
        Make prediction using type-specific threshold.

        Args:
            entity_type: Entity type
            probability: Predicted probability

        Returns:
            Boolean prediction (True = positive, False = negative)
        """
        if not self.is_fitted:
            raise RuntimeError("Thresholds not fitted. Call fit() first.")

        # Get threshold for this type (fallback to default if not found)
        threshold = self.thresholds.get(entity_type, self.default_threshold)

        return probability >= threshold

    def get_threshold(self, entity_type: str) -> float:
        """
        Get threshold for a specific entity type.

        Args:
            entity_type: Entity type

        Returns:
            Threshold value
        """
        return self.thresholds.get(entity_type, self.default_threshold)

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 100)
        logger.info("ADAPTIVE THRESHOLD SUMMARY")
        logger.info("=" * 100)

        # Overall statistics
        num_types = len(self.thresholds)
        num_types_tuned = len([t for t, s in self.type_stats.items() if s["num_samples"] >= self.min_samples])
        num_types_default = num_types - num_types_tuned

        logger.info(f"Total entity types: {num_types}")
        logger.info(f"  Tuned thresholds: {num_types_tuned}")
        logger.info(f"  Default thresholds: {num_types_default}")
        logger.info("")

        # Threshold distribution
        if self.type_stats:
            thresholds_tuned = [s["optimal_threshold"] for s in self.type_stats.values()]
            logger.info(f"Threshold distribution (tuned types):")
            logger.info(f"  Min:    {min(thresholds_tuned):.3f}")
            logger.info(f"  Q1:     {np.percentile(thresholds_tuned, 25):.3f}")
            logger.info(f"  Median: {np.percentile(thresholds_tuned, 50):.3f}")
            logger.info(f"  Q3:     {np.percentile(thresholds_tuned, 75):.3f}")
            logger.info(f"  Max:    {max(thresholds_tuned):.3f}")
            logger.info("")

        # Top/bottom types by threshold
        if self.type_stats:
            logger.info("Top 5 types with LOWEST thresholds (easiest to link):")
            sorted_types = sorted(
                self.type_stats.items(),
                key=lambda x: x[1]["optimal_threshold"]
            )
            for entity_type, stats in sorted_types[:5]:
                logger.info(f"  {entity_type:<40} {stats['optimal_threshold']:.3f} (F1={stats['f1']:.4f})")

            logger.info("")
            logger.info("Top 5 types with HIGHEST thresholds (hardest to link):")
            for entity_type, stats in sorted_types[-5:]:
                logger.info(f"  {entity_type:<40} {stats['optimal_threshold']:.3f} (F1={stats['f1']:.4f})")

        logger.info("=" * 100)
        logger.info("")

    def save(self, save_path: str):
        """Save thresholds to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'thresholds': self.thresholds,
                'type_stats': self.type_stats,
                'default_threshold': self.default_threshold,
                'min_samples': self.min_samples,
                'objective': self.objective,
                'is_fitted': self.is_fitted,
            }, f)

        logger.info(f"Saved adaptive thresholds to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "AdaptiveThresholdTuner":
        """Load thresholds from disk."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        tuner = cls(
            default_threshold=data['default_threshold'],
            min_samples=data['min_samples'],
            objective=data['objective'],
        )
        tuner.thresholds = data['thresholds']
        tuner.type_stats = data['type_stats']
        tuner.is_fitted = data['is_fitted']

        logger.info(f"Loaded adaptive thresholds from {load_path}")
        logger.info(f"  Loaded {len(tuner.thresholds)} type-specific thresholds")

        return tuner

    def get_statistics(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with threshold statistics
        """
        if not self.type_stats:
            return {}

        thresholds_tuned = [s["optimal_threshold"] for s in self.type_stats.values()]
        f1_scores = [s["f1"] for s in self.type_stats.values()]

        return {
            "num_types": len(self.thresholds),
            "num_tuned": len(self.type_stats),
            "threshold_min": min(thresholds_tuned) if thresholds_tuned else 0.0,
            "threshold_max": max(thresholds_tuned) if thresholds_tuned else 0.0,
            "threshold_mean": np.mean(thresholds_tuned) if thresholds_tuned else 0.0,
            "threshold_median": np.median(thresholds_tuned) if thresholds_tuned else 0.0,
            "f1_mean": np.mean(f1_scores) if f1_scores else 0.0,
            "f1_median": np.median(f1_scores) if f1_scores else 0.0,
        }


class HierarchicalThresholdTuner:
    """
    Hierarchical threshold tuning.

    Uses UMLS semantic type hierarchy to share information:
    - If a specific type has insufficient data, use parent type threshold
    - Fallback chain: specific_type -> parent_type -> default
    """

    def __init__(
        self,
        default_threshold: float = 0.5,
        min_samples: int = 100,
        objective: str = "f1",
        type_hierarchy: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize hierarchical threshold tuner.

        Args:
            default_threshold: Default threshold
            min_samples: Minimum samples for tuning
            objective: Optimization objective
            type_hierarchy: Mapping from specific type to parent type
        """
        self.base_tuner = AdaptiveThresholdTuner(
            default_threshold=default_threshold,
            min_samples=min_samples,
            objective=objective,
        )
        self.type_hierarchy = type_hierarchy or {}

    def fit(
        self,
        entity_types: List[str],
        probabilities: np.ndarray,
        labels: np.ndarray,
    ):
        """Fit thresholds with hierarchical fallback."""
        self.base_tuner.fit(entity_types, probabilities, labels)

    def get_threshold(self, entity_type: str) -> float:
        """
        Get threshold with hierarchical fallback.

        Args:
            entity_type: Entity type

        Returns:
            Threshold (specific type -> parent type -> default)
        """
        # Try specific type
        if entity_type in self.base_tuner.thresholds:
            return self.base_tuner.thresholds[entity_type]

        # Try parent type
        parent_type = self.type_hierarchy.get(entity_type)
        if parent_type and parent_type in self.base_tuner.thresholds:
            logger.debug(f"Using parent threshold for '{entity_type}' -> '{parent_type}'")
            return self.base_tuner.thresholds[parent_type]

        # Fallback to default
        return self.base_tuner.default_threshold

    def predict(self, entity_type: str, probability: float) -> bool:
        """Make prediction with hierarchical threshold."""
        threshold = self.get_threshold(entity_type)
        return probability >= threshold

    def save(self, save_path: str):
        """Save hierarchical tuner."""
        self.base_tuner.save(save_path)

        # Also save hierarchy
        hierarchy_path = str(Path(save_path).with_suffix('.hierarchy.pkl'))
        with open(hierarchy_path, 'wb') as f:
            pickle.dump(self.type_hierarchy, f)

        logger.info(f"Saved hierarchical tuner to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "HierarchicalThresholdTuner":
        """Load hierarchical tuner."""
        base_tuner = AdaptiveThresholdTuner.load(load_path)

        # Load hierarchy
        hierarchy_path = str(Path(load_path).with_suffix('.hierarchy.pkl'))
        try:
            with open(hierarchy_path, 'rb') as f:
                type_hierarchy = pickle.load(f)
        except FileNotFoundError:
            type_hierarchy = {}

        tuner = cls(
            default_threshold=base_tuner.default_threshold,
            min_samples=base_tuner.min_samples,
            objective=base_tuner.objective,
            type_hierarchy=type_hierarchy,
        )
        tuner.base_tuner = base_tuner

        logger.info(f"Loaded hierarchical tuner from {load_path}")

        return tuner


def tune_thresholds_from_model(
    model,
    val_loader,
    umls_loader,
    device,
    objective: str = "f1",
    save_path: Optional[str] = None,
) -> AdaptiveThresholdTuner:
    """
    Tune adaptive thresholds from a trained model.

    Args:
        model: Trained cross-encoder
        val_loader: Validation data loader
        umls_loader: UMLS loader (for semantic types)
        device: Computation device
        objective: Optimization objective
        save_path: Path to save tuner (optional)

    Returns:
        Fitted AdaptiveThresholdTuner
    """
    logger.info("Collecting validation predictions for threshold tuning...")

    # Collect predictions
    model.eval()
    all_probs = []
    all_labels = []
    all_types = []

    with torch.no_grad():
        for batch in val_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )

            probs = outputs["probabilities"][:, 1].cpu().numpy()
            labels = batch_device["labels"].cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

            # Extract entity types from batch metadata (if available)
            if "entity_type" in batch:
                all_types.extend(batch["entity_type"])

    # Convert to arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Tune thresholds
    tuner = AdaptiveThresholdTuner(
        default_threshold=0.5,
        min_samples=100,
        objective=objective,
    )

    if all_types:
        tuner.fit(all_types, all_probs, all_labels)
    else:
        logger.warning("No entity type information available, using default threshold for all")

    # Save tuner
    if save_path:
        tuner.save(save_path)

    return tuner
