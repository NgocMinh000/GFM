"""
Evaluation Metrics for Cross-Encoder Training

Provides comprehensive metrics for model evaluation:
- Classification metrics: Accuracy, Precision, Recall, F1
- Ranking metrics: ROC-AUC, PR-AUC
- Calibration metrics: ECE, Brier Score

Reference:
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Naeini et al. (2015) "Obtaining Well Calibrated Probabilities"
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification and ranking metrics.

    Args:
        labels: True binary labels [N]
        predictions: Predicted binary labels [N]
        probabilities: Predicted probabilities for positive class [N]

    Returns:
        Dictionary with metrics
    """
    metrics = {}

    # Classification metrics
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["precision"] = precision_score(labels, predictions, zero_division=0)
    metrics["recall"] = recall_score(labels, predictions, zero_division=0)
    metrics["f1"] = f1_score(labels, predictions, zero_division=0)

    # Ranking metrics
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        # Handle case where only one class is present
        metrics["roc_auc"] = 0.0

    try:
        metrics["pr_auc"] = average_precision_score(labels, probabilities)
    except ValueError:
        metrics["pr_auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Specificity
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def compute_calibration_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute calibration metrics.

    Calibration measures how well predicted probabilities match actual frequencies.
    A well-calibrated model should predict 0.8 probability for events that occur
    80% of the time.

    Args:
        labels: True binary labels [N]
        probabilities: Predicted probabilities for positive class [N]
        num_bins: Number of bins for ECE calculation

    Returns:
        Dictionary with calibration metrics:
        - ece: Expected Calibration Error
        - mce: Maximum Calibration Error
        - brier_score: Brier Score (MSE for probabilities)
        - reliability_diagram_data: Data for plotting reliability diagram
    """
    # Brier Score (lower is better, 0 is perfect)
    brier = brier_score_loss(labels, probabilities)

    # Expected Calibration Error (ECE)
    ece, mce, bin_data = compute_ece(labels, probabilities, num_bins)

    return {
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "reliability_data": bin_data,
    }


def compute_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float, List[Dict]]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy
    across different confidence bins.

    Args:
        labels: True binary labels [N]
        probabilities: Predicted probabilities [N]
        num_bins: Number of bins

    Returns:
        Tuple of (ece, mce, bin_data):
        - ece: Expected Calibration Error (weighted average)
        - mce: Maximum Calibration Error (worst bin)
        - bin_data: List of dicts with bin statistics
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Assign samples to bins
    bin_indices = np.digitize(probabilities, bin_boundaries[1:-1], right=True)

    ece = 0.0
    mce = 0.0
    bin_data = []

    total_samples = len(labels)

    for bin_idx in range(num_bins):
        # Get samples in this bin
        in_bin = bin_indices == bin_idx
        bin_count = in_bin.sum()

        if bin_count == 0:
            bin_data.append({
                "bin_lower": bin_lowers[bin_idx],
                "bin_upper": bin_uppers[bin_idx],
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_error": 0.0,
            })
            continue

        # Accuracy in bin (fraction of correct predictions)
        bin_labels = labels[in_bin]
        bin_probs = probabilities[in_bin]
        bin_preds = (bin_probs > 0.5).astype(int)
        bin_accuracy = (bin_labels == bin_preds).mean()

        # Average confidence in bin
        bin_confidence = bin_probs.mean()

        # Calibration error for this bin
        bin_error = abs(bin_accuracy - bin_confidence)

        # Update ECE (weighted by bin size)
        ece += (bin_count / total_samples) * bin_error

        # Update MCE (maximum error)
        mce = max(mce, bin_error)

        bin_data.append({
            "bin_lower": float(bin_lowers[bin_idx]),
            "bin_upper": float(bin_uppers[bin_idx]),
            "count": int(bin_count),
            "accuracy": float(bin_accuracy),
            "confidence": float(bin_confidence),
            "calibration_error": float(bin_error),
        })

    return ece, mce, bin_data


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics section
    """
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    # Classification metrics
    print("\nClassification Metrics:")
    print("-" * 80)
    for key in ["accuracy", "precision", "recall", "f1", "specificity"]:
        if key in metrics:
            print(f"  {key.capitalize():20s} {metrics[key]:7.4f}")

    # Ranking metrics
    print("\nRanking Metrics:")
    print("-" * 80)
    for key in ["roc_auc", "pr_auc"]:
        if key in metrics:
            print(f"  {key.upper():20s} {metrics[key]:7.4f}")

    # Calibration metrics
    print("\nCalibration Metrics:")
    print("-" * 80)
    for key in ["ece", "mce", "brier_score"]:
        if key in metrics:
            print(f"  {key.upper():20s} {metrics[key]:7.4f}")

    # Confusion matrix
    if "true_positives" in metrics:
        print("\nConfusion Matrix:")
        print("-" * 80)
        tp = metrics["true_positives"]
        tn = metrics["true_negatives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]

        print(f"                    Predicted")
        print(f"                Negative   Positive")
        print(f"  Actual")
        print(f"    Negative   {tn:8,}   {fp:8,}")
        print(f"    Positive   {fn:8,}   {tp:8,}")

    print("=" * 80)


def compute_per_type_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    entity_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics stratified by entity type.

    Args:
        labels: True labels [N]
        predictions: Predicted labels [N]
        probabilities: Predicted probabilities [N]
        entity_types: Entity type for each sample [N]

    Returns:
        Dictionary mapping entity_type -> metrics dict
    """
    unique_types = sorted(set(entity_types))

    type_metrics = {}

    for entity_type in unique_types:
        # Get samples for this type
        type_mask = np.array([t == entity_type for t in entity_types])

        if type_mask.sum() == 0:
            continue

        # Compute metrics
        type_labels = labels[type_mask]
        type_preds = predictions[type_mask]
        type_probs = probabilities[type_mask]

        metrics = compute_metrics(type_labels, type_preds, type_probs)
        metrics.update(compute_calibration_metrics(type_labels, type_probs))

        type_metrics[entity_type] = metrics

    return type_metrics


def plot_reliability_diagram(
    bin_data: List[Dict],
    save_path: Optional[str] = None,
):
    """
    Plot reliability diagram (calibration curve).

    Args:
        bin_data: Bin statistics from compute_ece
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed, cannot plot reliability diagram")
        return

    # Extract data
    confidences = [b["confidence"] for b in bin_data if b["count"] > 0]
    accuracies = [b["accuracy"] for b in bin_data if b["count"] > 0]
    counts = [b["count"] for b in bin_data if b["count"] > 0]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Plot actual calibration
    ax.scatter(confidences, accuracies, s=counts, alpha=0.6, label="Model Calibration")

    # Labels and formatting
    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Actual Frequency", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved reliability diagram to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot precision-recall curve.

    Args:
        labels: True labels
        probabilities: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ImportError:
        logger.warning("matplotlib not installed, cannot plot PR curve")
        return

    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved PR curve to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot ROC curve.

    Args:
        labels: True labels
        probabilities: Predicted probabilities
        save_path: Path to save plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve
    except ImportError:
        logger.warning("matplotlib not installed, cannot plot ROC curve")
        return

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    roc_auc = roc_auc_score(labels, probabilities)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {save_path}")
    else:
        plt.show()

    plt.close()
