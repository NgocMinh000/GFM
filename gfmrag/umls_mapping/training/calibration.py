"""
Probability Calibration for Cross-Encoder

Implements calibration methods to improve probability estimates:
- Platt Scaling (logistic regression on validation scores)
- Temperature Scaling
- Isotonic Regression

Well-calibrated probabilities ensure that:
- 80% confidence predictions are correct 80% of the time
- Confidence scores accurately reflect true likelihood

Reference:
- Platt (1999): "Probabilistic Outputs for Support Vector Machines"
- Guo et al. (2017): "On Calibration of Modern Neural Networks"
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

from gfmrag.umls_mapping.training.metrics import compute_ece

logger = logging.getLogger(__name__)


class PlattScaling:
    """
    Platt Scaling for probability calibration.

    Fits a logistic regression on validation set logits to calibrate probabilities.
    Maps raw model outputs to calibrated probabilities.

    Formula: P_calibrated(y=1|x) = 1 / (1 + exp(A*f(x) + B))
    where f(x) is the raw model output (logit).
    """

    def __init__(self):
        """Initialize Platt scaling calibrator."""
        self.model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.is_fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Fit Platt scaling on validation data.

        Args:
            logits: Raw model logits [N, 2] (binary classification)
            labels: True binary labels [N]
        """
        logger.info("Fitting Platt scaling calibrator...")

        # Extract logit differences (more numerically stable)
        # logit_diff = logit[positive] - logit[negative]
        if logits.ndim == 2:
            logit_diff = logits[:, 1] - logits[:, 0]
        else:
            logit_diff = logits

        # Reshape for sklearn
        X = logit_diff.reshape(-1, 1)
        y = labels

        # Fit logistic regression
        self.model.fit(X, y)
        self.is_fitted = True

        # Log calibration improvement
        uncalibrated_probs = 1 / (1 + np.exp(-logit_diff))
        calibrated_probs = self.predict_proba(logits)

        uncal_ece, _, _ = compute_ece(labels, uncalibrated_probs)
        cal_ece, _, _ = compute_ece(labels, calibrated_probs)

        logger.info(f"Calibration fitted:")
        logger.info(f"  ECE before: {uncal_ece:.4f}")
        logger.info(f"  ECE after:  {cal_ece:.4f}")
        logger.info(f"  Improvement: {(uncal_ece - cal_ece):.4f}")

    def predict_proba(
        self,
        logits: np.ndarray,
    ) -> np.ndarray:
        """
        Apply calibration to get calibrated probabilities.

        Args:
            logits: Raw model logits [N, 2] or [N]

        Returns:
            Calibrated probabilities for positive class [N]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        # Extract logit differences
        if logits.ndim == 2:
            logit_diff = logits[:, 1] - logits[:, 0]
        else:
            logit_diff = logits

        # Reshape for sklearn
        X = logit_diff.reshape(-1, 1)

        # Get calibrated probabilities
        calibrated_probs = self.model.predict_proba(X)[:, 1]

        return calibrated_probs

    def save(self, save_path: str):
        """Save calibrator to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
            }, f)

        logger.info(f"Saved Platt scaling calibrator to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "PlattScaling":
        """Load calibrator from disk."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls()
        calibrator.model = data['model']
        calibrator.is_fitted = data['is_fitted']

        logger.info(f"Loaded Platt scaling calibrator from {load_path}")

        return calibrator


class TemperatureScaling:
    """
    Temperature Scaling for probability calibration.

    Scales logits by a temperature parameter T before softmax:
    P_calibrated = softmax(logits / T)

    Simpler than Platt scaling, only one parameter to fit.
    """

    def __init__(self):
        """Initialize temperature scaling calibrator."""
        self.temperature = 1.0
        self.is_fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        method: str = "ece",
    ):
        """
        Fit temperature parameter on validation data.

        Args:
            logits: Raw model logits [N, 2]
            labels: True binary labels [N]
            method: Optimization objective ("ece" or "nll")
        """
        logger.info("Fitting temperature scaling calibrator...")

        # Convert to torch for optimization
        logits_torch = torch.tensor(logits, dtype=torch.float32)
        labels_torch = torch.tensor(labels, dtype=torch.long)

        # Initialize temperature
        temperature = torch.nn.Parameter(torch.ones(1))

        # Optimizer
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

        def eval_loss():
            optimizer.zero_grad()

            # Apply temperature scaling
            scaled_logits = logits_torch / temperature

            # Compute loss
            if method == "nll":
                # Negative log-likelihood
                loss = torch.nn.functional.cross_entropy(scaled_logits, labels_torch)
            else:
                # ECE approximation (use NLL as proxy)
                loss = torch.nn.functional.cross_entropy(scaled_logits, labels_torch)

            loss.backward()
            return loss

        # Optimize temperature
        optimizer.step(eval_loss)

        self.temperature = temperature.item()
        self.is_fitted = True

        # Log calibration improvement
        uncalibrated_probs = torch.softmax(logits_torch, dim=1)[:, 1].numpy()
        calibrated_probs = self.predict_proba(logits)

        uncal_ece, _, _ = compute_ece(labels, uncalibrated_probs)
        cal_ece, _, _ = compute_ece(labels, calibrated_probs)

        logger.info(f"Temperature scaling fitted:")
        logger.info(f"  Temperature: {self.temperature:.4f}")
        logger.info(f"  ECE before: {uncal_ece:.4f}")
        logger.info(f"  ECE after:  {cal_ece:.4f}")
        logger.info(f"  Improvement: {(uncal_ece - cal_ece):.4f}")

    def predict_proba(
        self,
        logits: np.ndarray,
    ) -> np.ndarray:
        """
        Apply temperature scaling to get calibrated probabilities.

        Args:
            logits: Raw model logits [N, 2]

        Returns:
            Calibrated probabilities for positive class [N]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        # Apply temperature scaling
        logits_torch = torch.tensor(logits, dtype=torch.float32)
        scaled_logits = logits_torch / self.temperature

        # Softmax
        calibrated_probs = torch.softmax(scaled_logits, dim=1)[:, 1].numpy()

        return calibrated_probs

    def save(self, save_path: str):
        """Save calibrator to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'temperature': self.temperature,
                'is_fitted': self.is_fitted,
            }, f)

        logger.info(f"Saved temperature scaling calibrator to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "TemperatureScaling":
        """Load calibrator from disk."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls()
        calibrator.temperature = data['temperature']
        calibrator.is_fitted = data['is_fitted']

        logger.info(f"Loaded temperature scaling calibrator from {load_path}")

        return calibrator


class IsotonicCalibration:
    """
    Isotonic Regression for probability calibration.

    Non-parametric method that learns a monotonic mapping from
    uncalibrated to calibrated probabilities.

    More flexible than Platt/Temperature scaling, but requires more data.
    """

    def __init__(self):
        """Initialize isotonic calibrator."""
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Fit isotonic regression on validation data.

        Args:
            probabilities: Uncalibrated probabilities [N]
            labels: True binary labels [N]
        """
        logger.info("Fitting isotonic calibrator...")

        # Fit isotonic regression
        self.model.fit(probabilities, labels)
        self.is_fitted = True

        # Log calibration improvement
        calibrated_probs = self.predict_proba(probabilities)

        uncal_ece, _, _ = compute_ece(labels, probabilities)
        cal_ece, _, _ = compute_ece(labels, calibrated_probs)

        logger.info(f"Isotonic calibration fitted:")
        logger.info(f"  ECE before: {uncal_ece:.4f}")
        logger.info(f"  ECE after:  {cal_ece:.4f}")
        logger.info(f"  Improvement: {(uncal_ece - cal_ece):.4f}")

    def predict_proba(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Apply isotonic calibration.

        Args:
            probabilities: Uncalibrated probabilities [N]

        Returns:
            Calibrated probabilities [N]
        """
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        calibrated_probs = self.model.predict(probabilities)

        return calibrated_probs

    def save(self, save_path: str):
        """Save calibrator to disk."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
            }, f)

        logger.info(f"Saved isotonic calibrator to {save_path}")

    @classmethod
    def load(cls, load_path: str) -> "IsotonicCalibration":
        """Load calibrator from disk."""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        calibrator = cls()
        calibrator.model = data['model']
        calibrator.is_fitted = data['is_fitted']

        logger.info(f"Loaded isotonic calibrator from {load_path}")

        return calibrator


def calibrate_model(
    model,
    val_loader,
    device: torch.device,
    method: str = "platt",
    save_path: Optional[str] = None,
):
    """
    Calibrate a trained model on validation data.

    Args:
        model: Trained cross-encoder model
        val_loader: Validation data loader
        device: Computation device
        method: Calibration method ("platt", "temperature", or "isotonic")
        save_path: Path to save calibrator (optional)

    Returns:
        Fitted calibrator
    """
    logger.info(f"Calibrating model using {method} scaling...")

    # Collect validation predictions
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            all_logits.append(outputs["logits"].cpu().numpy())
            all_probs.append(outputs["probabilities"][:, 1].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    # Concatenate
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Fit calibrator
    if method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(all_logits, all_labels)
    elif method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(all_logits, all_labels)
    elif method == "isotonic":
        calibrator = IsotonicCalibration()
        calibrator.fit(all_probs, all_labels)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    # Save calibrator
    if save_path:
        calibrator.save(save_path)

    return calibrator


def compare_calibration_methods(
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Dict]:
    """
    Compare different calibration methods on validation data.

    Args:
        logits: Raw model logits [N, 2]
        probabilities: Uncalibrated probabilities [N]
        labels: True labels [N]

    Returns:
        Dictionary with metrics for each method
    """
    results = {}

    # Uncalibrated baseline
    uncal_ece, _, _ = compute_ece(labels, probabilities)
    uncal_brier = brier_score_loss(labels, probabilities)
    uncal_nll = log_loss(labels, probabilities)

    results["uncalibrated"] = {
        "ece": uncal_ece,
        "brier_score": uncal_brier,
        "nll": uncal_nll,
    }

    # Platt scaling
    platt = PlattScaling()
    platt.fit(logits, labels)
    platt_probs = platt.predict_proba(logits)
    platt_ece, _, _ = compute_ece(labels, platt_probs)
    platt_brier = brier_score_loss(labels, platt_probs)
    platt_nll = log_loss(labels, platt_probs)

    results["platt"] = {
        "ece": platt_ece,
        "brier_score": platt_brier,
        "nll": platt_nll,
    }

    # Temperature scaling
    temp = TemperatureScaling()
    temp.fit(logits, labels)
    temp_probs = temp.predict_proba(logits)
    temp_ece, _, _ = compute_ece(labels, temp_probs)
    temp_brier = brier_score_loss(labels, temp_probs)
    temp_nll = log_loss(labels, temp_probs)

    results["temperature"] = {
        "ece": temp_ece,
        "brier_score": temp_brier,
        "nll": temp_nll,
        "temperature": temp.temperature,
    }

    # Isotonic regression
    isotonic = IsotonicCalibration()
    isotonic.fit(probabilities, labels)
    isotonic_probs = isotonic.predict_proba(probabilities)
    isotonic_ece, _, _ = compute_ece(labels, isotonic_probs)
    isotonic_brier = brier_score_loss(labels, isotonic_probs)
    isotonic_nll = log_loss(labels, isotonic_probs)

    results["isotonic"] = {
        "ece": isotonic_ece,
        "brier_score": isotonic_brier,
        "nll": isotonic_nll,
    }

    # Print comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("CALIBRATION METHOD COMPARISON")
    logger.info("=" * 80)
    logger.info(f"{'Method':<15} {'ECE':<10} {'Brier':<10} {'NLL':<10}")
    logger.info("-" * 80)

    for method, metrics in results.items():
        logger.info(f"{method:<15} {metrics['ece']:<10.4f} {metrics['brier_score']:<10.4f} {metrics['nll']:<10.4f}")

    logger.info("=" * 80)
    logger.info("")

    return results
