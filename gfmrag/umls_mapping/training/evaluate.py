"""
Evaluation Script for Fine-Tuned Cross-Encoder

Evaluates a trained cross-encoder model on test set and generates:
- Comprehensive metrics report
- Calibration analysis
- Per-entity-type breakdown
- Visualization plots

Usage:
    python -m gfmrag.umls_mapping.training.evaluate \
        --model_path models/cross_encoder_finetuned/checkpoint-best \
        --test_data tmp/training/medmentions_splits.pkl \
        --output_dir results/evaluation
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from gfmrag.umls_mapping.training.dataset import CrossEncoderDataset
from gfmrag.umls_mapping.training.hard_negative_miner import HardNegativeMiner
from gfmrag.umls_mapping.training.metrics import (
    compute_metrics,
    compute_calibration_metrics,
    compute_per_type_metrics,
    print_metrics,
    plot_reliability_diagram,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from gfmrag.umls_mapping.training.models.binary_cross_encoder import BinaryCrossEncoder
from gfmrag.umls_mapping.umls_loader import UMLSLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates fine-tuned cross-encoder model.
    """

    def __init__(
        self,
        model: BinaryCrossEncoder,
        tokenizer,
        umls_loader: UMLSLoader,
        device: torch.device,
    ):
        """
        Initialize evaluator.

        Args:
            model: Fine-tuned binary cross-encoder
            tokenizer: HuggingFace tokenizer
            umls_loader: UMLS loader
            device: Computation device
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.umls_loader = umls_loader
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        entity_types: List[str] = None,
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            entity_types: Entity type for each sample (for per-type metrics)

        Returns:
            Dictionary with:
            - overall_metrics: Overall performance metrics
            - per_type_metrics: Metrics stratified by entity type (if provided)
            - predictions: List of (label, pred, prob) tuples
        """
        logger.info("Running evaluation...")

        all_labels = []
        all_preds = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0

        # Evaluate
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                weights=batch.get("weight"),
            )

            # Get predictions
            probs = outputs["probabilities"][:, 1]  # Probability of positive class
            preds = (probs > 0.5).long()

            # Collect results
            all_labels.extend(batch["labels"].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_loss += outputs["loss"].item()
            num_batches += 1

        # Convert to numpy
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Compute overall metrics
        logger.info("Computing metrics...")
        overall_metrics = compute_metrics(all_labels, all_preds, all_probs)
        overall_metrics.update(compute_calibration_metrics(all_labels, all_probs))
        overall_metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute per-type metrics if entity types provided
        per_type_metrics = None
        if entity_types is not None:
            logger.info("Computing per-type metrics...")
            per_type_metrics = compute_per_type_metrics(
                all_labels, all_preds, all_probs, entity_types
            )

        # Package results
        results = {
            "overall_metrics": overall_metrics,
            "per_type_metrics": per_type_metrics,
            "predictions": list(zip(
                all_labels.tolist(),
                all_preds.tolist(),
                all_probs.tolist(),
            )),
        }

        return results

    def analyze_errors(
        self,
        predictions: List[tuple],
        mentions: List[Dict],
        top_k: int = 50,
    ) -> Dict:
        """
        Analyze prediction errors.

        Args:
            predictions: List of (label, pred, prob) tuples
            mentions: Original mention data
            top_k: Number of top errors to return

        Returns:
            Dictionary with error analysis
        """
        logger.info("Analyzing errors...")

        # Separate error types
        false_positives = []  # Predicted positive, actually negative
        false_negatives = []  # Predicted negative, actually positive

        for i, (label, pred, prob) in enumerate(predictions):
            if label == 0 and pred == 1:
                # False positive
                false_positives.append({
                    "index": i,
                    "mention": mentions[i // 10] if i < len(mentions) * 10 else None,  # Approximate
                    "confidence": prob,
                })
            elif label == 1 and pred == 0:
                # False negative
                false_negatives.append({
                    "index": i,
                    "mention": mentions[i // 10] if i < len(mentions) * 10 else None,  # Approximate
                    "confidence": prob,
                })

        # Sort by confidence (most confident errors are most interesting)
        false_positives = sorted(false_positives, key=lambda x: x["confidence"], reverse=True)[:top_k]
        false_negatives = sorted(false_negatives, key=lambda x: x["confidence"])[:top_k]

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "num_false_positives": len(false_positives),
            "num_false_negatives": len(false_negatives),
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned cross-encoder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (pickle file)")
    parser.add_argument("--config", type=str, default="gfmrag/umls_mapping/training/config/training_config.yaml",
                        help="Training config (for dataset parameters)")
    parser.add_argument("--output_dir", type=str, default="results/evaluation", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load UMLS
    logger.info("Loading UMLS...")
    umls_loader = UMLSLoader()

    # Load test data
    logger.info(f"Loading test data from {args.test_data}...")
    with open(args.test_data, "rb") as f:
        splits = pickle.load(f)
        test_mentions = splits["test"]

    logger.info(f"Test samples: {len(test_mentions):,}")

    # Mine negatives for test set
    logger.info("Mining hard negatives for test set...")
    hard_neg_config = config["hard_negatives"]
    miner = HardNegativeMiner(
        umls_loader=umls_loader,
        faiss_index_path=hard_neg_config["faiss_index_path"],
        similarity_threshold=hard_neg_config["similarity_threshold"],
        top_k_candidates=hard_neg_config["top_k_candidates"],
        num_semantic_negatives=hard_neg_config["semantic_negatives"],
        num_type_negatives=hard_neg_config["type_negatives"],
        num_random_negatives=hard_neg_config["random_negatives"],
        cache_path=None,  # Don't cache test negatives
        random_state=42,
    )
    test_negatives = miner.mine_negatives_batch(test_mentions, show_progress=True)

    # Create test dataset
    logger.info("Creating test dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    test_dataset = CrossEncoderDataset(
        mentions=test_mentions,
        cui_to_negatives=test_negatives,
        umls_loader=umls_loader,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        positive_weight=config["loss"]["positive_weight"],
        hard_negative_weight=config["loss"]["hard_negative_weight"],
        easy_negative_weight=config["loss"]["easy_negative_weight"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = BinaryCrossEncoder(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )

    state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)

    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        umls_loader=umls_loader,
        device=device,
    )

    # Extract entity types for per-type metrics
    entity_types = []
    for mention in test_mentions:
        cui = mention["cui"]
        concept = umls_loader.concepts.get(cui)
        if concept and concept.semantic_types:
            entity_types.append(concept.semantic_types[0])
        else:
            entity_types.append("Unknown")

    # Replicate for all samples (positives + negatives)
    samples_per_mention = 1 + len(test_negatives.get(test_mentions[0]["cui"], {}).get("semantic_negatives", [])) + \
                          len(test_negatives.get(test_mentions[0]["cui"], {}).get("type_negatives", [])) + \
                          len(test_negatives.get(test_mentions[0]["cui"], {}).get("random_negatives", []))
    entity_types_full = []
    for et in entity_types:
        entity_types_full.extend([et] * samples_per_mention)

    # Evaluate
    results = evaluator.evaluate(test_loader, entity_types=entity_types_full[:len(test_dataset)])

    # Print overall metrics
    print_metrics(results["overall_metrics"], title="Test Set Evaluation")

    # Print per-type metrics
    if results["per_type_metrics"]:
        print("\n")
        print("=" * 80)
        print(f"{'PER-TYPE METRICS':^80}")
        print("=" * 80)

        for entity_type, metrics in sorted(results["per_type_metrics"].items()):
            print(f"\n{entity_type}:")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  ECE:       {metrics['ece']:.4f}")

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        # Convert numpy types for JSON serialization
        serializable_results = {
            "overall_metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                                for k, v in results["overall_metrics"].items()
                                if k != "reliability_data"},
            "per_type_metrics": {
                entity_type: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in metrics.items() if k != "reliability_data"}
                for entity_type, metrics in (results["per_type_metrics"] or {}).items()
            } if results["per_type_metrics"] else None,
        }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Saved results to {results_file}")

    # Generate plots
    logger.info("Generating visualization plots...")

    labels = np.array([p[0] for p in results["predictions"]])
    probs = np.array([p[2] for p in results["predictions"]])

    # Reliability diagram
    bin_data = results["overall_metrics"].get("reliability_data", [])
    if bin_data:
        plot_reliability_diagram(bin_data, save_path=str(output_dir / "reliability_diagram.png"))

    # PR curve
    plot_precision_recall_curve(labels, probs, save_path=str(output_dir / "pr_curve.png"))

    # ROC curve
    plot_roc_curve(labels, probs, save_path=str(output_dir / "roc_curve.png"))

    logger.info(f"Saved plots to {output_dir}")

    # Error analysis
    # error_analysis = evaluator.analyze_errors(results["predictions"], test_mentions, top_k=50)
    # logger.info(f"False positives: {error_analysis['num_false_positives']}")
    # logger.info(f"False negatives: {error_analysis['num_false_negatives']}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
