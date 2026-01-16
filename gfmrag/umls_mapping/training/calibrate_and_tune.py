"""
Post-Training Calibration and Threshold Tuning

After training the cross-encoder (Day 2), run this script to:
1. Calibrate model probabilities using Platt scaling
2. Tune adaptive thresholds per entity type
3. Save calibrator and threshold tuner for production use

Usage:
    python -m gfmrag.umls_mapping.training.calibrate_and_tune \
        --model_path models/cross_encoder_finetuned/checkpoint-best \
        --val_data tmp/training/medmentions_splits.pkl \
        --output_dir models/calibration

This is Day 3 - Component 1&2 from STAGE3_PHASE2_PLAN.md
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from gfmrag.umls_mapping.training.calibration import (
    PlattScaling,
    TemperatureScaling,
    IsotonicCalibration,
    compare_calibration_methods,
)
from gfmrag.umls_mapping.training.adaptive_thresholds import AdaptiveThresholdTuner
from gfmrag.umls_mapping.training.dataset import CrossEncoderDataset
from gfmrag.umls_mapping.training.hard_negative_miner import HardNegativeMiner
from gfmrag.umls_mapping.training.models.binary_cross_encoder import BinaryCrossEncoder
from gfmrag.umls_mapping.umls_loader import UMLSLoader
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def collect_predictions(
    model: BinaryCrossEncoder,
    val_loader: DataLoader,
    umls_loader: UMLSLoader,
    device: torch.device,
):
    """
    Collect model predictions on validation set.

    Returns:
        Tuple of (logits, probabilities, labels, entity_types)
    """
    logger.info("Collecting validation predictions...")

    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []
    all_entity_types = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            batch_device = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device["attention_mask"],
            )

            all_logits.append(outputs["logits"].cpu().numpy())
            all_probs.append(outputs["probabilities"][:, 1].cpu().numpy())
            all_labels.append(batch_device["labels"].cpu().numpy())

            # Extract entity types (if available in metadata)
            # This would require modifying the dataset to include entity type info
            # For now, we'll extract from the dataset's mention data

    # Concatenate
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Collected {len(all_labels):,} predictions")

    return all_logits, all_probs, all_labels, all_entity_types


def extract_entity_types(
    val_mentions,
    umls_loader: UMLSLoader,
    num_samples_per_mention: int = 10,
):
    """
    Extract entity types for validation samples.

    Args:
        val_mentions: Validation mentions
        umls_loader: UMLS loader
        num_samples_per_mention: Number of samples per mention (1 pos + 9 neg)

    Returns:
        List of entity types (one per sample)
    """
    entity_types = []

    for mention in val_mentions:
        cui = mention["cui"]
        concept = umls_loader.concepts.get(cui)

        if concept and concept.semantic_types:
            entity_type = concept.semantic_types[0]
        else:
            entity_type = "Unknown"

        # Replicate for all samples from this mention
        entity_types.extend([entity_type] * num_samples_per_mention)

    return entity_types


def main():
    """Main calibration and tuning function."""
    parser = argparse.ArgumentParser(description="Calibrate model and tune thresholds")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data (pickle file)")
    parser.add_argument("--config", type=str, default="gfmrag/umls_mapping/training/config/training_config.yaml",
                       help="Training config")
    parser.add_argument("--output_dir", type=str, default="models/calibration", help="Output directory")
    parser.add_argument("--calibration_method", type=str, default="platt",
                       choices=["platt", "temperature", "isotonic", "compare"],
                       help="Calibration method")
    parser.add_argument("--threshold_objective", type=str, default="f1",
                       choices=["f1", "precision", "recall"],
                       help="Threshold optimization objective")
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

    # Load validation data
    logger.info(f"Loading validation data from {args.val_data}...")
    with open(args.val_data, "rb") as f:
        splits = pickle.load(f)
        val_mentions = splits["val"]

    logger.info(f"Validation mentions: {len(val_mentions):,}")

    # Mine negatives for validation set (if not cached)
    logger.info("Mining hard negatives for validation set...")
    hard_neg_config = config["hard_negatives"]

    cache_path = output_dir / "val_negatives_cache.pkl"
    if cache_path.exists():
        logger.info(f"Loading cached negatives from {cache_path}")
        with open(cache_path, "rb") as f:
            val_negatives = pickle.load(f)
    else:
        miner = HardNegativeMiner(
            umls_loader=umls_loader,
            faiss_index_path=hard_neg_config["faiss_index_path"],
            similarity_threshold=hard_neg_config["similarity_threshold"],
            top_k_candidates=hard_neg_config["top_k_candidates"],
            num_semantic_negatives=hard_neg_config["semantic_negatives"],
            num_type_negatives=hard_neg_config["type_negatives"],
            num_random_negatives=hard_neg_config["random_negatives"],
            cache_path=None,
            random_state=42,
        )
        val_negatives = miner.mine_negatives_batch(val_mentions, show_progress=True)

        # Cache for future use
        with open(cache_path, "wb") as f:
            pickle.dump(val_negatives, f)
        logger.info(f"Cached negatives to {cache_path}")

    # Create validation dataset
    logger.info("Creating validation dataset...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    val_dataset = CrossEncoderDataset(
        mentions=val_mentions,
        cui_to_negatives=val_negatives,
        umls_loader=umls_loader,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        positive_weight=config["loss"]["positive_weight"],
        hard_negative_weight=config["loss"]["hard_negative_weight"],
        easy_negative_weight=config["loss"]["easy_negative_weight"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load trained model
    logger.info(f"Loading trained model from {args.model_path}...")
    model = BinaryCrossEncoder(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )

    state_dict = torch.load(f"{args.model_path}/pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # Collect predictions
    all_logits, all_probs, all_labels, _ = collect_predictions(
        model, val_loader, umls_loader, device
    )

    # ========================================
    # STEP 1: Calibration
    # ========================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 1: PROBABILITY CALIBRATION")
    logger.info("=" * 80)
    logger.info("")

    if args.calibration_method == "compare":
        # Compare all methods
        results = compare_calibration_methods(all_logits, all_probs, all_labels)

        # Select best method based on ECE
        best_method = min(results.keys(), key=lambda k: results[k]["ece"])
        logger.info(f"Best calibration method: {best_method}")
        args.calibration_method = best_method

    # Fit selected calibration method
    if args.calibration_method == "platt":
        calibrator = PlattScaling()
        calibrator.fit(all_logits, all_labels)
        save_path = output_dir / "platt_scaler.pkl"
        calibrator.save(str(save_path))
        logger.info(f"Saved Platt scaling calibrator to {save_path}")

    elif args.calibration_method == "temperature":
        calibrator = TemperatureScaling()
        calibrator.fit(all_logits, all_labels)
        save_path = output_dir / "temperature_scaler.pkl"
        calibrator.save(str(save_path))
        logger.info(f"Saved temperature scaling calibrator to {save_path}")

    elif args.calibration_method == "isotonic":
        calibrator = IsotonicCalibration()
        calibrator.fit(all_probs, all_labels)
        save_path = output_dir / "isotonic_calibrator.pkl"
        calibrator.save(str(save_path))
        logger.info(f"Saved isotonic calibrator to {save_path}")

    # ========================================
    # STEP 2: Adaptive Threshold Tuning
    # ========================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: ADAPTIVE THRESHOLD TUNING")
    logger.info("=" * 80)
    logger.info("")

    # Extract entity types
    logger.info("Extracting entity types...")
    num_samples_per_mention = 1 + hard_neg_config["semantic_negatives"] + \
                              hard_neg_config["type_negatives"] + \
                              hard_neg_config["random_negatives"]

    entity_types = extract_entity_types(val_mentions, umls_loader, num_samples_per_mention)
    entity_types = entity_types[:len(all_labels)]  # Trim to actual sample count

    # Get calibrated probabilities
    if args.calibration_method in ["platt", "temperature"]:
        calibrated_probs = calibrator.predict_proba(all_logits)
    else:
        calibrated_probs = calibrator.predict_proba(all_probs)

    # Tune thresholds
    tuner = AdaptiveThresholdTuner(
        default_threshold=0.5,
        min_samples=100,
        objective=args.threshold_objective,
    )

    tuner.fit(entity_types, calibrated_probs, all_labels)

    # Save tuner
    tuner_path = output_dir / "adaptive_tuner.pkl"
    tuner.save(str(tuner_path))
    logger.info(f"Saved adaptive threshold tuner to {tuner_path}")

    # ========================================
    # STEP 3: Summary
    # ========================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("CALIBRATION AND TUNING COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  Calibrator:       {output_dir / (args.calibration_method + '_*')}")
    logger.info(f"  Threshold tuner:  {tuner_path}")
    logger.info(f"  Negatives cache:  {cache_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Test integration:")
    logger.info(f"   python -m gfmrag.umls_mapping.training.integration_test \\")
    logger.info(f"       --model_path {args.model_path} \\")
    logger.info(f"       --calibrator_path {save_path} \\")
    logger.info(f"       --threshold_tuner_path {tuner_path}")
    logger.info("")
    logger.info("2. Use in production:")
    logger.info("   Update your Stage 3 config with these paths and use")
    logger.info("   FineTunedCrossEncoderReranker instead of CrossEncoderReranker")
    logger.info("")


if __name__ == "__main__":
    main()
