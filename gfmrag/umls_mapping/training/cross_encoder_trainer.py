"""
Cross-Encoder Training Script

Fine-tunes PubMedBERT as binary cross-encoder for UMLS entity linking.

Usage:
    python -m gfmrag.umls_mapping.training.cross_encoder_trainer \
        --config gfmrag/umls_mapping/training/config/training_config.yaml \
        --output_dir models/cross_encoder_finetuned

Features:
- Mixed precision training (fp16) for 2-3x speedup
- Hard negative contrastive learning
- Weighted BCE loss (emphasize hard negatives)
- Early stopping and checkpointing
- TensorBoard logging
- Gradient accumulation for large effective batch size
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import yaml

from gfmrag.umls_mapping.training.data_loader import load_medmentions
from gfmrag.umls_mapping.training.dataset import CrossEncoderDataset, get_dataloader
from gfmrag.umls_mapping.training.hard_negative_miner import HardNegativeMiner
from gfmrag.umls_mapping.training.models.binary_cross_encoder import BinaryCrossEncoder
from gfmrag.umls_mapping.training.metrics import compute_metrics, compute_calibration_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CrossEncoderTrainer:
    """
    Trainer for binary cross-encoder.

    Handles training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: BinaryCrossEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: str,
    ):
        """
        Initialize trainer.

        Args:
            model: Binary cross-encoder model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            output_dir: Output directory for checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = self._setup_device()
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Mixed precision scaler
        self.use_fp16 = config["training"].get("fp16", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0
        self.best_checkpoint = None
        self.patience_counter = 0

        # TensorBoard logging
        log_dir = config["logging"].get("tensorboard", {}).get("log_dir", "tmp/training/tensorboard")
        self.writer = SummaryWriter(log_dir=log_dir)

        # Gradient accumulation
        self.gradient_accumulation_steps = config["training"].get("gradient_accumulation_steps", 1)

        logger.info(f"Trainer initialized. Output dir: {output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_fp16}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_name = self.config["hardware"].get("device", "cuda")

        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA not available, using CPU (training will be slow)")

        return device

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup AdamW optimizer."""
        training_config = self.config["training"]

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config["learning_rate"],
            weight_decay=training_config.get("weight_decay", 0.01),
            eps=training_config.get("adam_epsilon", 1e-8),
        )

        logger.info(f"Optimizer: AdamW(lr={training_config['learning_rate']}, "
                    f"weight_decay={training_config.get('weight_decay', 0.01)})")

        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        training_config = self.config["training"]

        num_epochs = training_config["epochs"]
        num_training_steps = len(self.train_loader) * num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * training_config.get("warmup_ratio", 0.1))

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(f"Scheduler: Linear warmup + decay")
        logger.info(f"  Total steps: {num_training_steps:,}")
        logger.info(f"  Warmup steps: {num_warmup_steps:,}")

        return scheduler

    def train(self):
        """Run full training loop."""
        num_epochs = self.config["training"]["epochs"]

        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Training samples: {len(self.train_loader.dataset):,}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset):,}")
        logger.info(f"Batch size: {self.config['training']['batch_size']}")
        logger.info(f"Effective batch size: {self.config['training']['batch_size'] * self.gradient_accumulation_steps}")
        logger.info("=" * 80)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics)

            # Save checkpoint
            self._save_checkpoint(val_metrics)

            # Early stopping check
            if self._check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Load best model
        self._load_best_checkpoint()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        logger.info(f"Best checkpoint: {self.best_checkpoint}")
        logger.info("=" * 80)

        self.writer.close()

    def train_epoch(self) -> Dict:
        """Train one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1} [Train]",
            leave=True,
        )

        # Zero gradients at start
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with mixed precision
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        weights=batch.get("weight"),
                    )
                    loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaled gradients
                self.scaler.scale(loss).backward()

                # Update weights every N steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale and clip gradients
                    self.scaler.unscale_(self.optimizer)
                    max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1

            else:
                # Standard forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    weights=batch.get("weight"),
                )
                loss = outputs["loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Update weights every N steps
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    max_grad_norm = self.config["training"].get("max_grad_norm", 1.0)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                    self.global_step += 1

            # Accumulate loss (unscaled for logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

            # Log to TensorBoard
            if self.global_step % self.config["logging"].get("logging_steps", 100) == 0:
                self.writer.add_scalar("train/loss", loss.item() * self.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar("train/learning_rate", self.scheduler.get_last_lr()[0], self.global_step)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict:
        """Validate on validation set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch + 1} [Val]  ",
            leave=True,
        )

        for batch in progress_bar:
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
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            total_loss += outputs["loss"].item()
            num_batches += 1

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        metrics["loss"] = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute calibration metrics
        calibration = compute_calibration_metrics(all_labels, all_probs)
        metrics.update(calibration)

        return metrics

    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics for epoch."""
        logger.info("")
        logger.info(f"Epoch {self.epoch + 1} Results:")
        logger.info(f"  Train Loss:      {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss:        {val_metrics['loss']:.4f}")
        logger.info(f"  Val Accuracy:    {val_metrics['accuracy']:.4f}")
        logger.info(f"  Val Precision:   {val_metrics['precision']:.4f}")
        logger.info(f"  Val Recall:      {val_metrics['recall']:.4f}")
        logger.info(f"  Val F1:          {val_metrics['f1']:.4f}")
        logger.info(f"  Val ROC-AUC:     {val_metrics['roc_auc']:.4f}")
        logger.info(f"  Val PR-AUC:      {val_metrics['pr_auc']:.4f}")
        logger.info(f"  Val ECE:         {val_metrics['ece']:.4f}")
        logger.info(f"  Val Brier Score: {val_metrics['brier_score']:.4f}")
        logger.info("")

        # TensorBoard logging
        self.writer.add_scalar("epoch/train_loss", train_metrics["loss"], self.epoch)
        self.writer.add_scalar("epoch/val_loss", val_metrics["loss"], self.epoch)
        self.writer.add_scalar("epoch/val_f1", val_metrics["f1"], self.epoch)
        self.writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], self.epoch)
        self.writer.add_scalar("epoch/val_roc_auc", val_metrics["roc_auc"], self.epoch)
        self.writer.add_scalar("epoch/val_ece", val_metrics["ece"], self.epoch)

    def _save_checkpoint(self, val_metrics: Dict):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-epoch-{self.epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_dir))

        # Save metrics
        metrics_file = checkpoint_dir / "metrics.yaml"
        with open(metrics_file, "w") as f:
            yaml.dump(val_metrics, f)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Track best model
        if val_metrics["f1"] > self.best_val_f1:
            self.best_val_f1 = val_metrics["f1"]
            self.best_checkpoint = checkpoint_dir

            # Save as "best" checkpoint
            best_dir = self.output_dir / "checkpoint-best"
            if best_dir.exists():
                import shutil
                shutil.rmtree(best_dir)
            import shutil
            shutil.copytree(checkpoint_dir, best_dir)

            logger.info(f"New best model! F1: {self.best_val_f1:.4f}")

        # Clean up old checkpoints (keep only best + latest N)
        save_total_limit = self.config["training"].get("save_total_limit", 3)
        self._cleanup_checkpoints(save_total_limit)

    def _cleanup_checkpoints(self, keep_last_n: int):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(
            [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")],
            key=lambda x: int(x.name.split("-")[-1]),
        )

        # Keep best + last N
        to_remove = checkpoints[:-keep_last_n]

        for checkpoint in to_remove:
            # Don't remove if it's the best checkpoint
            if checkpoint != self.best_checkpoint:
                import shutil
                shutil.rmtree(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint.name}")

    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """Check early stopping condition."""
        early_stopping_config = self.config["training"].get("early_stopping", {})

        if not early_stopping_config.get("enabled", False):
            return False

        patience = early_stopping_config.get("patience", 2)
        monitor = early_stopping_config.get("monitor", "val_f1")
        mode = early_stopping_config.get("mode", "max")

        current_value = val_metrics.get(monitor.replace("val_", ""), 0.0)

        # Check if improved
        if mode == "max":
            improved = current_value > self.best_val_f1
        else:
            improved = current_value < self.best_val_f1

        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= patience

    def _load_best_checkpoint(self):
        """Load best checkpoint at end of training."""
        if self.best_checkpoint:
            logger.info(f"Loading best checkpoint: {self.best_checkpoint}")
            state_dict = torch.load(
                self.best_checkpoint / "pytorch_model.bin",
                map_location=self.device,
            )
            self.model.load_state_dict(state_dict)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train binary cross-encoder for UMLS entity linking")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override output dir if provided
    output_dir = args.output_dir or config["model"]["checkpoint_dir"]

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Load UMLS
    logger.info("Loading UMLS...")
    from gfmrag.umls_mapping.umls_loader import UMLSLoader
    from gfmrag.umls_mapping.config import UMLSMappingConfig

    # Create UMLS config
    umls_config = UMLSMappingConfig(
        kg_clean_path='dummy',  # Not needed for training
        umls_data_dir='data/umls',
        output_root='tmp/umls_training',
        mrconso_path='data/umls/META/MRCONSO.RRF',
        mrsty_path='data/umls/META/MRSTY.RRF',
        umls_cache_dir='data/umls/processed',
    )

    umls_loader = UMLSLoader(umls_config)
    umls_loader.load()

    # Load MedMentions dataset
    logger.info("Loading MedMentions dataset...")
    medmentions_config = config["dataset"]["medmentions"]
    train_mentions, val_mentions, test_mentions = load_medmentions(
        data_path=medmentions_config["path"],
        umls_loader=umls_loader,
        train_ratio=config["dataset"]["train_ratio"],
        val_ratio=config["dataset"]["val_ratio"],
        test_ratio=config["dataset"]["test_ratio"],
        stratify=True,
        cache_path="tmp/training/medmentions_splits.pkl",
        random_state=seed,
    )

    logger.info(f"Train: {len(train_mentions):,}, Val: {len(val_mentions):,}, Test: {len(test_mentions):,}")

    # Mine hard negatives
    logger.info("Mining hard negatives...")
    hard_neg_config = config["hard_negatives"]
    miner = HardNegativeMiner(
        umls_loader=umls_loader,
        faiss_index_path=hard_neg_config["faiss_index_path"],
        similarity_threshold=hard_neg_config["similarity_threshold"],
        top_k_candidates=hard_neg_config["top_k_candidates"],
        num_semantic_negatives=hard_neg_config["semantic_negatives"],
        num_type_negatives=hard_neg_config["type_negatives"],
        num_random_negatives=hard_neg_config["random_negatives"],
        cache_path=hard_neg_config.get("cache_path") if hard_neg_config.get("cache_hard_negatives") else None,
        random_state=seed,
    )

    train_negatives = miner.mine_negatives_batch(train_mentions, show_progress=True)
    val_negatives = miner.mine_negatives_batch(val_mentions, show_progress=True)

    # Print statistics
    miner.print_statistics()

    # Create datasets
    logger.info("Creating PyTorch datasets...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    train_dataset = CrossEncoderDataset(
        mentions=train_mentions,
        cui_to_negatives=train_negatives,
        umls_loader=umls_loader,
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        positive_weight=config["loss"]["positive_weight"],
        hard_negative_weight=config["loss"]["hard_negative_weight"],
        easy_negative_weight=config["loss"]["easy_negative_weight"],
    )

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

    # Create data loaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"].get("num_workers", 4),
    )

    val_loader = get_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"].get("num_workers", 4),
    )

    # Initialize model
    logger.info("Initializing model...")
    model = BinaryCrossEncoder(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        state_dict = torch.load(f"{args.resume_from}/pytorch_model.bin")
        model.load_state_dict(state_dict)

    # Initialize trainer
    trainer = CrossEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
    )

    # Train
    trainer.train()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
