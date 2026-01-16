"""
PyTorch Dataset for Cross-Encoder Training

Combines MedMentions data + hard negatives for UMLS entity linking training.

Training format:
- Input: (entity_mention, candidate_cui_name)
- Output: label (1 for correct match, 0 for incorrect)
- Sample weights: Higher weight for hard negatives
"""

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class CrossEncoderDataset(Dataset):
    """
    PyTorch Dataset for cross-encoder training.

    For each positive example (mention, correct_cui), generates:
    - 1 positive sample: (mention, correct_cui_name, label=1)
    - N negative samples: (mention, negative_cui_name, label=0)

    Negative breakdown:
    - K semantic hard negatives (weight=1.5x)
    - M type negatives (weight=1.0x)
    - L random negatives (weight=0.5x)
    """

    def __init__(
        self,
        mentions: List[Dict],
        cui_to_negatives: Dict[str, Dict],
        umls_loader,
        tokenizer,
        max_length: int = 128,
        positive_weight: float = 1.0,
        hard_negative_weight: float = 1.5,
        easy_negative_weight: float = 0.5,
    ):
        """
        Initialize dataset.

        Args:
            mentions: List of mention dicts from MedMentions
            cui_to_negatives: Dictionary of mined negatives per CUI
            umls_loader: UMLS loader with concept names
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            positive_weight: Weight for positive samples
            hard_negative_weight: Weight for semantic hard negatives
            easy_negative_weight: Weight for random negatives
        """
        self.mentions = mentions
        self.cui_to_negatives = cui_to_negatives
        self.umls_loader = umls_loader
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.positive_weight = positive_weight
        self.hard_negative_weight = hard_negative_weight
        self.easy_negative_weight = easy_negative_weight

        # Build flat list of training samples
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        """Build flat list of (mention, cui, label, weight) tuples."""
        logger.info(f"Building training samples from {len(self.mentions)} mentions")

        for mention in self.mentions:
            mention_text = mention["mention_text"]
            correct_cui = mention["cui"]

            # Get concept name
            concept = self.umls_loader.concepts.get(correct_cui)
            if not concept:
                continue

            correct_name = concept.preferred_name

            # Positive sample
            self.samples.append({
                "mention": mention_text,
                "cui_name": correct_name,
                "label": 1,
                "weight": self.positive_weight,
                "type": "positive",
            })

            # Negative samples
            negatives = self.cui_to_negatives.get(correct_cui, {})

            # Semantic hard negatives
            for neg_cui, sim in negatives.get("semantic_negatives", []):
                neg_concept = self.umls_loader.concepts.get(neg_cui)
                if neg_concept:
                    self.samples.append({
                        "mention": mention_text,
                        "cui_name": neg_concept.preferred_name,
                        "label": 0,
                        "weight": self.hard_negative_weight,
                        "type": "semantic_hard",
                    })

            # Type negatives
            for neg_cui in negatives.get("type_negatives", []):
                neg_concept = self.umls_loader.concepts.get(neg_cui)
                if neg_concept:
                    self.samples.append({
                        "mention": mention_text,
                        "cui_name": neg_concept.preferred_name,
                        "label": 0,
                        "weight": self.positive_weight,  # Standard weight
                        "type": "type_negative",
                    })

            # Random negatives
            for neg_cui in negatives.get("random_negatives", []):
                neg_concept = self.umls_loader.concepts.get(neg_cui)
                if neg_concept:
                    self.samples.append({
                        "mention": mention_text,
                        "cui_name": neg_concept.preferred_name,
                        "label": 0,
                        "weight": self.easy_negative_weight,
                        "type": "random",
                    })

        logger.info(f"Built {len(self.samples)} training samples")

        # Print distribution
        type_counts = {}
        for sample in self.samples:
            sample_type = sample["type"]
            type_counts[sample_type] = type_counts.get(sample_type, 0) + 1

        logger.info("Sample distribution:")
        for stype, count in sorted(type_counts.items()):
            pct = count / len(self.samples) * 100
            logger.info(f"  {stype:20} {count:8,} ({pct:5.2f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            Dictionary with:
            - input_ids: Token IDs
            - attention_mask: Attention mask
            - labels: Binary label (0 or 1)
            - weight: Sample weight
        """
        sample = self.samples[idx]

        # Format input: [CLS] mention [SEP] cui_name [SEP]
        text = f"{sample['mention']} [SEP] {sample['cui_name']}"

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(sample["label"], dtype=torch.long),
            "weight": torch.tensor(sample["weight"], dtype=torch.float),
        }


def get_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
):
    """
    Create DataLoader for training.

    Args:
        dataset: CrossEncoderDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers

    Returns:
        torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
