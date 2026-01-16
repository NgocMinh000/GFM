"""
Binary Cross-Encoder for UMLS Entity Linking

Architecture:
- Base: PubMedBERT (biomedical domain pre-training)
- Head: Binary classification (correct vs incorrect match)
- Input: [CLS] entity_mention [SEP] cui_name [SEP]
- Output: Probability that (mention, cui) is a correct match

Training:
- Weighted BCE loss (emphasize hard negatives)
- Optional contrastive loss component
- Mixed precision (fp16) for faster training
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BinaryCrossEncoder(nn.Module):
    """
    Binary cross-encoder for UMLS entity linking.

    Outputs a probability score for whether (entity_mention, cui_name)
    is a correct match.
    """

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Initialize binary cross-encoder.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of labels (2 for binary)
            dropout: Dropout rate for classification head
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pre-trained model
        logger.info(f"Loading base model: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze base model if requested (only train classifier)
        if freeze_base:
            logger.info("Freezing base model parameters")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Initialize classifier weights
        self._init_weights()

        logger.info(f"Initialized BinaryCrossEncoder with {self.count_parameters():,} parameters")

    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Binary labels [batch_size] (optional)
            weights: Sample weights [batch_size] (optional)

        Returns:
            Dictionary with:
            - logits: Raw logits [batch_size, 2]
            - probabilities: Softmax probabilities [batch_size, 2]
            - loss: Weighted BCE loss (if labels provided)
        """
        # Encode input
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Apply dropout and classifier
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        # Compute probabilities
        probabilities = F.softmax(logits, dim=-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
        }

        # Compute loss if labels provided
        if labels is not None:
            loss = self.compute_loss(logits, labels, weights)
            result["loss"] = loss

        return result

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted cross-entropy loss.

        Args:
            logits: Raw logits [batch_size, num_labels]
            labels: Binary labels [batch_size]
            weights: Sample weights [batch_size] (optional)

        Returns:
            Scalar loss tensor
        """
        # Standard cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction="none")

        # Apply sample weights
        if weights is not None:
            loss = loss * weights

        # Average loss
        return loss.mean()

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict match probability.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]

        Returns:
            Tuple of (predictions, probabilities):
            - predictions: Binary predictions [batch_size]
            - probabilities: Probability of positive class [batch_size]
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            probabilities = outputs["probabilities"]

            # Get probability of positive class (index 1)
            positive_probs = probabilities[:, 1]

            # Binary predictions (threshold=0.5)
            predictions = (positive_probs > 0.5).long()

        return predictions, positive_probs

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "BinaryCrossEncoder":
        """
        Load fine-tuned model from checkpoint.

        Args:
            model_path: Path to model checkpoint directory
            **kwargs: Additional model arguments

        Returns:
            Loaded BinaryCrossEncoder instance
        """
        logger.info(f"Loading fine-tuned model from {model_path}")

        # Load model
        model = cls(**kwargs)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict)

        logger.info(f"Loaded model with {model.count_parameters():,} parameters")

        return model

    def save_pretrained(self, save_path: str):
        """
        Save model checkpoint.

        Args:
            save_path: Directory to save model
        """
        import os
        os.makedirs(save_path, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), f"{save_path}/pytorch_model.bin")

        # Save config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
        }
        torch.save(config, f"{save_path}/config.pt")

        logger.info(f"Saved model checkpoint to {save_path}")


class ContrastiveCrossEncoder(BinaryCrossEncoder):
    """
    Cross-encoder with additional contrastive loss.

    Combines:
    1. Binary classification loss (BCE)
    2. Triplet contrastive loss (anchor, positive, negative)

    This encourages the model to not only classify correctly but also
    separate positive and negative examples by a margin.
    """

    def __init__(
        self,
        margin: float = 0.2,
        contrastive_weight: float = 0.3,
        **kwargs,
    ):
        """
        Initialize contrastive cross-encoder.

        Args:
            margin: Margin for triplet loss
            contrastive_weight: Weight of contrastive loss vs BCE
            **kwargs: Arguments for BinaryCrossEncoder
        """
        super().__init__(**kwargs)

        self.margin = margin
        self.contrastive_weight = contrastive_weight
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)

        logger.info(f"Initialized ContrastiveCrossEncoder with margin={margin}, "
                    f"contrastive_weight={contrastive_weight}")

    def forward_contrastive(
        self,
        input_ids_anchor: torch.Tensor,
        attention_mask_anchor: torch.Tensor,
        input_ids_positive: torch.Tensor,
        attention_mask_positive: torch.Tensor,
        input_ids_negative: torch.Tensor,
        attention_mask_negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for triplets.

        Args:
            input_ids_anchor: Anchor (mention + correct CUI) token IDs
            attention_mask_anchor: Anchor attention mask
            input_ids_positive: Positive (same mention + correct CUI) token IDs
            attention_mask_positive: Positive attention mask
            input_ids_negative: Negative (mention + wrong CUI) token IDs
            attention_mask_negative: Negative attention mask

        Returns:
            Triplet contrastive loss
        """
        # Get embeddings
        anchor_embed = self._get_cls_embedding(input_ids_anchor, attention_mask_anchor)
        positive_embed = self._get_cls_embedding(input_ids_positive, attention_mask_positive)
        negative_embed = self._get_cls_embedding(input_ids_negative, attention_mask_negative)

        # Compute triplet loss
        loss = self.triplet_loss(anchor_embed, positive_embed, negative_embed)

        return loss

    def _get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract [CLS] token embedding."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 0, :]

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        contrastive_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined BCE + contrastive loss.

        Args:
            logits: Raw logits
            labels: Binary labels
            weights: Sample weights
            contrastive_loss: Pre-computed contrastive loss (optional)

        Returns:
            Combined loss
        """
        # BCE loss
        bce_loss = super().compute_loss(logits, labels, weights)

        # Add contrastive component if provided
        if contrastive_loss is not None:
            total_loss = (1 - self.contrastive_weight) * bce_loss + \
                         self.contrastive_weight * contrastive_loss
            return total_loss

        return bce_loss
