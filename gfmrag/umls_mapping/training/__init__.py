"""
UMLS Cross-Encoder Training Module

This module provides tools for fine-tuning cross-encoder models
for UMLS entity linking.

Components:
- data_loader: MedMentions/BC5CDR dataset loading
- hard_negative_miner: Generate hard negatives using FAISS
- dataset: PyTorch Dataset for training
- cross_encoder_trainer: Fine-tuning script
"""

__version__ = "1.0.0"
