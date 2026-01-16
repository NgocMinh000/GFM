"""
MedMentions Dataset Loader for Cross-Encoder Training

Loads and processes the MedMentions corpus for UMLS entity linking training.

Dataset: MedMentions Full (Mohan & Li, 2019)
- 4,000 PubMed abstracts
- 352,496 entity mentions
- Linked to UMLS 2017AA
- URL: https://github.com/chanzuckerberg/MedMentions

Format:
PMID | Start | End | Mention Text | Semantic Type | CUI

Example:
25763772 | 0 | 20 | Autosomal recessive | T047 | C0000744
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MedMentionsLoader:
    """
    Loader for MedMentions dataset.

    Handles:
    - Parsing MedMentions annotations
    - UMLS CUI version mapping (2017AA → 2020AB)
    - Train/val/test splitting
    - Stratified sampling by entity type and CUI frequency
    """

    def __init__(
        self,
        data_path: str = "data/MedMentions/full",
        umls_version_source: str = "2017AA",
        umls_version_target: str = "2020AB",
        cui_mapping_file: Optional[str] = None,
    ):
        """
        Initialize MedMentions loader.

        Args:
            data_path: Path to MedMentions data directory
            umls_version_source: Source UMLS version (MedMentions uses 2017AA)
            umls_version_target: Target UMLS version (current system uses 2020AB)
            cui_mapping_file: Optional CUI mapping file path
        """
        self.data_path = Path(data_path)
        self.umls_version_source = umls_version_source
        self.umls_version_target = umls_version_target
        self.cui_mapping_file = cui_mapping_file

        # CUI mapping (if version mismatch)
        self.cui_mapping: Dict[str, str] = {}
        if cui_mapping_file and Path(cui_mapping_file).exists():
            self._load_cui_mapping()

        # Dataset storage
        self.mentions: List[Dict] = []  # All mentions
        self.cui_to_mentions: Dict[str, List[Dict]] = defaultdict(list)
        self.type_to_mentions: Dict[str, List[Dict]] = defaultdict(list)

    def _load_cui_mapping(self):
        """Load CUI mapping from source to target UMLS version."""
        logger.info(
            f"Loading CUI mapping: {self.umls_version_source} → {self.umls_version_target}"
        )

        # Format: source_cui \t target_cui
        cui_map_path = Path(self.cui_mapping_file)
        if not cui_map_path.exists():
            logger.warning(f"CUI mapping file not found: {cui_map_path}")
            return

        with open(cui_map_path) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    source_cui, target_cui = parts
                    self.cui_mapping[source_cui] = target_cui

        logger.info(f"Loaded {len(self.cui_mapping)} CUI mappings")

    def map_cui(self, source_cui: str) -> str:
        """
        Map CUI from source to target version.

        Args:
            source_cui: CUI in source UMLS version

        Returns:
            Mapped CUI in target version (or original if no mapping exists)
        """
        if not self.cui_mapping:
            return source_cui

        return self.cui_mapping.get(source_cui, source_cui)

    def load_corpus(self, corpus_file: str = "corpus_pubtator.txt") -> Dict[str, str]:
        """
        Load PubMed abstracts from corpus file.

        Format (PubTator):
        PMID|t|Title text
        PMID|a|Abstract text

        Args:
            corpus_file: Corpus file name

        Returns:
            Dictionary mapping PMID -> abstract text
        """
        corpus_path = self.data_path / corpus_file
        if not corpus_path.exists():
            logger.error(f"Corpus file not found: {corpus_path}")
            return {}

        logger.info(f"Loading corpus from {corpus_path}")

        abstracts = {}
        current_pmid = None
        title = ""
        abstract = ""

        with open(corpus_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                if not line:  # Empty line = document separator
                    if current_pmid:
                        abstracts[current_pmid] = f"{title} {abstract}".strip()
                        current_pmid = None
                        title = ""
                        abstract = ""
                    continue

                parts = line.split('|')
                if len(parts) < 3:
                    continue

                pmid, section_type, text = parts[0], parts[1], '|'.join(parts[2:])

                if section_type == 't':  # Title
                    current_pmid = pmid
                    title = text
                elif section_type == 'a':  # Abstract
                    abstract = text

        # Don't forget last document
        if current_pmid:
            abstracts[current_pmid] = f"{title} {abstract}".strip()

        logger.info(f"Loaded {len(abstracts)} abstracts")
        return abstracts

    def load_annotations(
        self,
        annotation_file: str = "corpus_pubtator_pmids_trng.txt"
    ) -> List[Dict]:
        """
        Load entity mentions from annotation file.

        Format:
        PMID \t Start \t End \t Mention Text \t Semantic Type \t CUI

        Args:
            annotation_file: Annotation file name

        Returns:
            List of mention dictionaries
        """
        annotation_path = self.data_path / annotation_file
        if not annotation_path.exists():
            logger.error(f"Annotation file not found: {annotation_path}")
            return []

        logger.info(f"Loading annotations from {annotation_path}")

        mentions = []
        with open(annotation_path, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 6:
                    logger.warning(f"Line {line_num}: Invalid format (expected 6 fields)")
                    continue

                pmid, start, end, mention_text, semantic_type, source_cui = parts[:6]

                # Map CUI to target version
                target_cui = self.map_cui(source_cui)

                mention = {
                    "pmid": pmid,
                    "start": int(start),
                    "end": int(end),
                    "mention_text": mention_text,
                    "semantic_type": semantic_type,
                    "cui": target_cui,
                    "source_cui": source_cui,
                }

                mentions.append(mention)

                # Index by CUI and type
                self.cui_to_mentions[target_cui].append(mention)
                self.type_to_mentions[semantic_type].append(mention)

        logger.info(f"Loaded {len(mentions)} entity mentions")
        logger.info(f"Unique CUIs: {len(self.cui_to_mentions)}")
        logger.info(f"Unique semantic types: {len(self.type_to_mentions)}")

        self.mentions = mentions
        return mentions

    def compute_cui_frequencies(self) -> Dict[str, int]:
        """
        Compute CUI frequencies for stratification.

        Returns:
            Dictionary mapping CUI -> mention count
        """
        return {
            cui: len(mentions)
            for cui, mentions in self.cui_to_mentions.items()
        }

    def get_difficulty_labels(self, mentions: List[Dict]) -> List[str]:
        """
        Assign difficulty labels for stratification.

        Difficulty based on CUI frequency:
        - Easy: Very common CUIs (top 20%)
        - Medium: Moderately common CUIs (middle 60%)
        - Hard: Rare CUIs (bottom 20%)

        Args:
            mentions: List of mentions

        Returns:
            List of difficulty labels (same length as mentions)
        """
        cui_freq = self.compute_cui_frequencies()
        freq_values = sorted(cui_freq.values(), reverse=True)

        # Compute percentile thresholds
        p80 = np.percentile(freq_values, 80)  # Top 20%
        p20 = np.percentile(freq_values, 20)  # Bottom 20%

        difficulty_labels = []
        for mention in mentions:
            cui = mention["cui"]
            freq = cui_freq[cui]

            if freq >= p80:
                difficulty_labels.append("easy")
            elif freq >= p20:
                difficulty_labels.append("medium")
            else:
                difficulty_labels.append("hard")

        return difficulty_labels

    def split_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/val/test sets.

        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify: Whether to use stratified sampling
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_mentions, val_mentions, test_mentions)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        if not self.mentions:
            logger.error("No mentions loaded. Call load_annotations() first.")
            return [], [], []

        mentions = self.mentions.copy()

        # Prepare stratification labels
        if stratify:
            # Combine semantic type + difficulty for stratification
            type_labels = [m["semantic_type"] for m in mentions]
            difficulty_labels = self.get_difficulty_labels(mentions)
            stratify_labels = [
                f"{t}_{d}" for t, d in zip(type_labels, difficulty_labels)
            ]
        else:
            stratify_labels = None

        # First split: train vs (val + test)
        train_mentions, temp_mentions = train_test_split(
            mentions,
            test_size=(val_ratio + test_ratio),
            stratify=stratify_labels if stratify else None,
            random_state=random_state,
        )

        # Second split: val vs test
        if stratify:
            # Re-compute stratify labels for temp set
            temp_type_labels = [m["semantic_type"] for m in temp_mentions]
            temp_difficulty_labels = self.get_difficulty_labels(temp_mentions)
            temp_stratify_labels = [
                f"{t}_{d}" for t, d in zip(temp_type_labels, temp_difficulty_labels)
            ]
        else:
            temp_stratify_labels = None

        val_mentions, test_mentions = train_test_split(
            temp_mentions,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_stratify_labels if stratify else None,
            random_state=random_state,
        )

        logger.info(f"Dataset split complete:")
        logger.info(f"  Train: {len(train_mentions)} mentions ({len(train_mentions)/len(mentions):.1%})")
        logger.info(f"  Val:   {len(val_mentions)} mentions ({len(val_mentions)/len(mentions):.1%})")
        logger.info(f"  Test:  {len(test_mentions)} mentions ({len(test_mentions)/len(mentions):.1%})")

        return train_mentions, val_mentions, test_mentions

    def save_split(
        self,
        train_mentions: List[Dict],
        val_mentions: List[Dict],
        test_mentions: List[Dict],
        output_dir: str = "tmp/training/data_splits",
    ):
        """
        Save train/val/test splits to disk.

        Args:
            train_mentions: Training mentions
            val_mentions: Validation mentions
            test_mentions: Test mentions
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save as pickle (faster loading)
        with open(output_path / "train.pkl", "wb") as f:
            pickle.dump(train_mentions, f)

        with open(output_path / "val.pkl", "wb") as f:
            pickle.dump(val_mentions, f)

        with open(output_path / "test.pkl", "wb") as f:
            pickle.dump(test_mentions, f)

        # Also save as JSON (human-readable)
        with open(output_path / "train.json", "w") as f:
            json.dump(train_mentions, f, indent=2)

        with open(output_path / "val.json", "w") as f:
            json.dump(val_mentions, f, indent=2)

        with open(output_path / "test.json", "w") as f:
            json.dump(test_mentions, f, indent=2)

        logger.info(f"Splits saved to {output_path}")

    def load_split(
        self,
        split_dir: str = "tmp/training/data_splits",
        split_name: str = "train",
    ) -> List[Dict]:
        """
        Load a saved split from disk.

        Args:
            split_dir: Directory containing splits
            split_name: Split name ("train", "val", or "test")

        Returns:
            List of mentions
        """
        split_path = Path(split_dir) / f"{split_name}.pkl"

        if not split_path.exists():
            logger.error(f"Split file not found: {split_path}")
            return []

        with open(split_path, "rb") as f:
            mentions = pickle.load(f)

        logger.info(f"Loaded {len(mentions)} mentions from {split_path}")
        return mentions

    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        if not self.mentions:
            return {}

        cui_freq = self.compute_cui_frequencies()
        type_dist = {
            sem_type: len(mentions)
            for sem_type, mentions in self.type_to_mentions.items()
        }

        return {
            "total_mentions": len(self.mentions),
            "unique_cuis": len(self.cui_to_mentions),
            "unique_semantic_types": len(self.type_to_mentions),
            "semantic_type_distribution": type_dist,
            "avg_mentions_per_cui": np.mean(list(cui_freq.values())),
            "median_mentions_per_cui": np.median(list(cui_freq.values())),
            "max_mentions_per_cui": max(cui_freq.values()) if cui_freq else 0,
            "min_mentions_per_cui": min(cui_freq.values()) if cui_freq else 0,
        }

    def print_statistics(self):
        """Print dataset statistics to console."""
        stats = self.get_statistics()

        print("=" * 80)
        print("MEDMENTIONS DATASET STATISTICS")
        print("=" * 80)
        print(f"Total Mentions:          {stats['total_mentions']:,}")
        print(f"Unique CUIs:             {stats['unique_cuis']:,}")
        print(f"Unique Semantic Types:   {stats['unique_semantic_types']}")
        print(f"Avg Mentions per CUI:    {stats['avg_mentions_per_cui']:.2f}")
        print(f"Median Mentions per CUI: {stats['median_mentions_per_cui']:.2f}")
        print(f"Max Mentions per CUI:    {stats['max_mentions_per_cui']:,}")
        print(f"Min Mentions per CUI:    {stats['min_mentions_per_cui']:,}")
        print("\nTop 10 Semantic Types:")
        print("-" * 80)

        sorted_types = sorted(
            stats['semantic_type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for sem_type, count in sorted_types[:10]:
            pct = count / stats['total_mentions'] * 100
            print(f"  {sem_type:40} {count:8,} ({pct:5.2f}%)")

        print("=" * 80)


# Utility function for quick loading
def load_medmentions(
    data_path: str = "data/MedMentions/full",
    annotation_file: str = "corpus_pubtator_pmids_trng.txt",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    cache_dir: str = "tmp/training/data_splits",
    force_reload: bool = False,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convenience function to load MedMentions dataset.

    Args:
        data_path: Path to MedMentions data
        annotation_file: Annotation file name
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        cache_dir: Cache directory for splits
        force_reload: Force reload from source (ignore cache)

    Returns:
        Tuple of (train_mentions, val_mentions, test_mentions)
    """
    cache_path = Path(cache_dir)

    # Check if cached splits exist
    if not force_reload and cache_path.exists():
        train_pkl = cache_path / "train.pkl"
        val_pkl = cache_path / "val.pkl"
        test_pkl = cache_path / "test.pkl"

        if train_pkl.exists() and val_pkl.exists() and test_pkl.exists():
            logger.info("Loading cached dataset splits")
            loader = MedMentionsLoader(data_path=data_path)
            train_mentions = loader.load_split(cache_dir, "train")
            val_mentions = loader.load_split(cache_dir, "val")
            test_mentions = loader.load_split(cache_dir, "test")
            return train_mentions, val_mentions, test_mentions

    # Load from source
    logger.info("Loading MedMentions from source")
    loader = MedMentionsLoader(data_path=data_path)
    loader.load_annotations(annotation_file)
    loader.print_statistics()

    # Split dataset
    # Note: stratify=False to avoid errors with rare classes (single sample per class)
    train_mentions, val_mentions, test_mentions = loader.split_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify=False,  # Disabled: some CUIs have only 1 mention
    )

    # Save splits
    loader.save_split(train_mentions, val_mentions, test_mentions, cache_dir)

    return train_mentions, val_mentions, test_mentions
