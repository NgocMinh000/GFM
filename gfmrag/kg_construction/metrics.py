"""
================================================================================
FILE: metrics.py - Metrics collection for Stage 2 Entity Resolution
================================================================================

DESCRIPTION:
This module provides comprehensive metrics for evaluating the quality of
knowledge graph construction and entity resolution (Stage 2).

METRICS CATEGORIES:
1. Entity Clustering Metrics - Cluster precision, recall, F1
2. Graph Quality Metrics - Coverage, cluster distribution, connectivity
3. Similarity Score Analysis - Score distributions, confidence tiers
4. Error Analysis - Triple quality, formatting errors
5. Efficiency Metrics - Processing time, throughput

REFERENCES:
- "How to Evaluate Entity Resolution Systems" (2024 arXiv:2404.05622)
  https://arxiv.org/pdf/2404.05622
- "Unsupervised Graph-Based Entity Resolution for Complex Entities" (ACM TKDD)
  https://dl.acm.org/doi/10.1145/3533016
- "Network metrics for assessing the quality of entity resolution" (2020)
  https://www.researchgate.net/publication/347379650

USAGE:
    metrics = EntityResolutionMetrics()

    # Track OpenIE extraction
    metrics.record_openie_stats(
        total_passages=100,
        total_triples=500,
        clean_triples=450
    )

    # Track entity linking
    metrics.record_entity_linking(
        total_entities=200,
        synonym_pairs=[(ent1, ent2, 0.95), ...],
        similarity_scores=[0.95, 0.87, ...]
    )

    # Save metrics
    metrics.save_metrics(output_dir="tmp/kg_construction")
================================================================================
"""

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EntityResolutionMetrics:
    """
    Comprehensive metrics collection for knowledge graph construction
    and entity resolution evaluation.

    This class tracks metrics across all stages of KG construction:
    - OpenIE extraction quality
    - Entity linking and resolution
    - Graph structure and quality
    - Processing efficiency

    Metrics are designed based on recent academic research on entity
    resolution evaluation (2024-2025).
    """

    def __init__(self):
        """Initialize metrics collectors."""
        # OpenIE Extraction Metrics
        self.openie_metrics = {
            "total_passages": 0,
            "total_triples_extracted": 0,
            "clean_triples": 0,
            "incorrectly_formatted_triples": 0,
            "triples_without_ner_entities": 0,
            "unique_triples": 0,
            "extraction_time": 0.0,
        }

        # Entity Statistics
        self.entity_metrics = {
            "total_phrases": 0,
            "unique_phrases": 0,
            "total_entities": 0,
            "unique_entities": 0,
        }

        # Entity Linking/Resolution Metrics
        self.linking_metrics = {
            "entities_indexed": 0,
            "synonym_pairs_found": 0,
            "avg_similarity_score": 0.0,
            "median_similarity_score": 0.0,
            "min_similarity_score": 0.0,
            "max_similarity_score": 0.0,
            "linking_time": 0.0,
        }

        # Blocking Efficiency Metrics (Phase A addition)
        self.blocking_metrics = {
            "reduction_ratio": 0.0,  # RR = 1 - |C|/|P|
            "reduction_ratio_pct": "0%",
            "candidate_pairs": 0,
            "total_possible_pairs": 0,
        }

        # Cluster-based Metrics (inspired by 2024 arxiv:2404.05622)
        self.cluster_metrics = {
            "num_clusters": 0,
            "num_singletons": 0,  # Entities with no synonyms
            "avg_cluster_size": 0.0,
            "max_cluster_size": 0,
            "cluster_size_distribution": {},  # {size: count}
        }

        # Coverage Metrics
        self.coverage_metrics = {
            "entities_with_synonyms": 0,
            "entities_without_synonyms": 0,
            "coverage_percentage": 0.0,  # % entities with at least 1 synonym
        }

        # Similarity Score Distribution
        self.score_distribution = {
            "high_confidence": 0,      # score >= 0.9
            "medium_confidence": 0,     # 0.8 <= score < 0.9
            "low_confidence": 0,        # threshold <= score < 0.8
            "high_confidence_pct": 0.0,
            "medium_confidence_pct": 0.0,
            "low_confidence_pct": 0.0,
        }

        # Score Margin Metrics (Phase A addition)
        self.score_margin_metrics = {
            "mean_margin": 0.0,  # Δ(entity) = score_1 - score_2
            "median_margin": 0.0,
            "min_margin": 0.0,
            "max_margin": 0.0,
            "entities_with_margin": 0,  # Entities with >= 2 candidates
            "high_margin_count": 0,  # margin >= 0.20 (confident matches)
            "high_margin_pct": 0.0,
        }

        # Raw data for margin analysis
        self._score_margins = []  # List of margins per entity

        # Graph Structure Metrics
        self.graph_metrics = {
            "total_edges": 0,
            "synonymy_edges": 0,
            "relation_edges": 0,
            "unique_relations": 0,
        }

        # Efficiency Metrics
        self.efficiency_metrics = {
            "total_time": 0.0,
            "openie_time": 0.0,
            "graph_creation_time": 0.0,
            "entity_linking_time": 0.0,
            "entities_per_second": 0.0,
            "triples_per_second": 0.0,
        }

        # Raw data for analysis
        self._similarity_scores = []
        self._cluster_sizes = []
        self._synonym_pairs = []  # [(entity1, entity2, score)]

        # Timing
        self._start_time = None
        self._stage_start_times = {}

    def start_timer(self, stage: str = "total"):
        """Start timing for a stage."""
        if stage == "total":
            self._start_time = time.time()
        self._stage_start_times[stage] = time.time()

    def stop_timer(self, stage: str = "total") -> float:
        """Stop timing for a stage and return elapsed time."""
        if stage not in self._stage_start_times:
            logger.warning(f"Timer for stage '{stage}' was not started")
            return 0.0

        elapsed = time.time() - self._stage_start_times[stage]

        if stage == "total":
            self.efficiency_metrics["total_time"] = elapsed
        elif stage == "openie":
            self.efficiency_metrics["openie_time"] = elapsed
            self.openie_metrics["extraction_time"] = elapsed
        elif stage == "graph_creation":
            self.efficiency_metrics["graph_creation_time"] = elapsed
        elif stage == "entity_linking":
            self.efficiency_metrics["entity_linking_time"] = elapsed
            self.linking_metrics["linking_time"] = elapsed

        return elapsed

    def record_openie_stats(
        self,
        total_passages: int,
        total_triples: int,
        clean_triples: int,
        incorrectly_formatted: int,
        triples_without_ner: int,
        unique_triples: int,
    ):
        """
        Record OpenIE extraction statistics.

        Args:
            total_passages: Number of documents processed
            total_triples: Total triples extracted
            clean_triples: Number of valid triples
            incorrectly_formatted: Number of malformed triples
            triples_without_ner: Triples missing NER entities
            unique_triples: Number of unique triples
        """
        self.openie_metrics.update({
            "total_passages": total_passages,
            "total_triples_extracted": total_triples,
            "clean_triples": clean_triples,
            "incorrectly_formatted_triples": incorrectly_formatted,
            "triples_without_ner_entities": triples_without_ner,
            "unique_triples": unique_triples,
        })

        # Calculate error rates
        if total_triples > 0:
            self.openie_metrics["formatting_error_rate"] = \
                incorrectly_formatted / total_triples
            self.openie_metrics["ner_error_rate"] = \
                triples_without_ner / total_triples
            self.openie_metrics["clean_triple_rate"] = \
                clean_triples / total_triples

    def record_entity_stats(
        self,
        total_phrases: int,
        unique_phrases: int,
        total_entities: int,
        unique_entities: int,
    ):
        """Record entity statistics."""
        self.entity_metrics.update({
            "total_phrases": total_phrases,
            "unique_phrases": unique_phrases,
            "total_entities": total_entities,
            "unique_entities": unique_entities,
        })

    def record_synonym_pair(
        self,
        entity1: str,
        entity2: str,
        similarity_score: float,
    ):
        """
        Record a synonym pair found during entity resolution.

        Args:
            entity1: First entity
            entity2: Second entity (similar to entity1)
            similarity_score: Similarity score between entities
        """
        self._synonym_pairs.append((entity1, entity2, similarity_score))
        self._similarity_scores.append(similarity_score)

    def record_entity_linking(
        self,
        entities_indexed: int,
        synonym_pairs: List[Tuple[str, str, float]],
        threshold: float = 0.8,
    ):
        """
        Record entity linking results.

        Args:
            entities_indexed: Number of entities indexed
            synonym_pairs: List of (entity1, entity2, score) tuples
            threshold: Similarity threshold used
        """
        self.linking_metrics["entities_indexed"] = entities_indexed
        self.linking_metrics["synonym_pairs_found"] = len(synonym_pairs)

        # Record all pairs
        for e1, e2, score in synonym_pairs:
            self.record_synonym_pair(e1, e2, score)

        # Calculate score statistics
        if self._similarity_scores:
            scores = self._similarity_scores
            self.linking_metrics["avg_similarity_score"] = float(np.mean(scores))
            self.linking_metrics["median_similarity_score"] = float(np.median(scores))
            self.linking_metrics["min_similarity_score"] = float(np.min(scores))
            self.linking_metrics["max_similarity_score"] = float(np.max(scores))
            self.linking_metrics["std_similarity_score"] = float(np.std(scores))

            # Calculate score distribution
            high_conf = sum(1 for s in scores if s >= 0.9)
            medium_conf = sum(1 for s in scores if 0.8 <= s < 0.9)
            low_conf = sum(1 for s in scores if threshold <= s < 0.8)
            total = len(scores)

            self.score_distribution.update({
                "high_confidence": high_conf,
                "medium_confidence": medium_conf,
                "low_confidence": low_conf,
                "high_confidence_pct": high_conf / total if total > 0 else 0,
                "medium_confidence_pct": medium_conf / total if total > 0 else 0,
                "low_confidence_pct": low_conf / total if total > 0 else 0,
            })

    def compute_reduction_ratio(self, num_entities: int, candidate_pairs: int):
        """
        Compute Reduction Ratio (RR) - blocking efficiency metric.

        RR measures how much the blocking step reduced the comparison space:
        RR = 1 - |C| / |P|

        Where:
        - C = number of candidate pairs (after blocking)
        - P = total possible pairs = n(n-1)/2
        - RR close to 1 = very effective blocking (reduced many comparisons)
        - RR close to 0 = ineffective blocking (still many comparisons)

        Reference: BlockingPy documentation
        https://blockingpy.readthedocs.io/en/latest/metrics.html

        Args:
            num_entities: Total number of entities
            candidate_pairs: Number of candidate pairs after blocking
        """
        if num_entities <= 1:
            logger.warning("Cannot compute RR with <= 1 entity")
            return

        # Total possible pairs = n(n-1)/2
        total_possible_pairs = num_entities * (num_entities - 1) // 2

        # Reduction ratio
        if total_possible_pairs > 0:
            reduction_ratio = 1.0 - (candidate_pairs / total_possible_pairs)
        else:
            reduction_ratio = 0.0

        self.blocking_metrics.update({
            "reduction_ratio": reduction_ratio,
            "reduction_ratio_pct": f"{reduction_ratio * 100:.2f}%",
            "candidate_pairs": candidate_pairs,
            "total_possible_pairs": total_possible_pairs,
        })

        logger.info(
            f"Reduction Ratio: {reduction_ratio:.4f} "
            f"({candidate_pairs:,} / {total_possible_pairs:,} pairs)"
        )

    def compute_score_margins(self, entity_scores: Dict[str, List[float]]):
        """
        Compute Score Margin (Δ) for each entity.

        Score margin measures the confidence gap between top-1 and top-2 matches:
        Δ(entity) = score_1 - score_2

        High margin (e.g., Δ >= 0.20) indicates confident, unambiguous matching.
        Low margin (e.g., Δ < 0.10) indicates ambiguous cases requiring review.

        Reference: ER-Evaluation framework, BlockingPy metrics
        https://blockingpy.readthedocs.io/en/latest/metrics.html

        Args:
            entity_scores: Dictionary mapping entity -> list of candidate scores
                          (sorted descending, top-1 first)
        """
        margins = []
        high_margin_threshold = 0.20

        for entity_id, scores in entity_scores.items():
            if len(scores) >= 2:
                # Ensure sorted descending
                sorted_scores = sorted(scores, reverse=True)
                margin = sorted_scores[0] - sorted_scores[1]
                margins.append(margin)

        if not margins:
            logger.warning("No entities with >= 2 candidates for margin calculation")
            return

        self._score_margins = margins

        # Compute statistics
        mean_margin = float(np.mean(margins))
        median_margin = float(np.median(margins))
        min_margin = float(np.min(margins))
        max_margin = float(np.max(margins))
        high_margin_count = sum(1 for m in margins if m >= high_margin_threshold)

        self.score_margin_metrics.update({
            "mean_margin": mean_margin,
            "median_margin": median_margin,
            "min_margin": min_margin,
            "max_margin": max_margin,
            "entities_with_margin": len(margins),
            "high_margin_count": high_margin_count,
            "high_margin_pct": high_margin_count / len(margins) if margins else 0.0,
        })

        logger.info(
            f"Score Margin: mean={mean_margin:.3f}, median={median_margin:.3f}, "
            f"high_margin%={high_margin_count / len(margins) * 100:.1f}%"
        )

    def analyze_clusters(self, graph: Dict[Tuple[str, str], str]):
        """
        Analyze entity clusters from the graph structure.

        Uses Union-Find algorithm to identify connected components
        (entity clusters) and calculate cluster-based metrics.

        Args:
            graph: Dictionary mapping (entity1, entity2) -> relation
        """
        # Extract synonymy edges
        synonymy_edges = [
            (e1, e2) for (e1, e2), rel in graph.items()
            if rel == "equivalent"
        ]

        if not synonymy_edges:
            logger.warning("No synonymy edges found in graph")
            return

        # Build Union-Find structure to find clusters
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union all synonym pairs
        for e1, e2 in synonymy_edges:
            union(e1, e2)

        # Group entities by cluster
        clusters = defaultdict(list)
        for entity in parent:
            root = find(entity)
            clusters[root].append(entity)

        # Calculate cluster statistics
        cluster_sizes = [len(entities) for entities in clusters.values()]
        self._cluster_sizes = cluster_sizes

        num_singletons = sum(1 for size in cluster_sizes if size == 1)

        self.cluster_metrics.update({
            "num_clusters": len(clusters),
            "num_singletons": num_singletons,
            "avg_cluster_size": float(np.mean(cluster_sizes)) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "median_cluster_size": float(np.median(cluster_sizes)) if cluster_sizes else 0,
        })

        # Cluster size distribution
        size_dist = Counter(cluster_sizes)
        self.cluster_metrics["cluster_size_distribution"] = dict(size_dist)

        # Calculate coverage
        total_entities_in_graph = len(parent)
        entities_with_synonyms = sum(
            len(entities) for entities in clusters.values() if len(entities) > 1
        )

        self.coverage_metrics.update({
            "entities_with_synonyms": entities_with_synonyms,
            "entities_without_synonyms": num_singletons,
            "coverage_percentage":
                entities_with_synonyms / total_entities_in_graph
                if total_entities_in_graph > 0 else 0,
        })

    def record_graph_stats(
        self,
        total_edges: int,
        synonymy_edges: int,
        unique_relations: int,
    ):
        """Record graph structure statistics."""
        self.graph_metrics.update({
            "total_edges": total_edges,
            "synonymy_edges": synonymy_edges,
            "relation_edges": total_edges - synonymy_edges,
            "unique_relations": unique_relations,
        })

    def finalize(self):
        """Calculate final derived metrics."""
        # Calculate throughput
        total_time = self.efficiency_metrics["total_time"]
        if total_time > 0:
            self.efficiency_metrics["entities_per_second"] = \
                self.entity_metrics["unique_entities"] / total_time
            self.efficiency_metrics["triples_per_second"] = \
                self.openie_metrics["clean_triples"] / total_time

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        return {
            "openie_extraction": self.openie_metrics,
            "entity_statistics": self.entity_metrics,
            "entity_linking": self.linking_metrics,
            "blocking_efficiency": self.blocking_metrics,  # Phase A addition
            "cluster_analysis": self.cluster_metrics,
            "coverage": self.coverage_metrics,
            "score_distribution": self.score_distribution,
            "score_margin": self.score_margin_metrics,  # Phase A addition
            "graph_structure": self.graph_metrics,
            "efficiency": self.efficiency_metrics,
        }

    def save_metrics(self, output_dir: str | Path):
        """
        Save metrics to JSON file.

        Args:
            output_dir: Directory to save metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Finalize metrics
        self.finalize()

        # Save metrics JSON
        metrics_path = output_dir / "entity_resolution_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.get_all_metrics(), f, indent=2)

        logger.info(f"✓ Metrics saved to {metrics_path}")

        # Save detailed data
        details_path = output_dir / "entity_resolution_details.json"
        with open(details_path, "w") as f:
            json.dump({
                "similarity_scores": self._similarity_scores,
                "cluster_sizes": self._cluster_sizes,
                "synonym_pairs": [
                    {"entity1": e1, "entity2": e2, "score": score}
                    for e1, e2, score in self._synonym_pairs
                ],
            }, f, indent=2)

        logger.info(f"✓ Detailed data saved to {details_path}")

        # Generate text report
        self.generate_report(output_dir)

    def generate_report(self, output_dir: str | Path):
        """
        Generate human-readable metrics report.

        Args:
            output_dir: Directory to save report
        """
        output_dir = Path(output_dir)
        report_path = output_dir / "entity_resolution_report.txt"

        with open(report_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("STAGE 2: ENTITY RESOLUTION METRICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            # OpenIE Extraction
            f.write("1. OPEN INFORMATION EXTRACTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Passages Processed:        {self.openie_metrics['total_passages']}\n")
            f.write(f"Total Triples Extracted:         {self.openie_metrics['total_triples_extracted']}\n")
            f.write(f"Clean Triples:                   {self.openie_metrics['clean_triples']}\n")
            f.write(f"Unique Triples:                  {self.openie_metrics['unique_triples']}\n")
            f.write(f"Incorrectly Formatted:           {self.openie_metrics['incorrectly_formatted_triples']}\n")
            f.write(f"Missing NER Entities:            {self.openie_metrics['triples_without_ner_entities']}\n")
            if "clean_triple_rate" in self.openie_metrics:
                f.write(f"Clean Triple Rate:               {self.openie_metrics['clean_triple_rate']:.1%}\n")
            f.write(f"Extraction Time:                 {self.openie_metrics['extraction_time']:.2f}s\n")
            f.write("\n")

            # Entity Statistics
            f.write("2. ENTITY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Phrases:                   {self.entity_metrics['total_phrases']}\n")
            f.write(f"Unique Phrases:                  {self.entity_metrics['unique_phrases']}\n")
            f.write(f"Total Entities:                  {self.entity_metrics['total_entities']}\n")
            f.write(f"Unique Entities:                 {self.entity_metrics['unique_entities']}\n")
            f.write("\n")

            # Entity Linking
            f.write("3. ENTITY LINKING & RESOLUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Entities Indexed:                {self.linking_metrics['entities_indexed']}\n")
            f.write(f"Synonym Pairs Found:             {self.linking_metrics['synonym_pairs_found']}\n")
            if self._similarity_scores:
                f.write(f"Avg Similarity Score:            {self.linking_metrics['avg_similarity_score']:.3f}\n")
                f.write(f"Median Similarity Score:         {self.linking_metrics['median_similarity_score']:.3f}\n")
                f.write(f"Score Range:                     [{self.linking_metrics['min_similarity_score']:.3f}, {self.linking_metrics['max_similarity_score']:.3f}]\n")
            f.write(f"Linking Time:                    {self.linking_metrics['linking_time']:.2f}s\n")
            f.write("\n")

            # Blocking Efficiency (Phase A addition)
            f.write("4. BLOCKING EFFICIENCY (Phase A)\n")
            f.write("-" * 80 + "\n")
            if self.blocking_metrics['total_possible_pairs'] > 0:
                f.write(f"Reduction Ratio (RR):            {self.blocking_metrics['reduction_ratio']:.4f} ")
                f.write(f"({self.blocking_metrics['reduction_ratio_pct']})\n")
                f.write(f"Candidate Pairs:                 {self.blocking_metrics['candidate_pairs']:,}\n")
                f.write(f"Total Possible Pairs:            {self.blocking_metrics['total_possible_pairs']:,}\n")
                f.write(f"Interpretation:                  RR={self.blocking_metrics['reduction_ratio']:.4f} → ")
                if self.blocking_metrics['reduction_ratio'] >= 0.999:
                    f.write("Excellent blocking (>99.9% reduction)\n")
                elif self.blocking_metrics['reduction_ratio'] >= 0.99:
                    f.write("Very good blocking (>99% reduction)\n")
                elif self.blocking_metrics['reduction_ratio'] >= 0.95:
                    f.write("Good blocking (>95% reduction)\n")
                else:
                    f.write("Moderate blocking (<95% reduction)\n")
            else:
                f.write("No blocking metrics available\n")
            f.write("\n")

            # Score Margin (Phase A addition)
            f.write("5. SCORE MARGIN ANALYSIS (Phase A)\n")
            f.write("-" * 80 + "\n")
            if self._score_margins:
                f.write(f"Entities with ≥2 Candidates:     {self.score_margin_metrics['entities_with_margin']}\n")
                f.write(f"Mean Score Margin (Δ):           {self.score_margin_metrics['mean_margin']:.3f}\n")
                f.write(f"Median Score Margin:             {self.score_margin_metrics['median_margin']:.3f}\n")
                f.write(f"Margin Range:                    [{self.score_margin_metrics['min_margin']:.3f}, {self.score_margin_metrics['max_margin']:.3f}]\n")
                f.write(f"High Margin (Δ≥0.20):            {self.score_margin_metrics['high_margin_count']} ")
                f.write(f"({self.score_margin_metrics['high_margin_pct']:.1%})\n")
                f.write(f"Interpretation:                  Mean Δ={self.score_margin_metrics['mean_margin']:.3f} → ")
                if self.score_margin_metrics['mean_margin'] >= 0.20:
                    f.write("Confident matches (high margin)\n")
                elif self.score_margin_metrics['mean_margin'] >= 0.10:
                    f.write("Moderate confidence\n")
                else:
                    f.write("Ambiguous matches (low margin, needs review)\n")
            else:
                f.write("No score margin data available\n")
            f.write("\n")

            # Cluster Analysis
            f.write("6. CLUSTER ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of Clusters:              {self.cluster_metrics['num_clusters']}\n")
            f.write(f"Singletons (no synonyms):        {self.cluster_metrics['num_singletons']}\n")
            f.write(f"Avg Cluster Size:                {self.cluster_metrics['avg_cluster_size']:.2f}\n")
            f.write(f"Max Cluster Size:                {self.cluster_metrics['max_cluster_size']}\n")
            f.write(f"Median Cluster Size:             {self.cluster_metrics.get('median_cluster_size', 0):.2f}\n")
            f.write("\n")

            # Coverage
            f.write("7. COVERAGE METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Entities with Synonyms:          {self.coverage_metrics['entities_with_synonyms']}\n")
            f.write(f"Entities without Synonyms:       {self.coverage_metrics['entities_without_synonyms']}\n")
            f.write(f"Coverage:                        {self.coverage_metrics['coverage_percentage']:.1%}\n")
            f.write("\n")

            # Score Distribution
            f.write("8. SIMILARITY SCORE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"High Confidence (≥0.9):          {self.score_distribution['high_confidence']} ")
            f.write(f"({self.score_distribution['high_confidence_pct']:.1%})\n")
            f.write(f"Medium Confidence [0.8-0.9):     {self.score_distribution['medium_confidence']} ")
            f.write(f"({self.score_distribution['medium_confidence_pct']:.1%})\n")
            f.write(f"Low Confidence [thresh-0.8):     {self.score_distribution['low_confidence']} ")
            f.write(f"({self.score_distribution['low_confidence_pct']:.1%})\n")
            f.write("\n")

            # Graph Structure
            f.write("9. GRAPH STRUCTURE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Edges:                     {self.graph_metrics['total_edges']}\n")
            f.write(f"Synonymy Edges:                  {self.graph_metrics['synonymy_edges']}\n")
            f.write(f"Relation Edges:                  {self.graph_metrics['relation_edges']}\n")
            f.write(f"Unique Relations:                {self.graph_metrics['unique_relations']}\n")
            f.write("\n")

            # Efficiency
            f.write("10. EFFICIENCY METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Time:                      {self.efficiency_metrics['total_time']:.2f}s\n")
            f.write(f"  - OpenIE:                      {self.efficiency_metrics['openie_time']:.2f}s\n")
            f.write(f"  - Graph Creation:              {self.efficiency_metrics['graph_creation_time']:.2f}s\n")
            f.write(f"  - Entity Linking:              {self.efficiency_metrics['entity_linking_time']:.2f}s\n")
            if self.efficiency_metrics['entities_per_second'] > 0:
                f.write(f"Throughput (entities/s):         {self.efficiency_metrics['entities_per_second']:.2f}\n")
                f.write(f"Throughput (triples/s):          {self.efficiency_metrics['triples_per_second']:.2f}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")

        logger.info(f"✓ Report saved to {report_path}")
