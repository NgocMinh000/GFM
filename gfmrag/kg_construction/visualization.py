"""
================================================================================
FILE: visualization.py - Visualization for Stage 2 Entity Resolution Metrics
================================================================================

DESCRIPTION:
This module provides comprehensive visualizations for Stage 2 entity resolution
metrics. Generates publication-quality charts to analyze:
- OpenIE extraction quality
- Entity linking performance
- Cluster analysis
- Similarity score distributions
- Graph structure
- Processing efficiency

REFERENCES:
Academic papers on entity resolution visualization and evaluation:
- "How to Evaluate Entity Resolution Systems" (2024)
- "Network metrics for assessing the quality of entity resolution" (2020)
- ColBERT and DPR semantic similarity benchmarks

USAGE:
    # Auto-generate all visualizations from metrics
    visualize_entity_resolution(output_dir="tmp/kg_construction/dataset_name")

    # Or use the visualizer directly
    viz = EntityResolutionVisualizer(output_dir)
    viz.generate_all_plots()
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning(
        "Matplotlib and/or Seaborn not installed. "
        "Visualizations will not be generated. "
        "Install with: pip install matplotlib seaborn"
    )


class EntityResolutionVisualizer:
    """
    Visualizer for entity resolution metrics.

    Creates publication-quality charts for analyzing entity resolution
    performance, cluster quality, and graph structure.
    """

    def __init__(self, output_dir: str | Path):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory containing metrics JSON files
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Visualization libraries not available. "
                "Install with: pip install matplotlib seaborn"
            )

        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load metrics
        metrics_path = self.output_dir / "entity_resolution_metrics.json"
        details_path = self.output_dir / "entity_resolution_details.json"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}\n"
                "Please run Stage 2 with metrics collection first."
            )

        with open(metrics_path) as f:
            self.metrics = json.load(f)

        # Load details if available
        self.details = {}
        if details_path.exists():
            with open(details_path) as f:
                self.details = json.load(f)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10

    def plot_openie_quality(self):
        """
        Plot OpenIE extraction quality metrics.

        Creates a bar chart showing:
        - Clean triples
        - Incorrectly formatted triples
        - Triples without NER entities
        """
        openie = self.metrics["openie_extraction"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Triple counts
        categories = [
            "Clean\nTriples",
            "Incorrectly\nFormatted",
            "Missing\nNER Entities"
        ]
        counts = [
            openie["clean_triples"],
            openie["incorrectly_formatted_triples"],
            openie["triples_without_ner_entities"]
        ]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        bars = ax1.bar(categories, counts, color=colors, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("Number of Triples", fontsize=12, fontweight="bold")
        ax1.set_title("OpenIE Extraction Quality", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

        # Error rates
        if "formatting_error_rate" in openie:
            error_categories = ["Formatting\nErrors", "NER\nErrors"]
            error_rates = [
                openie.get("formatting_error_rate", 0) * 100,
                openie.get("ner_error_rate", 0) * 100
            ]

            bars2 = ax2.bar(
                error_categories, error_rates,
                color=["#e74c3c", "#f39c12"],
                edgecolor="black", linewidth=1.5
            )
            ax2.set_ylabel("Error Rate (%)", fontsize=12, fontweight="bold")
            ax2.set_title("OpenIE Error Rates", fontsize=14, fontweight="bold")
            ax2.set_ylim([0, max(error_rates) * 1.2 if error_rates else 1])
            ax2.grid(axis="y", alpha=0.3)

            for bar, rate in zip(bars2, error_rates):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )

        plt.tight_layout()
        plt.savefig(self.viz_dir / "01_openie_quality.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated OpenIE quality plot")

    def plot_similarity_distribution(self):
        """
        Plot similarity score distribution.

        Creates a histogram with confidence tier annotations.
        """
        if "similarity_scores" not in self.details or not self.details["similarity_scores"]:
            logger.warning("No similarity scores available for plotting")
            return

        scores = self.details["similarity_scores"]
        score_dist = self.metrics["score_distribution"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(
            scores, bins=50, color="#3498db", alpha=0.7,
            edgecolor="black", linewidth=0.5
        )
        ax1.axvline(0.9, color="#2ecc71", linestyle="--", linewidth=2, label="High Conf (≥0.9)")
        ax1.axvline(0.8, color="#f39c12", linestyle="--", linewidth=2, label="Medium Conf (≥0.8)")
        ax1.set_xlabel("Similarity Score", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Frequency", fontsize=12, fontweight="bold")
        ax1.set_title("Similarity Score Distribution", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        # Confidence pie chart
        labels = ["High\n(≥0.9)", "Medium\n(0.8-0.9)", "Low\n(<0.8)"]
        sizes = [
            score_dist["high_confidence"],
            score_dist["medium_confidence"],
            score_dist["low_confidence"]
        ]
        colors_pie = ["#2ecc71", "#f39c12", "#e74c3c"]
        explode = (0.05, 0, 0)

        ax2.pie(
            sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=explode,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
        )
        ax2.set_title("Confidence Tier Distribution", fontsize=14, fontweight="bold")

        plt.tight_layout()
        plt.savefig(self.viz_dir / "02_similarity_distribution.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated similarity distribution plot")

    def plot_cluster_analysis(self):
        """
        Plot cluster analysis metrics.

        Creates visualizations for:
        - Cluster size distribution
        - Coverage metrics
        """
        cluster = self.metrics["cluster_analysis"]
        coverage = self.metrics["coverage"]

        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        # Cluster size distribution
        if "cluster_size_distribution" in cluster and cluster["cluster_size_distribution"]:
            sizes = sorted(cluster["cluster_size_distribution"].items())
            x_vals = [s[0] for s in sizes if s[0] <= 20]  # Show sizes up to 20
            y_vals = [s[1] for s in sizes if s[0] <= 20]

            bars = ax1.bar(
                x_vals, y_vals, color="#3498db",
                edgecolor="black", linewidth=1
            )
            ax1.set_xlabel("Cluster Size", fontsize=12, fontweight="bold")
            ax1.set_ylabel("Number of Clusters", fontsize=12, fontweight="bold")
            ax1.set_title("Cluster Size Distribution", fontsize=14, fontweight="bold")
            ax1.grid(axis="y", alpha=0.3)

            # Highlight singletons
            if x_vals and x_vals[0] == 1:
                bars[0].set_color("#e74c3c")

        # Cluster statistics
        stats_labels = [
            f"Total\nClusters\n{cluster['num_clusters']}",
            f"Avg Size\n{cluster['avg_cluster_size']:.1f}",
            f"Max Size\n{cluster['max_cluster_size']}",
            f"Singletons\n{cluster['num_singletons']}"
        ]
        stats_values = [
            cluster['num_clusters'],
            cluster['avg_cluster_size'],
            cluster['max_cluster_size'],
            cluster['num_singletons']
        ]
        stats_colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]

        bars2 = ax2.bar(
            range(len(stats_labels)), stats_values,
            color=stats_colors, edgecolor="black", linewidth=1.5
        )
        ax2.set_xticks(range(len(stats_labels)))
        ax2.set_xticklabels(stats_labels, fontsize=9)
        ax2.set_ylabel("Count / Value", fontsize=12, fontweight="bold")
        ax2.set_title("Cluster Statistics", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # Coverage pie chart
        coverage_labels = ["With\nSynonyms", "Without\nSynonyms"]
        coverage_values = [
            coverage["entities_with_synonyms"],
            coverage["entities_without_synonyms"]
        ]
        coverage_colors = ["#2ecc71", "#95a5a6"]

        ax3.pie(
            coverage_values, labels=coverage_labels,
            colors=coverage_colors, autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
        )
        ax3.set_title(
            f"Entity Coverage\n({coverage['coverage_percentage']:.1%} with synonyms)",
            fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        plt.savefig(self.viz_dir / "03_cluster_analysis.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated cluster analysis plot")

    def plot_graph_structure(self):
        """
        Plot graph structure metrics.

        Shows edge distribution and relation types.
        """
        graph = self.metrics["graph_structure"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Edge type distribution
        edge_labels = ["Synonymy\nEdges", "Relation\nEdges"]
        edge_counts = [graph["synonymy_edges"], graph["relation_edges"]]
        edge_colors = ["#3498db", "#2ecc71"]

        bars = ax1.bar(
            edge_labels, edge_counts, color=edge_colors,
            edgecolor="black", linewidth=1.5
        )
        ax1.set_ylabel("Number of Edges", fontsize=12, fontweight="bold")
        ax1.set_title("Graph Edge Distribution", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        for bar, count in zip(bars, edge_counts):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=11, fontweight='bold'
            )

        # Graph statistics
        stats_data = [
            ("Total Edges", graph["total_edges"], "#3498db"),
            ("Synonymy Edges", graph["synonymy_edges"], "#2ecc71"),
            ("Relation Edges", graph["relation_edges"], "#9b59b6"),
            ("Unique Relations", graph["unique_relations"], "#f39c12")
        ]

        y_pos = np.arange(len(stats_data))
        values = [d[1] for d in stats_data]
        colors = [d[2] for d in stats_data]
        labels = [d[0] for d in stats_data]

        bars2 = ax2.barh(y_pos, values, color=colors, edgecolor="black", linewidth=1.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontsize=11)
        ax2.set_xlabel("Count", fontsize=12, fontweight="bold")
        ax2.set_title("Graph Structure Summary", fontsize=14, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for i, (bar, value) in enumerate(zip(bars2, values)):
            width = bar.get_width()
            ax2.text(
                width, i,
                f'  {int(value):,}',
                ha='left', va='center', fontsize=10, fontweight='bold'
            )

        plt.tight_layout()
        plt.savefig(self.viz_dir / "04_graph_structure.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated graph structure plot")

    def plot_efficiency_metrics(self):
        """
        Plot processing efficiency metrics.

        Shows timing breakdown and throughput.
        """
        efficiency = self.metrics["efficiency"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Timing breakdown
        stages = ["OpenIE\nExtraction", "Graph\nCreation", "Entity\nLinking"]
        times = [
            efficiency["openie_time"],
            efficiency["graph_creation_time"],
            efficiency["entity_linking_time"]
        ]
        colors_time = ["#3498db", "#2ecc71", "#9b59b6"]

        bars = ax1.bar(stages, times, color=colors_time, edgecolor="black", linewidth=1.5)
        ax1.set_ylabel("Time (seconds)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Processing Time Breakdown\n(Total: {efficiency['total_time']:.2f}s)",
            fontsize=14, fontweight="bold"
        )
        ax1.grid(axis="y", alpha=0.3)

        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{time_val:.2f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

        # Throughput metrics
        if efficiency["entities_per_second"] > 0:
            throughput_labels = ["Entities/sec", "Triples/sec"]
            throughput_values = [
                efficiency["entities_per_second"],
                efficiency["triples_per_second"]
            ]
            throughput_colors = ["#e74c3c", "#f39c12"]

            bars2 = ax2.bar(
                throughput_labels, throughput_values,
                color=throughput_colors, edgecolor="black", linewidth=1.5
            )
            ax2.set_ylabel("Throughput (items/second)", fontsize=12, fontweight="bold")
            ax2.set_title("Processing Throughput", fontsize=14, fontweight="bold")
            ax2.grid(axis="y", alpha=0.3)

            for bar, value in zip(bars2, throughput_values):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2, height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold'
                )

        plt.tight_layout()
        plt.savefig(self.viz_dir / "05_efficiency_metrics.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated efficiency metrics plot")

    def plot_entity_statistics(self):
        """
        Plot entity and linking statistics.
        """
        entity = self.metrics["entity_statistics"]
        linking = self.metrics["entity_linking"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Entity statistics
        entity_labels = ["Total\nPhrases", "Unique\nPhrases", "Total\nEntities", "Unique\nEntities"]
        entity_values = [
            entity["total_phrases"],
            entity["unique_phrases"],
            entity["total_entities"],
            entity["unique_entities"]
        ]
        entity_colors = ["#3498db", "#2ecc71", "#9b59b6", "#e74c3c"]

        bars = ax1.bar(
            entity_labels, entity_values, color=entity_colors,
            edgecolor="black", linewidth=1.5
        )
        ax1.set_ylabel("Count", fontsize=12, fontweight="bold")
        ax1.set_title("Entity Statistics", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        for bar, value in zip(bars, entity_values):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2, height,
                f'{int(value):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )

        # Linking metrics
        if linking["synonym_pairs_found"] > 0:
            linking_data = [
                ("Entities\nIndexed", linking["entities_indexed"], "#3498db"),
                ("Synonym\nPairs", linking["synonym_pairs_found"], "#2ecc71"),
            ]

            y_pos = np.arange(len(linking_data))
            values = [d[1] for d in linking_data]
            colors = [d[2] for d in linking_data]
            labels = [d[0] for d in linking_data]

            bars2 = ax2.barh(y_pos, values, color=colors, edgecolor="black", linewidth=1.5)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=11)
            ax2.set_xlabel("Count", fontsize=12, fontweight="bold")
            ax2.set_title(
                f"Entity Linking Results\n(Avg Score: {linking.get('avg_similarity_score', 0):.3f})",
                fontsize=14, fontweight="bold"
            )
            ax2.grid(axis="x", alpha=0.3)

            for i, (bar, value) in enumerate(zip(bars2, values)):
                width = bar.get_width()
                ax2.text(
                    width, i,
                    f'  {int(value):,}',
                    ha='left', va='center', fontsize=10, fontweight='bold'
                )

        plt.tight_layout()
        plt.savefig(self.viz_dir / "06_entity_statistics.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated entity statistics plot")

    def plot_quality_dashboard(self):
        """
        Create a comprehensive quality dashboard.

        Combines key metrics into a single overview plot.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        openie = self.metrics["openie_extraction"]
        linking = self.metrics["entity_linking"]
        cluster = self.metrics["cluster_analysis"]
        coverage = self.metrics["coverage"]
        score_dist = self.metrics["score_distribution"]
        efficiency = self.metrics["efficiency"]

        # 1. OpenIE Quality (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if "clean_triple_rate" in openie:
            quality_score = openie["clean_triple_rate"] * 100
            ax1.barh([0], [quality_score], color="#2ecc71", edgecolor="black", linewidth=2)
            ax1.set_xlim([0, 100])
            ax1.set_yticks([])
            ax1.set_xlabel("Quality Score (%)", fontsize=10, fontweight="bold")
            ax1.set_title("OpenIE Quality", fontsize=12, fontweight="bold")
            ax1.text(quality_score / 2, 0, f'{quality_score:.1f}%', ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white')
            ax1.grid(axis="x", alpha=0.3)

        # 2. Similarity Scores (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if linking.get("avg_similarity_score", 0) > 0:
            scores_data = [
                linking["avg_similarity_score"],
                linking.get("median_similarity_score", 0),
                linking.get("max_similarity_score", 0)
            ]
            labels = ["Avg", "Median", "Max"]
            bars = ax2.bar(labels, scores_data, color=["#3498db", "#2ecc71", "#9b59b6"],
                          edgecolor="black", linewidth=1.5)
            ax2.set_ylabel("Similarity Score", fontsize=10, fontweight="bold")
            ax2.set_title("Similarity Scores", fontsize=12, fontweight="bold")
            ax2.set_ylim([0, 1])
            ax2.grid(axis="y", alpha=0.3)

            for bar, val in zip(bars, scores_data):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        # 3. Coverage (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        coverage_pct = coverage["coverage_percentage"] * 100
        colors_cov = ["#2ecc71" if coverage_pct >= 50 else "#f39c12"]
        ax3.barh([0], [coverage_pct], color=colors_cov, edgecolor="black", linewidth=2)
        ax3.set_xlim([0, 100])
        ax3.set_yticks([])
        ax3.set_xlabel("Coverage (%)", fontsize=10, fontweight="bold")
        ax3.set_title("Entity Coverage", fontsize=12, fontweight="bold")
        ax3.text(coverage_pct / 2, 0, f'{coverage_pct:.1f}%', ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')
        ax3.grid(axis="x", alpha=0.3)

        # 4. Confidence Distribution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        conf_labels = ["High", "Medium", "Low"]
        conf_values = [
            score_dist["high_confidence_pct"] * 100,
            score_dist["medium_confidence_pct"] * 100,
            score_dist["low_confidence_pct"] * 100
        ]
        conf_colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        bars = ax4.bar(conf_labels, conf_values, color=conf_colors,
                      edgecolor="black", linewidth=1.5)
        ax4.set_ylabel("Percentage (%)", fontsize=10, fontweight="bold")
        ax4.set_title("Confidence Distribution", fontsize=12, fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, conf_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # 5. Cluster Stats (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        cluster_labels = ["Clusters", "Avg Size", "Singletons"]
        cluster_values = [
            cluster["num_clusters"],
            cluster["avg_cluster_size"],
            cluster["num_singletons"]
        ]
        bars = ax5.bar(cluster_labels, cluster_values,
                      color=["#3498db", "#2ecc71", "#e74c3c"],
                      edgecolor="black", linewidth=1.5)
        ax5.set_ylabel("Count / Value", fontsize=10, fontweight="bold")
        ax5.set_title("Cluster Metrics", fontsize=12, fontweight="bold")
        ax5.grid(axis="y", alpha=0.3")

        for bar, val in zip(bars, cluster_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{val:.1f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # 6. Processing Time (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        time_labels = ["OpenIE", "Graph", "Linking"]
        time_values = [
            efficiency["openie_time"],
            efficiency["graph_creation_time"],
            efficiency["entity_linking_time"]
        ]
        bars = ax6.bar(time_labels, time_values,
                      color=["#3498db", "#2ecc71", "#9b59b6"],
                      edgecolor="black", linewidth=1.5)
        ax6.set_ylabel("Time (seconds)", fontsize=10, fontweight="bold")
        ax6.set_title("Processing Time", fontsize=12, fontweight="bold")
        ax6.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, time_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{val:.1f}s', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        # 7. Summary text (bottom, spanning all columns)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        summary_text = f"""
        ╔══════════════════════════════════════════════════════════════════════════════╗
        ║                    STAGE 2: ENTITY RESOLUTION SUMMARY                        ║
        ╠══════════════════════════════════════════════════════════════════════════════╣
        ║  Passages Processed: {openie['total_passages']:,}
        ║  Clean Triples: {openie['clean_triples']:,} ({openie.get('clean_triple_rate', 0) * 100:.1f}%)
        ║  Unique Entities: {self.metrics['entity_statistics']['unique_entities']:,}
        ║  Synonym Pairs: {linking['synonym_pairs_found']:,}
        ║  Avg Similarity: {linking.get('avg_similarity_score', 0):.3f}
        ║  Entity Coverage: {coverage['coverage_percentage'] * 100:.1f}%
        ║  High Confidence: {score_dist['high_confidence_pct'] * 100:.1f}%
        ║  Total Time: {efficiency['total_time']:.2f}s
        ╚══════════════════════════════════════════════════════════════════════════════╝
        """

        ax7.text(0.5, 0.5, summary_text, ha='center', va='center',
                fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle("Entity Resolution Quality Dashboard", fontsize=16, fontweight="bold", y=0.98)
        plt.savefig(self.viz_dir / "00_quality_dashboard.png", bbox_inches="tight")
        plt.close()
        logger.info("✓ Generated quality dashboard")

    def generate_all_plots(self):
        """Generate all visualization plots."""
        logger.info("Generating entity resolution visualizations...")

        try:
            self.plot_quality_dashboard()
            self.plot_openie_quality()
            self.plot_similarity_distribution()
            self.plot_cluster_analysis()
            self.plot_graph_structure()
            self.plot_efficiency_metrics()
            self.plot_entity_statistics()

            logger.info(f"\n{'=' * 80}")
            logger.info("✓ All visualizations generated successfully!")
            logger.info(f"  Location: {self.viz_dir}")
            logger.info(f"{'=' * 80}\n")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise


def visualize_entity_resolution(output_dir: str | Path):
    """
    Convenience function to generate all entity resolution visualizations.

    Args:
        output_dir: Directory containing entity_resolution_metrics.json

    Example:
        visualize_entity_resolution("tmp/kg_construction/my_dataset")
    """
    if not VISUALIZATION_AVAILABLE:
        logger.warning(
            "Visualization libraries not available. Skipping visualization generation."
        )
        return

    try:
        visualizer = EntityResolutionVisualizer(output_dir)
        visualizer.generate_all_plots()
    except FileNotFoundError as e:
        logger.error(f"Cannot generate visualizations: {e}")
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise
