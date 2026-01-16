"""
Visualization Module for Stage 2 Entity Resolution Pipeline

Generates charts and plots for analyzing entity resolution performance:
- Type distribution across tiers
- Confidence distributions
- Cluster size distributions
- Stage-by-stage metrics
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not installed. Visualization disabled.")
    logger.warning("Install with: pip install matplotlib seaborn")


def visualize_stage2_metrics(output_dir: Path):
    """
    Generate all visualization plots for Stage 2 Entity Resolution

    Args:
        output_dir: Directory containing stage2 outputs and metrics
    """
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting libraries not available. Skipping visualization.")
        return

    viz = Stage2Visualizer(output_dir)
    viz.generate_all_plots()


class Stage2Visualizer:
    """Visualizes metrics from Stage 2 Entity Resolution pipeline"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        if PLOTTING_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")

    def generate_all_plots(self):
        """Generate all visualization plots"""
        if not PLOTTING_AVAILABLE:
            return

        logger.info("Generating Stage 2 visualization plots...")

        try:
            self.plot_type_distribution()
            self.plot_tier_distribution()
            self.plot_confidence_distribution()
            self.plot_cluster_size_distribution()
            self.plot_embedding_similarity_heatmap()

            logger.info(f"✓ All Stage 2 plots saved to: {self.viz_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    def plot_type_distribution(self):
        """Plot entity type distribution from Stage 0"""
        type_file = self.output_dir / "stage0_entity_types.json"
        if not type_file.exists():
            logger.warning(f"Type file not found: {type_file}")
            return

        with open(type_file, 'r') as f:
            entity_types = json.load(f)

        # Count types
        from collections import Counter
        type_counts = Counter(e["type"] for e in entity_types.values())

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        types = list(type_counts.keys())
        counts = list(type_counts.values())

        bars = ax.bar(types, counts, color=sns.color_palette("husl", len(types)))
        ax.set_xlabel('Entity Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Entity Type Distribution (Stage 0)', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.viz_dir / "type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Generated: type_distribution.png")

    def plot_tier_distribution(self):
        """Plot 3-tier cascading distribution"""
        type_file = self.output_dir / "stage0_entity_types.json"
        if not type_file.exists():
            return

        with open(type_file, 'r') as f:
            entity_types = json.load(f)

        # Count tiers
        from collections import Counter
        tier_counts = Counter(e.get("tier", "unknown") for e in entity_types.values())

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        tiers = ['tier1_keyword', 'tier2_sapbert_knn', 'tier3_llm']
        counts = [tier_counts.get(t, 0) for t in tiers]
        labels = ['Tier 1\n(Keywords)', 'Tier 2\n(SapBERT kNN)', 'Tier 3\n(LLM)']
        colors = ['#2ecc71', '#3498db', '#e74c3c']

        bars = ax1.bar(labels, counts, color=colors)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Tier Distribution', fontsize=14, fontweight='bold')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)

        # Pie chart
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors,
               startangle=90, textprops={'fontsize': 10})
        ax2.set_title('Tier Distribution (%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.viz_dir / "tier_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Generated: tier_distribution.png")

    def plot_confidence_distribution(self):
        """Plot confidence score distribution"""
        type_file = self.output_dir / "stage0_entity_types.json"
        if not type_file.exists():
            return

        with open(type_file, 'r') as f:
            entity_types = json.load(f)

        confidences = [e["confidence"] for e in entity_types.values()]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(confidences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.axvline(np.median(confidences), color='green', linestyle='--',
                   label=f'Median: {np.median(confidences):.3f}')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Box plot
        ax2.boxplot(confidences, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence Box Plot', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "confidence_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Generated: confidence_distribution.png")

    def plot_cluster_size_distribution(self):
        """Plot synonym cluster size distribution"""
        clusters_file = self.output_dir / "stage1b_synonym_clusters.json"
        if not clusters_file.exists():
            logger.warning(f"Clusters file not found: {clusters_file}")
            return

        with open(clusters_file, 'r') as f:
            clusters = json.load(f)

        cluster_sizes = [len(members) for members in clusters.values()]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(cluster_sizes, bins=range(1, max(cluster_sizes)+2),
                color='coral', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Cluster Size', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Stats text
        stats_text = f"""
        Total Clusters: {len(clusters)}
        Singletons: {sum(1 for s in cluster_sizes if s == 1)}
        Min Size: {min(cluster_sizes)}
        Max Size: {max(cluster_sizes)}
        Mean Size: {np.mean(cluster_sizes):.2f}
        Median Size: {np.median(cluster_sizes):.0f}
        """

        ax2.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(self.viz_dir / "cluster_size_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Generated: cluster_size_distribution.png")

    def plot_embedding_similarity_heatmap(self):
        """Plot sample embedding similarity heatmap"""
        embeddings_file = self.output_dir / "stage1_embeddings.npy"
        entity_ids_file = self.output_dir / "stage1_entity_ids.json"

        if not embeddings_file.exists() or not entity_ids_file.exists():
            logger.warning("Embeddings not found, skipping similarity heatmap")
            return

        # Load embeddings
        embeddings = np.load(embeddings_file)
        with open(entity_ids_file, 'r') as f:
            entity_to_id = json.load(f)

        # Sample 50 entities for visualization
        sample_size = min(50, len(entity_to_id))
        sample_entities = list(entity_to_id.keys())[:sample_size]
        sample_ids = [entity_to_id[e] for e in sample_entities]
        sample_embeddings = embeddings[sample_ids]

        # Compute similarity matrix
        similarity_matrix = np.dot(sample_embeddings, sample_embeddings.T)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(similarity_matrix, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1, ax=ax)

        ax.set_title(f'Entity Embedding Similarity Heatmap (Sample {sample_size} entities)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Entity Index', fontsize=12)
        ax.set_ylabel('Entity Index', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.viz_dir / "embedding_similarity_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("  ✓ Generated: embedding_similarity_heatmap.png")
