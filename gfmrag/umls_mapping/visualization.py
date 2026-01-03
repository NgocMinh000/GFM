"""
Visualization Module for UMLS Mapping Pipeline Metrics

Generates charts and plots for analyzing pipeline performance:
- Score distributions across stages
- Confidence tier breakdowns
- Candidate reduction funnel
- Stage timing analysis
- Cluster size distributions
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


class PipelineVisualizer:
    """
    Visualizes metrics from UMLS mapping pipeline
    """

    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Directory containing pipeline metrics
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Load metrics
        self.metrics = None
        self.load_metrics()

        # Set plot style
        if PLOTTING_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_palette("husl")

    def load_metrics(self):
        """Load pipeline metrics from JSON"""
        metrics_path = self.output_dir / "pipeline_metrics.json"

        if not metrics_path.exists():
            logger.warning(f"Metrics file not found: {metrics_path}")
            return

        with open(metrics_path, 'r') as f:
            self.metrics = json.load(f)

        logger.info(f"Loaded metrics from: {metrics_path}")

    def generate_all_plots(self):
        """Generate all visualization plots"""
        if not PLOTTING_AVAILABLE:
            logger.error("Plotting libraries not available. Skipping visualization.")
            return

        if not self.metrics:
            logger.error("No metrics loaded. Cannot generate plots.")
            return

        logger.info("Generating visualization plots...")

        # Generate each plot
        self.plot_candidate_reduction_funnel()
        self.plot_confidence_distribution()
        self.plot_score_progression()
        self.plot_stage_timing()
        self.plot_cluster_statistics()
        self.plot_metric_heatmap()
        self.plot_quality_metrics()

        logger.info(f"✓ All plots saved to: {self.viz_dir}")

    def plot_candidate_reduction_funnel(self):
        """Plot funnel showing candidate reduction across stages"""
        if not PLOTTING_AVAILABLE:
            return

        # Extract candidate counts per stage
        stages = []
        counts = []

        for stage in self.metrics.get('stages', []):
            stage_name = stage['stage_name']
            metrics = stage.get('metrics', {})

            if 'Stage 3.2' in stage_name:
                stages.append('Stage 3.2\nCandidate Gen')
                counts.append(128)  # ensemble_final_k
            elif 'Stage 3.3' in stage_name:
                stages.append('Stage 3.3\nAggregation')
                counts.append(64)  # cluster_output_k
            elif 'Stage 3.4' in stage_name:
                stages.append('Stage 3.4\nFiltering')
                counts.append(32)  # hard_neg_output_k
            elif 'Stage 3.5' in stage_name:
                stages.append('Stage 3.5\nReranking')
                counts.append(32)
            elif 'Stage 3.6' in stage_name:
                stages.append('Stage 3.6\nFinal')
                counts.append(1)  # Top-1

        if not stages:
            return

        # Create funnel plot
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(stages)))
        bars = ax.barh(range(len(stages)), counts, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                   f'{count} candidates',
                   ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlabel('Number of Candidates', fontsize=12)
        ax.set_title('Candidate Reduction Funnel\n(Progressive Refinement Across Stages)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'candidate_reduction_funnel.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: candidate_reduction_funnel.png")

    def plot_confidence_distribution(self):
        """Plot confidence tier distribution (pie + bar chart)"""
        if not PLOTTING_AVAILABLE:
            return

        # Extract confidence metrics from Stage 3.6
        confidence_metrics = None
        for stage in self.metrics.get('stages', []):
            if 'Stage 3.6' in stage['stage_name']:
                confidence_metrics = stage.get('metrics', {})
                break

        if not confidence_metrics:
            return

        high = confidence_metrics.get('high_confidence', 0)
        medium = confidence_metrics.get('medium_confidence', 0)
        low = confidence_metrics.get('low_confidence', 0)

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Pie chart
        sizes = [high, medium, low]
        labels = [f'High\n({high})', f'Medium\n({medium})', f'Low\n({low})']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
        explode = (0.05, 0, 0)  # Explode high confidence slice

        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
               explode=explode, shadow=True, startangle=90,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax1.set_title('Confidence Tier Distribution', fontsize=14, fontweight='bold')

        # Bar chart
        ax2.bar(['High\n(≥0.75)', 'Medium\n(0.50-0.75)', 'Low\n(<0.50)'],
               sizes, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Number of Mappings', fontsize=12)
        ax2.set_title('Confidence Tier Counts', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (count, color) in enumerate(zip(sizes, colors)):
            ax2.text(i, count + max(sizes)*0.02, str(count),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add target line for high confidence (60%)
        total = sum(sizes)
        target_high = total * 0.6
        ax2.axhline(y=target_high, color='green', linestyle='--', linewidth=2,
                   label=f'Target High (60%): {target_high:.0f}')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: confidence_distribution.png")

    def plot_score_progression(self):
        """Plot score progression across stages"""
        if not PLOTTING_AVAILABLE:
            return

        stages = []
        avg_scores = []

        for stage in self.metrics.get('stages', []):
            stage_name = stage['stage_name']
            metrics = stage.get('metrics', {})

            # Extract average top-1 scores
            if 'avg_top1_score' in metrics:
                stages.append(stage_name.replace('Stage 3.', 'S').split(':')[0])
                avg_scores.append(metrics['avg_top1_score'])
            elif 'avg_final_score' in metrics:
                stages.append(stage_name.replace('Stage 3.', 'S').split(':')[0])
                avg_scores.append(metrics['avg_final_score'])

        if not stages:
            return

        # Create line plot with markers
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(stages, avg_scores, marker='o', markersize=10, linewidth=3,
               color='#3498db', label='Average Top-1 Score')

        # Add value labels
        for i, (stage, score) in enumerate(zip(stages, avg_scores)):
            ax.text(i, score + 0.02, f'{score:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add target line
        ax.axhline(y=0.75, color='green', linestyle='--', linewidth=2,
                  label='Target Score (0.75)')

        ax.set_xlabel('Pipeline Stage', fontsize=12)
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_title('Score Progression Across Pipeline Stages\n(Higher is Better)',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1.0)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'score_progression.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: score_progression.png")

    def plot_stage_timing(self):
        """Plot execution time for each stage"""
        if not PLOTTING_AVAILABLE:
            return

        stages = []
        durations = []

        for stage in self.metrics.get('stages', []):
            stage_name = stage['stage_name'].replace('Stage 3.', 'S')
            # Shorten long names
            if ':' in stage_name:
                stage_name = stage_name.split(':')[0]

            duration = stage.get('duration_seconds', 0)
            stages.append(stage_name)
            durations.append(duration)

        if not stages:
            return

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(stages)))
        bars = ax.barh(range(len(stages)), durations, color=colors, alpha=0.8, edgecolor='black')

        # Add duration labels
        for i, (bar, duration) in enumerate(zip(bars, durations)):
            width = bar.get_width()
            # Format duration
            if duration < 60:
                label = f'{duration:.1f}s'
            elif duration < 3600:
                label = f'{duration/60:.1f}m'
            else:
                label = f'{duration/3600:.1f}h'

            ax.text(width + max(durations)*0.01, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels(stages)
        ax.set_xlabel('Duration (seconds)', fontsize=12)
        ax.set_title('Stage Execution Time\n(Lower is Better)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add total time annotation
        total_time = sum(durations)
        if total_time < 3600:
            total_label = f'Total: {total_time/60:.1f} minutes'
        else:
            total_label = f'Total: {total_time/3600:.1f} hours'

        ax.text(0.95, 0.95, total_label,
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'stage_timing.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: stage_timing.png")

    def plot_cluster_statistics(self):
        """Plot cluster size distribution"""
        if not PLOTTING_AVAILABLE:
            return

        # Extract cluster metrics from Stage 3.1
        cluster_metrics = None
        for stage in self.metrics.get('stages', []):
            if 'Stage 3.1' in stage['stage_name']:
                cluster_metrics = stage.get('metrics', {})
                break

        if not cluster_metrics:
            return

        # Get cluster statistics
        total_clusters = cluster_metrics.get('total_clusters', 0)
        singleton_clusters = cluster_metrics.get('singleton_clusters', 0)
        min_size = cluster_metrics.get('min_cluster_size', 0)
        max_size = cluster_metrics.get('max_cluster_size', 0)
        avg_size = cluster_metrics.get('avg_cluster_size', 0)
        median_size = cluster_metrics.get('median_cluster_size', 0)

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = ['Total\nClusters', 'Singleton\nClusters', 'Min\nSize',
                     'Max\nSize', 'Avg\nSize', 'Median\nSize']
        values = [total_clusters, singleton_clusters, min_size, max_size, avg_size, median_size]
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel('Count / Size', fontsize=12)
        ax.set_title('Synonym Cluster Statistics\n(From Union-Find Clustering)',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'cluster_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: cluster_statistics.png")

    def plot_metric_heatmap(self):
        """Plot heatmap of key metrics across stages"""
        if not PLOTTING_AVAILABLE:
            return

        # Collect key metrics
        stage_names = []
        metric_data = {
            'Avg Score': [],
            'Output Count': [],
            'Duration (s)': [],
            'Warnings': []
        }

        for stage in self.metrics.get('stages', []):
            stage_name = stage['stage_name'].replace('Stage 3.', 'S').split(':')[0]
            stage_names.append(stage_name)

            metrics = stage.get('metrics', {})

            # Extract score (various keys)
            score = (metrics.get('avg_top1_score') or
                    metrics.get('avg_final_score') or
                    metrics.get('avg_confidence') or 0)
            metric_data['Avg Score'].append(score)

            metric_data['Output Count'].append(stage.get('output_count', 0))
            metric_data['Duration (s)'].append(stage.get('duration_seconds', 0))
            metric_data['Warnings'].append(len(stage.get('warnings', [])))

        if not stage_names:
            return

        # Normalize each metric to 0-1 scale for heatmap
        data_matrix = []
        for metric_name, values in metric_data.items():
            if max(values) > 0:
                normalized = [v / max(values) for v in values]
            else:
                normalized = values
            data_matrix.append(normalized)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 6))

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(range(len(stage_names)))
        ax.set_xticklabels(stage_names, rotation=45, ha='right')
        ax.set_yticks(range(len(metric_data)))
        ax.set_yticklabels(metric_data.keys())

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value (0-1)', fontsize=11)

        # Add text annotations
        for i, metric_name in enumerate(metric_data.keys()):
            for j, stage_name in enumerate(stage_names):
                value = metric_data[metric_name][j]
                # Format based on metric type
                if 'Score' in metric_name:
                    text = f'{value:.2f}'
                elif 'Duration' in metric_name:
                    text = f'{value:.0f}'
                else:
                    text = f'{value}'

                ax.text(j, i, text, ha='center', va='center',
                       color='white' if data_matrix[i][j] > 0.5 else 'black',
                       fontsize=9, fontweight='bold')

        ax.set_title('Pipeline Metrics Heatmap (Normalized)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'metric_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: metric_heatmap.png")

    def plot_quality_metrics(self):
        """Plot comprehensive quality metrics"""
        if not PLOTTING_AVAILABLE:
            return

        # Extract Stage 3.6 metrics
        stage6_metrics = None
        for stage in self.metrics.get('stages', []):
            if 'Stage 3.6' in stage['stage_name']:
                stage6_metrics = stage.get('metrics', {})
                break

        if not stage6_metrics:
            return

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Confidence percentages (Target vs Actual)
        categories = ['High\n(≥0.75)', 'Medium\n(0.50-0.75)', 'Low\n(<0.50)']

        # Parse percentages
        high_pct = float(stage6_metrics.get('high_confidence_pct', '0%').rstrip('%'))
        med_pct = float(stage6_metrics.get('medium_confidence_pct', '0%').rstrip('%'))
        low_pct = float(stage6_metrics.get('low_confidence_pct', '0%').rstrip('%'))

        actual = [high_pct, med_pct, low_pct]
        target = [60, 25, 15]  # Target percentages

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax1.bar(x - width/2, actual, width, label='Actual', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, target, width, label='Target', color='#2ecc71', alpha=0.8)

        ax1.set_ylabel('Percentage (%)', fontsize=11)
        ax1.set_title('Confidence Distribution: Actual vs Target', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        # 2. Confidence factors breakdown
        avg_score_margin = stage6_metrics.get('avg_score_margin', 0)
        avg_consensus = stage6_metrics.get('avg_cluster_consensus', 0)
        avg_confidence = stage6_metrics.get('avg_confidence', 0)

        factors = ['Score\nMargin', 'Cluster\nConsensus', 'Overall\nConfidence']
        values = [avg_score_margin, avg_consensus, avg_confidence]
        colors_factors = ['#e74c3c', '#f39c12', '#9b59b6']

        bars = ax2.bar(factors, values, color=colors_factors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Confidence Factors (Averages)', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 3. Propagation statistics
        total = stage6_metrics.get('total_mappings', 0)
        propagated = stage6_metrics.get('propagated_count', 0)
        non_propagated = total - propagated

        labels_prop = ['Propagated', 'Direct Mapping']
        sizes_prop = [propagated, non_propagated]
        colors_prop = ['#e67e22', '#3498db']

        ax3.pie(sizes_prop, labels=labels_prop, autopct='%1.1f%%',
               colors=colors_prop, startangle=90, shadow=True,
               textprops={'fontsize': 11, 'fontweight': 'bold'})
        ax3.set_title('Mapping Source Distribution', fontsize=12, fontweight='bold')

        # 4. Quality summary table
        ax4.axis('off')

        summary_data = [
            ['Metric', 'Value', 'Target', 'Status'],
            ['Avg Confidence', f"{avg_confidence:.3f}", '> 0.65',
             '✓' if avg_confidence > 0.65 else '✗'],
            ['High Conf %', f"{high_pct:.1f}%", '> 60%',
             '✓' if high_pct > 60 else '✗'],
            ['Low Conf %', f"{low_pct:.1f}%", '< 20%',
             '✓' if low_pct < 20 else '✗'],
            ['Score Margin', f"{avg_score_margin:.3f}", '> 0.20',
             '✓' if avg_score_margin > 0.20 else '✗'],
            ['Cluster Consensus', f"{avg_consensus:.3f}", '> 0.70',
             '✓' if avg_consensus > 0.70 else '✗'],
        ]

        table = ax4.table(cellText=summary_data, cellLoc='left',
                         bbox=[0, 0, 1, 1], colWidths=[0.35, 0.25, 0.20, 0.20])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Color status column
        for i in range(1, len(summary_data)):
            status = summary_data[i][3]
            color = '#2ecc71' if status == '✓' else '#e74c3c'
            table[(i, 3)].set_facecolor(color)
            table[(i, 3)].set_text_props(weight='bold', color='white', ha='center')

        ax4.set_title('Quality Metrics Summary', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.viz_dir / 'quality_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info("✓ Generated: quality_metrics.png")

    def generate_summary_report(self):
        """Generate text summary of visualizations"""
        summary_path = self.viz_dir / "visualization_summary.txt"

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STAGE 3 UMLS MAPPING - VISUALIZATION SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write("Generated Visualizations:\n")
            f.write("-"*80 + "\n")
            plots = [
                "candidate_reduction_funnel.png - Shows progressive candidate reduction",
                "confidence_distribution.png - Confidence tier breakdown (pie + bar)",
                "score_progression.png - Score improvement across stages",
                "stage_timing.png - Execution time for each stage",
                "cluster_statistics.png - Synonym cluster size distribution",
                "metric_heatmap.png - Heatmap of all metrics across stages",
                "quality_metrics.png - Comprehensive quality analysis"
            ]

            for i, plot in enumerate(plots, 1):
                f.write(f"{i}. {plot}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Usage:\n")
            f.write("  Open PNG files in visualizations/ directory to analyze pipeline performance\n")
            f.write("="*80 + "\n")

        logger.info(f"✓ Generated visualization summary: {summary_path}")


def visualize_pipeline_metrics(output_dir: str or Path):
    """
    Convenience function to generate all visualizations

    Args:
        output_dir: Directory containing pipeline_metrics.json

    Usage:
        from gfmrag.umls_mapping.visualization import visualize_pipeline_metrics
        visualize_pipeline_metrics('./tmp/umls_mapping')
    """
    visualizer = PipelineVisualizer(output_dir)
    visualizer.generate_all_plots()
    visualizer.generate_summary_report()
    print(f"\n✓ Visualizations saved to: {visualizer.viz_dir}")
