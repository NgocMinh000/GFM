"""
Logging utilities for enhanced visualization of Stage 2 entity resolution.
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class Stage2Logger:
    """Enhanced logger for Stage 2 entity resolution with formatted output"""

    @staticmethod
    def print_stage_header(stage_num: int, stage_name: str):
        """Print formatted stage header"""
        border = "=" * 80
        logger.info(f"\n{border}")
        logger.info(f"📍 STAGE {stage_num}: {stage_name.upper()}")
        logger.info(border)

    @staticmethod
    def print_config(config_dict: Dict[str, Any]):
        """Print configuration parameters"""
        logger.info("\n⚙️  Configuration:")
        for key, value in config_dict.items():
            logger.info(f"   • {key}: {value}")

    @staticmethod
    def print_metrics_table(metrics: Dict[str, Any], title: str = "Metrics"):
        """Print metrics in a formatted table"""
        logger.info(f"\n📊 {title}:")
        logger.info("   " + "-" * 60)
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   | {metric_name:30s} | {value:>10.4f} |")
            elif isinstance(value, int):
                logger.info(f"   | {metric_name:30s} | {value:>10d} |")
            else:
                logger.info(f"   | {metric_name:30s} | {str(value):>10s} |")
        logger.info("   " + "-" * 60)

    @staticmethod
    def print_type_distribution(type_counts: Dict[str, int], total: int):
        """Print entity type distribution as bar chart"""
        logger.info("\n📈 Entity Type Distribution:")

        # Sort by count descending
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        # Calculate max bar width
        max_count = max(type_counts.values())
        bar_width = 40

        for type_name, count in sorted_types:
            percentage = 100 * count / total
            bar_length = int(bar_width * count / max_count)
            bar = "█" * bar_length
            logger.info(f"   {type_name:15s} | {bar:40s} | {count:6d} ({percentage:5.1f}%)")

    @staticmethod
    def print_comparison(before_metrics: Dict, after_metrics: Dict, title: str = "Comparison"):
        """Print before/after comparison"""
        logger.info(f"\n🔄 {title}:")
        logger.info("   " + "-" * 70)
        logger.info(f"   | {'Metric':25s} | {'Before':>15s} | {'After':>15s} | {'Change':>10s} |")
        logger.info("   " + "-" * 70)

        for key in before_metrics.keys():
            if key in after_metrics:
                before_val = before_metrics[key]
                after_val = after_metrics[key]

                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    change = after_val - before_val
                    change_str = f"{change:+.2f}" if isinstance(change, float) else f"{change:+d}"
                else:
                    change_str = "N/A"

                logger.info(f"   | {key:25s} | {str(before_val):>15s} | {str(after_val):>15s} | {change_str:>10s} |")

        logger.info("   " + "-" * 70)

    @staticmethod
    def print_stage_summary(stage_num: int, stage_name: str, execution_time: float, key_results: Dict):
        """Print stage completion summary"""
        logger.info(f"\n✅ Stage {stage_num} ({stage_name}) completed in {execution_time:.2f}s")
        logger.info("   Key Results:")
        for key, value in key_results.items():
            logger.info(f"     • {key}: {value}")

    @staticmethod
    def print_pipeline_summary(total_time: float, all_stage_results: List[Dict]):
        """Print overall pipeline summary"""
        border = "=" * 80
        logger.info(f"\n{border}")
        logger.info("🎉 ENTITY RESOLUTION PIPELINE COMPLETED")
        logger.info(border)
        logger.info(f"\n⏱️  Total Execution Time: {total_time:.2f}s")
        logger.info("\n📋 Stage-by-Stage Summary:")

        for stage_result in all_stage_results:
            stage_num = stage_result["stage"]
            stage_name = stage_result["name"]
            stage_time = stage_result["time"]
            logger.info(f"\n   Stage {stage_num}: {stage_name}")
            logger.info(f"     ⏱  Time: {stage_time:.2f}s")
            for key, value in stage_result.get("results", {}).items():
                logger.info(f"     • {key}: {value}")

        logger.info(f"\n{border}")

    @staticmethod
    def print_progress(current: int, total: int, prefix: str = "Progress"):
        """Print progress indicator"""
        percentage = 100 * current / total
        filled = int(50 * current / total)
        bar = "█" * filled + "░" * (50 - filled)
        logger.info(f"\r   {prefix}: |{bar}| {percentage:.1f}% ({current}/{total})")

    @staticmethod
    def print_warning(message: str):
        """Print warning message"""
        logger.warning(f"⚠️  {message}")

    @staticmethod
    def print_success(message: str):
        """Print success message"""
        logger.info(f"✅ {message}")

    @staticmethod
    def print_error(message: str):
        """Print error message"""
        logger.error(f"❌ {message}")
