"""
Evaluation Metrics for Stage 3 UMLS Mapping
Tracks performance and quality at each stage
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single stage"""
    stage_name: str
    start_time: float
    end_time: float
    duration_seconds: float
    
    # Input/Output counts
    input_count: int
    output_count: int
    
    # Stage-specific metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Warnings/Issues
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class MetricsTracker:
    """
    Tracks metrics across all stages of UMLS mapping pipeline
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage_metrics: List[StageMetrics] = []
        self.current_stage: Optional[str] = None
        self.stage_start_time: Optional[float] = None
        self.stage_input_count: int = 0
        self.stage_output_count: int = 0
        self.stage_metrics_dict: Dict[str, Any] = {}
        self.stage_warnings: List[str] = []
        self.stage_errors: List[str] = []
    
    def start_stage(self, stage_name: str, input_count: int):
        """Start tracking a stage"""
        self.current_stage = stage_name
        self.stage_start_time = time.time()
        self.stage_input_count = input_count
        self.stage_output_count = 0
        self.stage_metrics_dict = {}
        self.stage_warnings = []
        self.stage_errors = []
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[{stage_name}] Started with {input_count} inputs")
        logger.info(f"{'='*80}")
    
    def end_stage(self, output_count: int):
        """End tracking a stage"""
        if not self.current_stage:
            return
        
        end_time = time.time()
        duration = end_time - self.stage_start_time
        
        metrics = StageMetrics(
            stage_name=self.current_stage,
            start_time=self.stage_start_time,
            end_time=end_time,
            duration_seconds=duration,
            input_count=self.stage_input_count,
            output_count=output_count,
            metrics=self.stage_metrics_dict.copy(),
            warnings=self.stage_warnings.copy(),
            errors=self.stage_errors.copy()
        )
        
        self.stage_metrics.append(metrics)
        
        # Log summary
        logger.info(f"\n{'-'*80}")
        logger.info(f"[{self.current_stage}] Completed in {duration:.2f}s")
        logger.info(f"  Input: {self.stage_input_count}, Output: {output_count}")
        logger.info(f"  Warnings: {len(self.stage_warnings)}, Errors: {len(self.stage_errors)}")
        for key, value in self.stage_metrics_dict.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"{'-'*80}\n")
        
        # Reset
        self.current_stage = None
        self.stage_start_time = None
    
    def add_metric(self, name: str, value: Any):
        """Add a metric to current stage"""
        self.stage_metrics_dict[name] = value
    
    def add_warning(self, warning: str):
        """Add a warning to current stage"""
        self.stage_warnings.append(warning)
        logger.warning(f"[{self.current_stage}] {warning}")
    
    def add_error(self, error: str):
        """Add an error to current stage"""
        self.stage_errors.append(error)
        logger.error(f"[{self.current_stage}] {error}")
    
    def save_metrics(self):
        """Save all metrics to file"""
        output_path = self.output_dir / "pipeline_metrics.json"
        
        metrics_data = {
            'stages': [asdict(m) for m in self.stage_metrics],
            'total_duration': sum(m.duration_seconds for m in self.stage_metrics),
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Metrics saved to: {output_path}")
        logger.info(f"{'='*80}\n")
        
        # Also save human-readable report
        self._save_report()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary"""
        total_warnings = sum(len(m.warnings) for m in self.stage_metrics)
        total_errors = sum(len(m.errors) for m in self.stage_metrics)
        
        return {
            'total_stages': len(self.stage_metrics),
            'total_duration_seconds': sum(m.duration_seconds for m in self.stage_metrics),
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'stage_durations': {
                m.stage_name: f"{m.duration_seconds:.2f}s" 
                for m in self.stage_metrics
            }
        }
    
    def _save_report(self):
        """Save human-readable report"""
        report_path = self.output_dir / "pipeline_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("STAGE 3 UMLS MAPPING - PIPELINE REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary
            summary = self._generate_summary()
            f.write("OVERALL SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Duration: {summary['total_duration_seconds']:.2f}s\n")
            f.write(f"Total Warnings: {summary['total_warnings']}\n")
            f.write(f"Total Errors: {summary['total_errors']}\n\n")
            
            # Stage by stage
            for metrics in self.stage_metrics:
                f.write("="*80 + "\n")
                f.write(f"{metrics.stage_name}\n")
                f.write("="*80 + "\n")
                f.write(f"Duration: {metrics.duration_seconds:.2f}s\n")
                f.write(f"Input Count: {metrics.input_count}\n")
                f.write(f"Output Count: {metrics.output_count}\n")
                f.write(f"Throughput: {metrics.output_count/metrics.duration_seconds:.2f} items/s\n\n")
                
                if metrics.metrics:
                    f.write("Metrics:\n")
                    for key, value in metrics.metrics.items():
                        f.write(f"  - {key}: {value}\n")
                    f.write("\n")
                
                if metrics.warnings:
                    f.write(f"Warnings ({len(metrics.warnings)}):\n")
                    for w in metrics.warnings[:10]:  # Show first 10
                        f.write(f"  - {w}\n")
                    if len(metrics.warnings) > 10:
                        f.write(f"  ... and {len(metrics.warnings)-10} more\n")
                    f.write("\n")
                
                if metrics.errors:
                    f.write(f"Errors ({len(metrics.errors)}):\n")
                    for e in metrics.errors[:10]:
                        f.write(f"  - {e}\n")
                    if len(metrics.errors) > 10:
                        f.write(f"  ... and {len(metrics.errors)-10} more\n")
                    f.write("\n")
        
        logger.info(f"Report saved to: {report_path}")


# Stage-specific metric calculators

class Stage0Metrics:
    """Metrics for Stage 3.0: UMLS Loader"""

    @staticmethod
    def compute(umls_loader) -> Dict[str, Any]:
        return {
            'total_concepts': len(umls_loader.concepts),
            'total_unique_names': len(umls_loader.umls_aliases),
            'avg_names_per_concept': len(umls_loader.umls_aliases) / max(len(umls_loader.concepts), 1),
            'concepts_with_definitions': sum(1 for c in umls_loader.concepts.values() if c.definitions),
            'avg_semantic_types_per_concept': np.mean([len(c.semantic_types) for c in umls_loader.concepts.values()]) if umls_loader.concepts else 0.0
        }


class Stage1Metrics:
    """Metrics for Stage 3.1: Preprocessor"""
    
    @staticmethod
    def compute(preprocessor) -> Dict[str, Any]:
        cluster_sizes = [len(members) for members in preprocessor.synonym_clusters.values()]
        
        return {
            'total_entities': len(preprocessor.entities),
            'total_clusters': len(preprocessor.synonym_clusters),
            'singleton_clusters': sum(1 for size in cluster_sizes if size == 1),
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0
        }


class Stage2Metrics:
    """Metrics for Stage 3.2: Candidate Generator"""
    
    @staticmethod
    def compute(entity_candidates: Dict) -> Dict[str, Any]:
        candidate_counts = [len(cands) for cands in entity_candidates.values()]
        
        # Score distributions
        all_scores = []
        for cands in entity_candidates.values():
            all_scores.extend([c.score for c in cands])
        
        return {
            'entities_with_candidates': len(entity_candidates),
            'avg_candidates_per_entity': np.mean(candidate_counts) if candidate_counts else 0,
            'entities_with_no_candidates': sum(1 for c in candidate_counts if c == 0),
            'avg_top1_score': np.mean([cands[0].score for cands in entity_candidates.values() if len(cands) > 0]) if entity_candidates else 0,
            'min_candidate_score': min(all_scores) if all_scores else 0,
            'max_candidate_score': max(all_scores) if all_scores else 0,
            'avg_candidate_score': np.mean(all_scores) if all_scores else 0
        }


class Stage3Metrics:
    """Metrics for Stage 3.3: Cluster Aggregator"""
    
    @staticmethod
    def compute(aggregated_candidates: Dict) -> Dict[str, Any]:
        all_scores = []
        outlier_counts = []
        support_counts = []
        
        for cands in aggregated_candidates.values():
            if cands:
                all_scores.append(cands[0].score)
                outlier_counts.append(sum(1 for c in cands if c.is_outlier))
                support_counts.extend([c.cluster_support for c in cands])
        
        return {
            'clusters_processed': len(aggregated_candidates),
            'avg_top1_score_after_aggregation': np.mean(all_scores) if all_scores else 0,
            'avg_outliers_per_cluster': np.mean(outlier_counts) if outlier_counts else 0,
            'avg_cluster_support': np.mean(support_counts) if support_counts else 0
        }


class Stage4Metrics:
    """Metrics for Stage 3.4: Hard Negative Filter"""
    
    @staticmethod
    def compute(filtered_candidates: Dict) -> Dict[str, Any]:
        type_matches = []
        hard_neg_penalties = []
        all_scores = []
        
        for cands in filtered_candidates.values():
            if cands:
                all_scores.append(cands[0].score)
                type_matches.extend([c.type_match for c in cands])
                hard_neg_penalties.extend([c.hard_negative_penalty for c in cands])
        
        return {
            'entities_filtered': len(filtered_candidates),
            'avg_top1_score_after_filtering': np.mean(all_scores) if all_scores else 0,
            'type_match_rate': np.mean(type_matches) if type_matches else 0,
            'avg_hard_negative_penalty': np.mean(hard_neg_penalties) if hard_neg_penalties else 0,
            'candidates_with_penalties': sum(1 for p in hard_neg_penalties if p > 0)
        }


class Stage5Metrics:
    """Metrics for Stage 3.5: Cross-Encoder Reranker"""
    
    @staticmethod
    def compute(reranked_candidates: Dict) -> Dict[str, Any]:
        cross_scores = []
        previous_scores = []
        final_scores = []
        
        for cands in reranked_candidates.values():
            if cands:
                final_scores.append(cands[0].score)
                cross_scores.extend([c.cross_encoder_score for c in cands])
                previous_scores.extend([c.previous_score for c in cands])
        
        return {
            'entities_reranked': len(reranked_candidates),
            'avg_final_score': np.mean(final_scores) if final_scores else 0,
            'avg_cross_encoder_score': np.mean(cross_scores) if cross_scores else 0,
            'avg_previous_score': np.mean(previous_scores) if previous_scores else 0,
            'score_improvement': (np.mean(final_scores) - np.mean(previous_scores)) if (final_scores and previous_scores) else 0
        }


class Stage6Metrics:
    """Metrics for Stage 3.6: Confidence Propagator"""
    
    @staticmethod
    def compute(final_mappings: Dict) -> Dict[str, Any]:
        confidences = [m.confidence for m in final_mappings.values()]
        
        tier_counts = defaultdict(int)
        for m in final_mappings.values():
            tier_counts[m.tier] += 1
        
        propagated_count = sum(1 for m in final_mappings.values() if m.is_propagated)
        
        # Confidence factors analysis
        factor_means = defaultdict(list)
        for m in final_mappings.values():
            if m.confidence_factors:
                for key, value in m.confidence_factors.items():
                    if isinstance(value, (int, float)):
                        factor_means[key].append(value)
        
        return {
            'total_mappings': len(final_mappings),
            'high_confidence': tier_counts['high'],
            'medium_confidence': tier_counts['medium'],
            'low_confidence': tier_counts['low'],
            'high_confidence_pct': f"{tier_counts['high']/len(final_mappings)*100:.2f}%" if final_mappings else "0%",
            'medium_confidence_pct': f"{tier_counts['medium']/len(final_mappings)*100:.2f}%" if final_mappings else "0%",
            'low_confidence_pct': f"{tier_counts['low']/len(final_mappings)*100:.2f}%" if final_mappings else "0%",
            'propagated_count': propagated_count,
            'propagated_pct': f"{propagated_count/len(final_mappings)*100:.2f}%" if final_mappings else "0%",
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'median_confidence': np.median(confidences) if confidences else 0,
            'min_confidence': min(confidences) if confidences else 0,
            'max_confidence': max(confidences) if confidences else 0,
            'avg_score_margin': np.mean([np.mean(v) for k, v in factor_means.items() if k == 'score_margin']) if 'score_margin' in factor_means else 0,
            'avg_cluster_consensus': np.mean([np.mean(v) for k, v in factor_means.items() if k == 'cluster_consensus']) if 'cluster_consensus' in factor_means else 0
        }
