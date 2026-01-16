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
    
    def compute_funnel_efficiency(self) -> Dict[str, Any]:
        """
        Compute Funnel Efficiency (Phase A metric).

        Tracks candidate reduction through the pipeline stages:
        - Stage 3.2: Initial candidates (dual retrieval + RRF)
        - Stage 3.3: After cluster aggregation
        - Stage 3.4: After hard negative filtering
        - Stage 3.5: After cross-encoder reranking
        - Stage 3.6: Final mappings with confidence

        Measures:
        - Reduction rate per stage
        - Time per stage
        - Candidates/second throughput
        - Bottleneck identification

        Returns:
            Dictionary with funnel efficiency metrics
        """
        # Extract candidate counts and timing from stage metrics
        funnel_stages = []

        for stage_metric in self.stage_metrics:
            stage_name = stage_metric.stage_name
            input_count = stage_metric.input_count
            output_count = stage_metric.output_count
            duration = stage_metric.duration_seconds

            # Calculate reduction and throughput
            if input_count > 0:
                reduction_rate = (input_count - output_count) / input_count
                retention_rate = output_count / input_count
            else:
                reduction_rate = 0.0
                retention_rate = 1.0

            throughput = output_count / duration if duration > 0 else 0.0

            funnel_stages.append({
                "stage": stage_name,
                "input": input_count,
                "output": output_count,
                "reduction_rate": reduction_rate,
                "retention_rate": retention_rate,
                "duration_seconds": duration,
                "throughput_per_sec": throughput
            })

        # Find bottleneck (stage with highest duration)
        if funnel_stages:
            bottleneck = max(funnel_stages, key=lambda x: x["duration_seconds"])
            total_reduction = (
                (funnel_stages[0]["input"] - funnel_stages[-1]["output"]) /
                funnel_stages[0]["input"]
                if funnel_stages[0]["input"] > 0 else 0.0
            )
        else:
            bottleneck = None
            total_reduction = 0.0

        return {
            "funnel_stages": funnel_stages,
            "total_reduction_rate": total_reduction,
            "total_reduction_pct": f"{total_reduction * 100:.1f}%",
            "bottleneck_stage": bottleneck["stage"] if bottleneck else None,
            "bottleneck_duration": bottleneck["duration_seconds"] if bottleneck else 0.0
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall summary"""
        total_warnings = sum(len(m.warnings) for m in self.stage_metrics)
        total_errors = sum(len(m.errors) for m in self.stage_metrics)

        # Add funnel efficiency to summary
        funnel_efficiency = self.compute_funnel_efficiency()

        return {
            'total_stages': len(self.stage_metrics),
            'total_duration_seconds': sum(m.duration_seconds for m in self.stage_metrics),
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'stage_durations': {
                m.stage_name: f"{m.duration_seconds:.2f}s"
                for m in self.stage_metrics
            },
            'funnel_efficiency': funnel_efficiency  # Phase A addition
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

            # Funnel Efficiency (Phase A addition)
            if 'funnel_efficiency' in summary:
                funnel = summary['funnel_efficiency']
                f.write("FUNNEL EFFICIENCY (Phase A)\n")
                f.write("-"*80 + "\n")
                f.write(f"Total Reduction: {funnel['total_reduction_pct']}\n")
                f.write(f"Bottleneck Stage: {funnel['bottleneck_stage']}\n")
                f.write(f"Bottleneck Duration: {funnel['bottleneck_duration']:.2f}s\n\n")

                f.write("Stage-by-Stage Funnel:\n")
                for stage in funnel['funnel_stages']:
                    f.write(f"  {stage['stage']}:\n")
                    f.write(f"    Input → Output: {stage['input']} → {stage['output']}\n")
                    f.write(f"    Retention: {stage['retention_rate']:.1%}\n")
                    f.write(f"    Duration: {stage['duration_seconds']:.2f}s\n")
                    f.write(f"    Throughput: {stage['throughput_per_sec']:.2f} items/s\n")
                f.write("\n")
            
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
    """Metrics for Stage 3.6: Confidence Propagator + Phase A additions"""

    @staticmethod
    def compute_semantic_type_consistency(
        final_mappings: Dict,
        umls_loader,
        kg_entity_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Compute Semantic Type Consistency (Phase A metric).

        NO GOLD LABELS NEEDED - uses weak supervision from type inference.
        Checks if UMLS semantic type matches KG entity type as a quality signal.

        Type mapping from KG types to UMLS semantic types:
        - drug → Pharmacologic Substance, Antibiotic, Vitamin, etc.
        - disease → Disease or Syndrome, Neoplastic Process, etc.
        - symptom → Sign or Symptom, Finding, etc.
        - procedure → Therapeutic or Preventive Procedure, Diagnostic Procedure, etc.
        - gene → Gene or Genome, Amino Acid Sequence, etc.
        - anatomy → Body Part/Organ/Component, Tissue, Cell, etc.

        Reference: UMLS Semantic Network documentation
        https://lhncbc.nlm.nih.gov/semanticnetwork/

        Args:
            final_mappings: Dictionary of entity -> UMLSMapping
            umls_loader: UMLS loader with semantic type access
            kg_entity_types: Dictionary mapping entity -> type from Stage 2

        Returns:
            Dictionary with consistency metrics
        """
        # Type mapping from KG to UMLS semantic types
        type_mapping = {
            "drug": [
                "Pharmacologic Substance",
                "Antibiotic",
                "Vitamin",
                "Hormone",
                "Immunologic Factor",
                "Enzyme",
                "Receptor"
            ],
            "disease": [
                "Disease or Syndrome",
                "Neoplastic Process",
                "Mental or Behavioral Dysfunction",
                "Congenital Abnormality",
                "Acquired Abnormality"
            ],
            "symptom": [
                "Sign or Symptom",
                "Finding",
                "Clinical Attribute"
            ],
            "procedure": [
                "Therapeutic or Preventive Procedure",
                "Diagnostic Procedure",
                "Laboratory Procedure",
                "Health Care Activity"
            ],
            "gene": [
                "Gene or Genome",
                "Amino Acid, Peptide, or Protein",
                "Nucleic Acid, Nucleoside, or Nucleotide",
                "Amino Acid Sequence"
            ],
            "anatomy": [
                "Body Part, Organ, or Organ Component",
                "Tissue",
                "Cell",
                "Cell Component",
                "Body Location or Region",
                "Body Space or Junction"
            ],
            "other": []  # Accept any type for "other"
        }

        consistent_count = 0
        inconsistent_count = 0
        no_type_info = 0
        type_mismatches = []

        for entity, mapping in final_mappings.items():
            # Get KG type
            kg_type = kg_entity_types.get(entity, "other")

            # Get UMLS semantic types for mapped CUI
            try:
                umls_concept = umls_loader.concepts.get(mapping.cui)
                if not umls_concept or not umls_concept.semantic_types:
                    no_type_info += 1
                    continue

                umls_semantic_types = umls_concept.semantic_types
                expected_types = type_mapping.get(kg_type, [])

                # For "other", accept any type
                if kg_type == "other":
                    consistent_count += 1
                    continue

                # Check if any UMLS semantic type matches expected types
                if any(st in expected_types for st in umls_semantic_types):
                    consistent_count += 1
                else:
                    inconsistent_count += 1
                    type_mismatches.append({
                        "entity": entity,
                        "kg_type": kg_type,
                        "umls_types": umls_semantic_types,
                        "expected_types": expected_types
                    })

            except Exception as e:
                logger.warning(f"Error checking semantic type for {entity}: {e}")
                no_type_info += 1

        total_with_type = consistent_count + inconsistent_count
        consistency_rate = consistent_count / total_with_type if total_with_type > 0 else 0.0

        return {
            "consistent_mappings": consistent_count,
            "inconsistent_mappings": inconsistent_count,
            "no_type_info": no_type_info,
            "consistency_rate": consistency_rate,
            "consistency_pct": f"{consistency_rate * 100:.1f}%",
            "sample_mismatches": type_mismatches[:10]  # First 10 for debugging
        }

    @staticmethod
    def compute_coverage_curve(
        final_mappings: Dict,
        confidence_thresholds: List[float] = None
    ) -> Dict[str, Any]:
        """
        Compute Coverage Curve (Phase A metric).

        At each confidence threshold τ: what % of entities are covered?
        Helps find optimal confidence threshold for production use.

        Args:
            final_mappings: Dictionary of entity -> UMLSMapping
            confidence_thresholds: List of thresholds to evaluate
                                  Default: [0.50, 0.55, 0.60, 0.65, 0.70,
                                           0.75, 0.80, 0.85, 0.90, 0.95]

        Returns:
            Dictionary with coverage curve data
        """
        if confidence_thresholds is None:
            confidence_thresholds = [
                0.50, 0.55, 0.60, 0.65, 0.70,
                0.75, 0.80, 0.85, 0.90, 0.95
            ]

        total_entities = len(final_mappings)
        curve_data = []

        for tau in confidence_thresholds:
            above_threshold = [
                m for m in final_mappings.values()
                if m.confidence >= tau
            ]
            coverage = len(above_threshold) / total_entities if total_entities > 0 else 0.0

            # Compute tier distribution at this threshold
            tier_counts = {"high": 0, "medium": 0, "low": 0}
            for m in above_threshold:
                tier_counts[m.tier] += 1

            curve_data.append({
                "threshold": tau,
                "coverage": coverage,
                "coverage_pct": f"{coverage * 100:.1f}%",
                "entities_covered": len(above_threshold),
                "tier_distribution": tier_counts
            })

        return {
            "coverage_curve": curve_data,
            "total_entities": total_entities
        }

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
