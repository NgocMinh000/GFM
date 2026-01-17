"""
Stage 3: UMLS Mapping Pipeline
Maps biomedical entities to UMLS CUIs through 6-stage pipeline
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import asdict
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from gfmrag.umls_mapping import (
    UMLSMappingConfig,
    UMLSLoader,
    Preprocessor,
    CandidateGenerator,
    ClusterAggregator,
    HardNegativeFilter,
    CrossEncoderReranker,
    ConfidencePropagator,
    MetricsTracker,
    Stage0Metrics,
    Stage1Metrics,
    Stage2Metrics,
    Stage3Metrics,
    Stage4Metrics,
    Stage5Metrics,
    Stage6Metrics,
)

logger = logging.getLogger(__name__)


class Stage3UMLSMapping:
    """
    6-Stage UMLS Mapping Pipeline

    Stages:
    0. UMLS Data Loading & Indexing
    1. Preprocessing & Entity Extraction
    2. Candidate Generation (SapBERT + TF-IDF)
    3. Synonym Cluster Aggregation
    4. Hard Negative Filtering
    5. Cross-Encoder Reranking
    6. Confidence Scoring & Propagation
    """

    def __init__(self, config: UMLSMappingConfig, hydra_cfg=None):
        self.config = config
        self.hydra_cfg = hydra_cfg  # Store Hydra config for filtering
        self.output_dir = Path(config.output_root)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics tracker
        self.metrics = MetricsTracker(self.output_dir)

        # Initialize components
        self.umls_loader = None
        self.preprocessor = None
        self.candidate_generator = None
        self.cluster_aggregator = None
        self.hard_negative_filter = None
        self.cross_encoder = None
        self.confidence_propagator = None

    def run(self):
        """Run complete 6-stage pipeline"""

        logger.info("=" * 80)
        logger.info("Stage 3: UMLS Mapping Pipeline")
        logger.info("=" * 80)

        # Stage 3.0: Load UMLS data
        logger.info("\n[Stage 3.0] Loading UMLS data...")
        self.metrics.start_stage("Stage 3.0: UMLS Data Loading", input_count=0)

        self.umls_loader = UMLSLoader(self.config)
        umls_concepts = self.umls_loader.load()

        # Compute metrics
        stage0_metrics = Stage0Metrics.compute(self.umls_loader)
        for key, value in stage0_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(umls_concepts))

        # Stage 3.1: Preprocessing
        logger.info("\n[Stage 3.1] Preprocessing entities...")
        self.metrics.start_stage("Stage 3.1: Preprocessing", input_count=1)

        self.preprocessor = Preprocessor(self.config)
        entities = self.preprocessor.process(self.config.kg_clean_path)

        # Compute metrics
        stage1_metrics = Stage1Metrics.compute(self.preprocessor)
        for key, value in stage1_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(entities))

        # Stage 3.2: Candidate Generation
        logger.info("\n[Stage 3.2] Generating candidates...")
        self.metrics.start_stage("Stage 3.2: Candidate Generation", input_count=len(entities))

        self.candidate_generator = CandidateGenerator(self.config, self.umls_loader)

        entity_candidates = {}

        # Batch processing for 5-10x speedup
        batch_size = 32  # Process 32 entities at once
        entity_list = list(entities.keys())

        logger.info(f"Processing {len(entity_list)} entities in batches of {batch_size}")
        logger.info("Batch encoding + search for faster candidate generation")

        # Temporarily reduce logging to avoid tqdm conflicts
        import logging
        original_level = logging.getLogger('gfmrag.umls_mapping.candidate_generator').level
        logging.getLogger('gfmrag.umls_mapping.candidate_generator').setLevel(logging.WARNING)

        pbar = tqdm(range(0, len(entity_list), batch_size), desc="Generating candidates (batched)", unit="batch")
        for i in pbar:
            batch_entities = entity_list[i:i+batch_size]
            batch_texts = [entities[e].normalized for e in batch_entities]

            # Batch generate candidates (5-10x faster than single)
            batch_candidates = self.candidate_generator.generate_candidates_batch(
                batch_texts,
                k=self.config.ensemble_final_k
            )

            # Store results
            for entity, candidates in zip(batch_entities, batch_candidates):
                entity_candidates[entity] = candidates

                # Track warnings
                if len(candidates) == 0:
                    self.metrics.add_warning(f"No candidates found for: {entity}")

            # Update progress bar with stats
            pbar.set_postfix({
                'avg_candidates': f"{sum(len(c) for c in batch_candidates)/len(batch_candidates):.1f}"
            })

        # Restore logging level
        logging.getLogger('gfmrag.umls_mapping.candidate_generator').setLevel(original_level)

        # Compute metrics
        stage2_metrics = Stage2Metrics.compute(entity_candidates)
        for key, value in stage2_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(entity_candidates))

        if self.config.save_intermediate:
            self._save_json(entity_candidates, "stage32_candidates.json")

        # Stage 3.3: Cluster Aggregation
        logger.info("\n[Stage 3.3] Aggregating clusters...")
        self.metrics.start_stage("Stage 3.3: Cluster Aggregation", input_count=len(entity_candidates))

        self.cluster_aggregator = ClusterAggregator(self.config)

        # Build entity -> cluster members mapping
        entity_to_cluster = {
            entity: entities[entity].synonym_group
            for entity in entities.keys()
        }

        aggregated_candidates = self.cluster_aggregator.aggregate_multiple_clusters(
            entity_candidates,
            entity_to_cluster
        )

        # Compute metrics
        stage3_metrics = Stage3Metrics.compute(aggregated_candidates)
        for key, value in stage3_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(aggregated_candidates))

        if self.config.save_intermediate:
            self._save_json(aggregated_candidates, "stage33_aggregated.json")

        # Stage 3.4: Hard Negative Filtering
        logger.info("\n[Stage 3.4] Filtering hard negatives...")
        self.metrics.start_stage("Stage 3.4: Hard Negative Filtering", input_count=len(aggregated_candidates))

        self.hard_negative_filter = HardNegativeFilter(self.config, self.umls_loader)

        # TODO: Build KG context for semantic type inference
        kg_context = {}  # Placeholder

        filtered_candidates = {}
        for entity, candidates in tqdm(aggregated_candidates.items(), desc="Filtering"):
            filtered = self.hard_negative_filter.filter_candidates(
                entity,
                candidates,
                kg_context
            )
            filtered_candidates[entity] = filtered

        # Compute metrics
        stage4_metrics = Stage4Metrics.compute(filtered_candidates)
        for key, value in stage4_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(filtered_candidates))

        if self.config.save_intermediate:
            self._save_json(filtered_candidates, "stage34_filtered.json")

        # Stage 3.5: Cross-Encoder Reranking
        logger.info("\n[Stage 3.5] Reranking with cross-encoder...")
        self.metrics.start_stage("Stage 3.5: Cross-Encoder Reranking", input_count=len(filtered_candidates))

        self.cross_encoder = CrossEncoderReranker(self.config)

        reranked_candidates = {}
        for entity, candidates in tqdm(filtered_candidates.items(), desc="Reranking"):
            reranked = self.cross_encoder.rerank(entity, candidates)
            reranked_candidates[entity] = reranked

        # Compute metrics
        stage5_metrics = Stage5Metrics.compute(reranked_candidates)
        for key, value in stage5_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(reranked_candidates))

        if self.config.save_intermediate:
            self._save_json(reranked_candidates, "stage35_reranked.json")

        # Stage 3.6: Confidence Scoring & Propagation
        logger.info("\n[Stage 3.6] Computing confidence & propagating...")
        self.metrics.start_stage("Stage 3.6: Confidence & Propagation", input_count=len(reranked_candidates))

        self.confidence_propagator = ConfidencePropagator(self.config)

        # Compute initial mappings
        final_mappings = {}
        for entity, candidates in tqdm(reranked_candidates.items(), desc="Computing confidence"):
            cluster_members = entities[entity].synonym_group
            mapping = self.confidence_propagator.compute_confidence(
                entity,
                candidates,
                cluster_members
            )
            final_mappings[entity] = mapping

            # Track low confidence cases
            if mapping.tier == 'low':
                self.metrics.add_warning(f"Low confidence mapping: {entity} -> {mapping.cui} ({mapping.confidence:.3f})")

        # Propagate through clusters
        final_mappings = self.confidence_propagator.finalize_all_mappings(
            final_mappings,
            self.preprocessor.synonym_clusters
        )

        # Compute metrics
        stage6_metrics = Stage6Metrics.compute(final_mappings)
        for key, value in stage6_metrics.items():
            self.metrics.add_metric(key, value)

        self.metrics.end_stage(output_count=len(final_mappings))

        # Save final results (with filtering if enabled)
        self._save_final_outputs(final_mappings, cfg=self.hydra_cfg)

        # Save all metrics
        self.metrics.save_metrics()

        # Generate visualizations
        logger.info("\nGenerating visualization plots...")
        try:
            from gfmrag.umls_mapping.visualization import visualize_pipeline_metrics
            visualize_pipeline_metrics(self.output_dir)
            logger.info("✓ Visualizations generated successfully")
        except ImportError:
            logger.warning("Matplotlib/Seaborn not installed. Skipping visualization.")
            logger.warning("Install with: pip install matplotlib seaborn")
        except Exception as e:
            logger.warning(f"Failed to generate visualizations: {e}")

        logger.info("\n" + "=" * 80)
        logger.info("Stage 3 Complete!")
        logger.info("=" * 80)
        logger.info(f"\nOutput files:")
        logger.info(f"  - Mappings: {self.output_dir / 'final_umls_mappings.json'}")
        logger.info(f"  - Metrics: {self.output_dir / 'pipeline_metrics.json'}")
        logger.info(f"  - Report: {self.output_dir / 'pipeline_report.txt'}")
        logger.info(f"  - Visualizations: {self.output_dir / 'visualizations/'}")

        return final_mappings

    def _save_json(self, data, filename: str):
        """Save intermediate results as JSON"""
        output_path = self.output_dir / filename
        
        # Convert to JSON-serializable format
        json_data = {}
        for key, value in data.items():
            if isinstance(value, list):
                json_data[key] = [asdict(v) if hasattr(v, '__dataclass_fields__') else v for v in value]
            else:
                json_data[key] = asdict(value) if hasattr(value, '__dataclass_fields__') else value
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        logger.info(f"Saved intermediate results to: {output_path}")

    def _save_final_outputs(self, final_mappings: Dict, cfg=None):
        """Save final outputs in multiple formats with KG filtering"""

        logger.info("\nSaving final outputs...")

        # Import KG Filter
        from gfmrag.umls_mapping.kg_filter import KGFilter, FilterConfig

        # Initialize filter from config (if provided)
        kg_filter = None
        if cfg and cfg.get('output_format', {}).get('kg_filtering', {}).get('enabled', False):
            filter_cfg_dict = cfg['output_format']['kg_filtering']
            filter_config = FilterConfig(
                enabled=filter_cfg_dict.get('enabled', True),
                min_confidence=filter_cfg_dict.get('min_confidence_for_inclusion', 0.6),
                remove_all_triples=filter_cfg_dict.get('remove_all_triples', True),
                require_medical_semantic_type=filter_cfg_dict.get('require_medical_semantic_type', False),
                allowed_semantic_type_groups=filter_cfg_dict.get('allowed_semantic_type_groups', []),
                entity_blacklist=filter_cfg_dict.get('entity_blacklist', []),
                filter_patterns=filter_cfg_dict.get('filter_patterns', [])
            )
            kg_filter = KGFilter(filter_config, self.umls_loader)
            logger.info("\n" + "=" * 80)
            logger.info("KG FILTERING ENABLED")
            logger.info("=" * 80)

        # Apply filtering
        if kg_filter:
            included_mappings, excluded_entities = kg_filter.filter_mappings(final_mappings)
            logger.info(f"\n✓ Filtered {len(excluded_entities)} entities from KG")
        else:
            included_mappings = final_mappings
            excluded_entities = {}

        # 1. JSON with full details (UNFILTERED - for analysis)
        json_path = self.output_dir / "final_umls_mappings_all.json"
        json_data = {}
        for entity, mapping in final_mappings.items():
            json_data[entity] = {
                'cui': mapping.cui,
                'name': mapping.name,
                'confidence': mapping.confidence,
                'tier': mapping.tier,
                'alternatives': [
                    {'cui': cui, 'name': name, 'score': score}
                    for cui, name, score in mapping.alternatives
                ],
                'cluster_size': len(mapping.cluster_members),
                'is_propagated': mapping.is_propagated,
                'confidence_factors': mapping.confidence_factors
            }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, default=str)
        logger.info(f"Saved ALL mappings (unfiltered) to: {json_path}")

        # 1b. JSON with FILTERED mappings (for KG construction)
        if kg_filter:
            json_filtered_path = self.output_dir / "final_umls_mappings_filtered.json"
            json_filtered = {}
            for entity, mapping in included_mappings.items():
                json_filtered[entity] = {
                    'cui': mapping.cui,
                    'name': mapping.name,
                    'confidence': mapping.confidence,
                    'tier': mapping.tier,
                    'alternatives': [
                        {'cui': cui, 'name': name, 'score': score}
                        for cui, name, score in mapping.alternatives
                    ],
                    'cluster_size': len(mapping.cluster_members),
                    'is_propagated': mapping.is_propagated,
                    'confidence_factors': mapping.confidence_factors
                }

            with open(json_filtered_path, 'w', encoding='utf-8') as f:
                json.dump(json_filtered, f, indent=2, default=str)
            logger.info(f"Saved FILTERED mappings (for KG) to: {json_filtered_path}")

        # 1c. Save excluded entities with reasons
        if kg_filter and excluded_entities:
            excluded_path = self.output_dir / "excluded_entities.json"
            with open(excluded_path, 'w', encoding='utf-8') as f:
                json.dump(excluded_entities, f, indent=2)
            logger.info(f"Saved excluded entities to: {excluded_path}")

        # 2. Triples for KG (entity | mapped_to_cui | CUI) - FILTERED
        triples_path = self.output_dir / "umls_mapping_triples.txt"
        with open(triples_path, 'w', encoding='utf-8') as f:
            for entity, mapping in included_mappings.items():
                f.write(f"{entity}|mapped_to_cui|{mapping.cui}\n")
        logger.info(f"Saved KG triples (FILTERED) to: {triples_path}")

        # 3. Statistics (with filtering info)
        stats_path = self.output_dir / "mapping_statistics.json"
        total = len(final_mappings)
        high = sum(1 for m in final_mappings.values() if m.tier == 'high')
        medium = sum(1 for m in final_mappings.values() if m.tier == 'medium')
        low = sum(1 for m in final_mappings.values() if m.tier == 'low')
        propagated = sum(1 for m in final_mappings.values() if m.is_propagated)

        stats = {
            'total_entities': total,
            'high_confidence': high,
            'medium_confidence': medium,
            'low_confidence': low,
            'propagated': propagated,
            'high_confidence_pct': f"{high/total*100:.2f}%",
            'medium_confidence_pct': f"{medium/total*100:.2f}%",
            'low_confidence_pct': f"{low/total*100:.2f}%",
            'propagated_pct': f"{propagated/total*100:.2f}%"
        }

        # Add filtering stats
        if kg_filter:
            stats['kg_filtering'] = {
                'enabled': True,
                'total_entities': total,
                'included_in_kg': len(included_mappings),
                'excluded_from_kg': len(excluded_entities),
                'inclusion_rate': f"{len(included_mappings)/total*100:.2f}%",
                'exclusion_rate': f"{len(excluded_entities)/total*100:.2f}%",
                'min_confidence_threshold': kg_filter.config.min_confidence
            }
        else:
            stats['kg_filtering'] = {
                'enabled': False
            }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to: {stats_path}")

        # 4. Manual review queue
        review_path = self.output_dir / "manual_review_queue.json"
        review_queue = {
            entity: asdict(mapping)
            for entity, mapping in final_mappings.items()
            if mapping.tier == 'low' or (mapping.tier == 'medium' and not mapping.is_propagated)
        }

        with open(review_path, 'w') as f:
            json.dump(review_queue, f, indent=2, default=str)
        logger.info(f"Saved manual review queue to: {review_path}")


@hydra.main(version_base=None, config_path="config", config_name="stage3_umls_mapping")
def main(cfg: DictConfig):
    """Main entry point"""

    # Convert Hydra config to UMLSMappingConfig
    config = UMLSMappingConfig(
        kg_clean_path=cfg.input.kg_clean_path,
        umls_data_dir=cfg.input.umls_data_dir,
        output_root=cfg.output.root_dir,
        mrconso_path=cfg.umls.files.mrconso,
        mrsty_path=cfg.umls.files.mrsty,
        mrdef_path=cfg.umls.files.get('mrdef', None),
        umls_language=cfg.umls.language,
        umls_cache_dir=cfg.umls.cache_dir,
        precompute_embeddings=cfg.umls.precompute.embeddings,
        sapbert_model=cfg.candidate_generation.sapbert.model,
        sapbert_batch_size=cfg.candidate_generation.sapbert.batch_size,
        sapbert_device=cfg.candidate_generation.sapbert.device,
        sapbert_top_k=cfg.candidate_generation.sapbert.top_k,
        tfidf_ngram_range=tuple(cfg.candidate_generation.tfidf.ngram_range),
        ensemble_final_k=cfg.candidate_generation.ensemble.final_k,
        cluster_output_k=cfg.cluster_aggregation.output_k,
        hard_neg_similarity_threshold=cfg.hard_negative_filtering.hard_negative.similarity_threshold,
        hard_neg_output_k=cfg.hard_negative_filtering.output_k,
        cross_encoder_model=cfg.cross_encoder.model,
        cross_encoder_device=cfg.cross_encoder.device,
        confidence_high_threshold=cfg.confidence.tiers.high,
        propagation_min_agreement=cfg.confidence.propagation.min_cluster_agreement,
        num_processes=cfg.general.num_processes,
        force_recompute=cfg.general.force_recompute,
        save_intermediate=cfg.general.save_intermediate,
        device=cfg.general.device
    )

    # Run pipeline (pass Hydra config for KG filtering)
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # Convert to dict
    pipeline = Stage3UMLSMapping(config, hydra_cfg=cfg_dict)
    pipeline.run()


if __name__ == "__main__":
    main()
