"""
UMLS Mapping Pipeline Orchestrator

Automates the complete 6-stage UMLS mapping pipeline with:
- Stage tracking and resume capability
- Error handling and recovery
- Progress monitoring
- Unified logging
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import traceback

from .config import UMLSMappingConfig
from .umls_loader import UMLSLoader
from .preprocessor import Preprocessor
from .validation import Stage1Validator


logger = logging.getLogger(__name__)


class PipelineStatus:
    """Tracks pipeline execution status"""

    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.status = self._load_status()

    def _load_status(self) -> Dict[str, Any]:
        """Load status from file"""
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {
            'completed_stages': [],
            'failed_stages': [],
            'last_run': None,
            'last_successful_stage': None
        }

    def save(self):
        """Save status to file"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=2)

    def mark_completed(self, stage: str):
        """Mark stage as completed"""
        if stage not in self.status['completed_stages']:
            self.status['completed_stages'].append(stage)
        if stage in self.status['failed_stages']:
            self.status['failed_stages'].remove(stage)
        self.status['last_successful_stage'] = stage
        self.status['last_run'] = datetime.now().isoformat()
        self.save()

    def mark_failed(self, stage: str):
        """Mark stage as failed"""
        if stage not in self.status['failed_stages']:
            self.status['failed_stages'].append(stage)
        self.save()

    def is_completed(self, stage: str) -> bool:
        """Check if stage is completed"""
        return stage in self.status['completed_stages']

    def reset(self):
        """Reset all status"""
        self.status = {
            'completed_stages': [],
            'failed_stages': [],
            'last_run': None,
            'last_successful_stage': None
        }
        self.save()


class UMLSMappingPipeline:
    """
    Complete UMLS Mapping Pipeline Orchestrator

    Stages:
    - Stage 0: UMLS database loading (one-time)
    - Stage 1: Entity extraction and preprocessing
    - Stage 2 Setup: SapBERT + TF-IDF setup (one-time)
    - Stage 2: Candidate generation
    - Stage 3: Cluster aggregation
    - Stage 4: Hard negative filtering
    - Stage 5: Cross-encoder reranking
    - Stage 6: Confidence scoring and final output
    """

    STAGES = [
        'stage0_umls_loading',
        'stage1_preprocessing',
        'stage2_setup_sapbert',
        'stage2_setup_tfidf',
        'stage2_candidate_generation',
        'stage3_cluster_aggregation',
        'stage4_hard_negative_filtering',
        'stage5_cross_encoder_reranking',
        'stage6_final_output'
    ]

    def __init__(self, config: UMLSMappingConfig):
        self.config = config
        self.output_root = Path(config.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Pipeline status tracking
        self.status = PipelineStatus(self.output_root / '.pipeline_status.json')

        logger.info("=" * 70)
        logger.info("UMLS Mapping Pipeline Initialized")
        logger.info("=" * 70)
        logger.info(f"Output directory: {self.output_root}")

    def _setup_logging(self):
        """Setup centralized logging"""
        log_file = self.output_root / 'pipeline.log'

        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler(sys.stdout)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    def run(
        self,
        stages: Optional[List[str]] = None,
        resume: bool = False,
        force: bool = False
    ) -> bool:
        """
        Run pipeline

        Args:
            stages: List of stages to run (default: all)
            resume: Resume from last successful stage
            force: Force rerun even if completed

        Returns:
            True if successful, False otherwise
        """
        # Determine which stages to run
        if stages is None:
            stages_to_run = self.STAGES
        else:
            stages_to_run = [s for s in self.STAGES if s in stages]

        # Resume logic
        if resume and not force:
            last_stage = self.status.status.get('last_successful_stage')
            if last_stage and last_stage in self.STAGES:
                last_idx = self.STAGES.index(last_stage)
                stages_to_run = self.STAGES[last_idx + 1:]
                logger.info(f"Resuming from stage: {last_stage}")

        logger.info(f"Stages to run: {stages_to_run}")

        # Run stages
        for stage in stages_to_run:
            # Skip if completed (unless force)
            if not force and self.status.is_completed(stage):
                logger.info(f"✓ {stage} already completed (skipping)")
                continue

            logger.info("=" * 70)
            logger.info(f"Running: {stage}")
            logger.info("=" * 70)

            try:
                # Run stage
                success = self._run_stage(stage)

                if success:
                    self.status.mark_completed(stage)
                    logger.info(f"✓ {stage} completed successfully")
                else:
                    self.status.mark_failed(stage)
                    logger.error(f"✗ {stage} failed")
                    return False

            except Exception as e:
                logger.error(f"✗ {stage} failed with exception:")
                logger.error(traceback.format_exc())
                self.status.mark_failed(stage)
                return False

        logger.info("=" * 70)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)
        return True

    def _run_stage(self, stage: str) -> bool:
        """Run a single stage"""

        if stage == 'stage0_umls_loading':
            return self._run_stage0_umls_loading()

        elif stage == 'stage1_preprocessing':
            return self._run_stage1_preprocessing()

        elif stage == 'stage2_setup_sapbert':
            return self._run_stage2_setup_sapbert()

        elif stage == 'stage2_setup_tfidf':
            return self._run_stage2_setup_tfidf()

        elif stage == 'stage2_candidate_generation':
            return self._run_stage2_candidate_generation()

        elif stage == 'stage3_cluster_aggregation':
            return self._run_stage3_cluster_aggregation()

        elif stage == 'stage4_hard_negative_filtering':
            return self._run_stage4_hard_negative_filtering()

        elif stage == 'stage5_cross_encoder_reranking':
            return self._run_stage5_cross_encoder_reranking()

        elif stage == 'stage6_final_output':
            return self._run_stage6_final_output()

        else:
            logger.error(f"Unknown stage: {stage}")
            return False

    def _run_stage0_umls_loading(self) -> bool:
        """Stage 0: Load UMLS database"""
        logger.info("Loading UMLS database...")

        umls_loader = UMLSLoader(self.config)
        concepts = umls_loader.load_umls()

        logger.info(f"Loaded {len(concepts):,} UMLS concepts")
        return True

    def _run_stage1_preprocessing(self) -> bool:
        """Stage 1: Entity extraction and preprocessing"""
        logger.info("Running preprocessing...")

        preprocessor = Preprocessor(self.config)
        preprocessor.run()

        # Validate Stage 1 outputs
        logger.info("Validating Stage 1 outputs...")
        validator = Stage1Validator(
            preprocessing_dir=Path(self.config.output_root) / 'stage31_preprocessing',
            umls_cache_dir=Path(self.config.umls_cache_dir),
            kg_path=self.config.kg_clean_path
        )

        if not validator.validate_all():
            logger.error("Stage 1 validation failed")
            return False

        return True

    def _run_stage2_setup_sapbert(self) -> bool:
        """Stage 2 Setup: SapBERT embeddings + FAISS"""
        logger.info("Running SapBERT setup (this may take 2-3 hours)...")

        # Import and run task_2_1
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/task_2_1_sapbert_setup.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"SapBERT setup failed: {result.stderr}")
            return False

        logger.info(result.stdout)
        return True

    def _run_stage2_setup_tfidf(self) -> bool:
        """Stage 2 Setup: TF-IDF vectorizer"""
        logger.info("Running TF-IDF setup...")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/task_2_2_tfidf_setup.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"TF-IDF setup failed: {result.stderr}")
            return False

        logger.info(result.stdout)
        return True

    def _run_stage2_candidate_generation(self) -> bool:
        """Stage 2: Generate candidates"""
        logger.info("Generating candidates...")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/stage2_generate_candidates.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Candidate generation failed: {result.stderr}")
            return False

        logger.info(result.stdout)
        return True

    def _run_stage3_cluster_aggregation(self) -> bool:
        """Stage 3: Cluster aggregation"""
        logger.info("Running cluster aggregation...")

        # This is part of stages3_4_aggregate_filter.py
        # We'll extract just stage 3 logic
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/stages3_4_aggregate_filter.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Aggregation failed: {result.stderr}")
            return False

        return True

    def _run_stage4_hard_negative_filtering(self) -> bool:
        """Stage 4: Hard negative filtering"""
        # Already run in stage 3 script
        return True

    def _run_stage5_cross_encoder_reranking(self) -> bool:
        """Stage 5: Cross-encoder reranking"""
        logger.info("Running cross-encoder reranking...")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/stage5_rerank.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Reranking failed: {result.stderr}")
            return False

        return True

    def _run_stage6_final_output(self) -> bool:
        """Stage 6: Final output"""
        logger.info("Generating final output...")

        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/stage6_final_output.py'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Final output failed: {result.stderr}")
            return False

        # Run final validation
        logger.info("Running final validation...")
        result = subprocess.run(
            [sys.executable, 'scripts/final_validation.py'],
            capture_output=True,
            text=True
        )

        logger.info(result.stdout)

        return result.returncode == 0

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return self.status.status

    def reset(self):
        """Reset pipeline status"""
        self.status.reset()
        logger.info("Pipeline status reset")
