"""
Stage 1 Comprehensive Validation
Validates all outputs from Tasks 1.1-1.4
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


class Stage1Validator:
    """
    Comprehensive validator for Stage 1 outputs
    
    Validates:
    - Task 1.1: entities.txt
    - Task 1.2: synonym_clusters.json
    - Task 1.3: normalized_entities.json
    - Task 1.4: umls_concepts.pkl, umls_aliases.pkl, umls_stats.json
    """
    
    def __init__(self, preprocessing_dir: Path, umls_cache_dir: Path, kg_path: str = None):
        self.preprocessing_dir = Path(preprocessing_dir)
        self.umls_cache_dir = Path(umls_cache_dir)
        self.kg_path = kg_path
        
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        logger.info("=" * 70)
        logger.info("STAGE 1 COMPREHENSIVE VALIDATION")
        logger.info("=" * 70)
        
        self.errors = []
        self.warnings = []
        
        # Run all checks
        self._check_files_exist()
        self._validate_entities()
        self._validate_synonym_clusters()
        self._validate_normalized_entities()
        self._validate_umls_data()
        
        if self.kg_path:
            self._cross_validate_kg()
        
        # Print summary
        self._print_summary()
        
        return len(self.errors) == 0
    
    def _check_files_exist(self):
        """Check all required files exist with correct sizes"""
        logger.info("\n1. CHECKING FILES...")
        
        required_files = {
            # Task 1.1-1.3 outputs
            self.preprocessing_dir / 'entities.txt': (100*1024, 5*1024*1024),
            self.preprocessing_dir / 'synonym_clusters.json': (500*1024, 10*1024*1024),
            self.preprocessing_dir / 'normalized_entities.json': (200*1024, 5*1024*1024),
            # Task 1.4 outputs
            self.umls_cache_dir / 'umls_concepts.pkl': (2*1024**3, 5*1024**3),
            self.umls_cache_dir / 'umls_aliases.pkl': (1*1024**3, 3*1024**3),
            self.umls_cache_dir / 'umls_stats.json': (1024, 100*1024)
        }
        
        for filepath, (min_size, max_size) in required_files.items():
            if not filepath.exists():
                self.errors.append(f"File missing: {filepath}")
                logger.error(f"  ✗ {filepath}: MISSING")
            else:
                size = filepath.stat().st_size
                if size < min_size:
                    self.errors.append(f"File too small: {filepath} ({size} bytes)")
                    logger.error(f"  ✗ {filepath}: TOO SMALL ({size} bytes)")
                elif size > max_size:
                    self.warnings.append(f"File larger than expected: {filepath}")
                    logger.warning(f"  ⚠ {filepath}: LARGE ({size/(1024**2):.1f} MB)")
                else:
                    if size > 1024**3:
                        logger.info(f"  ✓ {filepath.name}: {size/(1024**3):.2f} GB")
                    elif size > 1024**2:
                        logger.info(f"  ✓ {filepath.name}: {size/(1024**2):.1f} MB")
                    else:
                        logger.info(f"  ✓ {filepath.name}: {size/1024:.1f} KB")
    
    def _validate_entities(self):
        """Validate entities.txt from Task 1.1"""
        logger.info("\n2. VALIDATING ENTITIES...")
        
        entities_file = self.preprocessing_dir / 'entities.txt'
        if not entities_file.exists():
            return
        
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = [line.strip() for line in f]
            
            logger.info(f"  ✓ Loaded {len(entities):,} entities")
            
            # Check duplicates
            if len(entities) != len(set(entities)):
                self.errors.append("Entities contain duplicates")
                logger.error("  ✗ Contains duplicates!")
            else:
                logger.info("  ✓ No duplicates")
            
            # Check empty
            empty_count = sum(1 for e in entities if not e.strip())
            if empty_count > 0:
                self.errors.append(f"{empty_count} empty entities")
                logger.error(f"  ✗ {empty_count} empty entities!")
            else:
                logger.info("  ✓ No empty entities")
            
            # Check sorted
            sorted_entities = sorted(entities, key=str.lower)
            if entities != sorted_entities:
                self.warnings.append("Entities not sorted")
                logger.warning("  ⚠ Not sorted alphabetically")
            else:
                logger.info("  ✓ Properly sorted")
            
            # Save for cross-validation
            self.entities = entities
            
        except Exception as e:
            self.errors.append(f"Cannot load entities: {e}")
            logger.error(f"  ✗ Error loading: {e}")
    
    def _validate_synonym_clusters(self):
        """Validate synonym_clusters.json from Task 1.2"""
        logger.info("\n3. VALIDATING SYNONYM CLUSTERS...")
        
        clusters_file = self.preprocessing_dir / 'synonym_clusters.json'
        if not clusters_file.exists():
            return
        
        try:
            with open(clusters_file, 'r', encoding='utf-8') as f:
                clusters = json.load(f)
            
            logger.info(f"  ✓ Loaded {len(clusters):,} clusters")
            
            # Check coverage
            entities_in_clusters = set()
            for members in clusters.values():
                for entity in members:
                    if entity in entities_in_clusters:
                        self.errors.append(f"Entity in multiple clusters: {entity}")
                    entities_in_clusters.add(entity)
            
            if hasattr(self, 'entities'):
                if entities_in_clusters != set(self.entities):
                    self.errors.append("Cluster coverage mismatch")
                    missing = set(self.entities) - entities_in_clusters
                    extra = entities_in_clusters - set(self.entities)
                    if missing:
                        logger.error(f"  ✗ {len(missing)} entities not in clusters")
                    if extra:
                        logger.error(f"  ✗ {len(extra)} unknown entities in clusters")
                else:
                    logger.info("  ✓ All entities covered")
            
            # Statistics
            sizes = [len(m) for m in clusters.values()]
            singleton = sum(1 for s in sizes if s == 1)
            singleton_rate = singleton / len(sizes)
            avg_size = sum(sizes) / len(sizes)
            
            logger.info(f"  ✓ Singleton rate: {singleton_rate:.1%}")
            logger.info(f"  ✓ Average size: {avg_size:.2f}")
            logger.info(f"  ✓ Max size: {max(sizes)}")
            
            if not (0.3 <= singleton_rate <= 0.8):
                self.warnings.append(f"Unusual singleton rate: {singleton_rate:.1%}")
            
            if not (1.2 <= avg_size <= 5.0):
                self.warnings.append(f"Unusual average size: {avg_size:.2f}")
            
        except Exception as e:
            self.errors.append(f"Cannot load clusters: {e}")
            logger.error(f"  ✗ Error loading: {e}")
    
    def _validate_normalized_entities(self):
        """Validate normalized_entities.json from Task 1.3"""
        logger.info("\n4. VALIDATING NORMALIZED ENTITIES...")
        
        normalized_file = self.preprocessing_dir / 'normalized_entities.json'
        if not normalized_file.exists():
            return
        
        try:
            with open(normalized_file, 'r', encoding='utf-8') as f:
                normalized = json.load(f)
            
            logger.info(f"  ✓ Loaded {len(normalized):,} normalized entities")
            
            # Check coverage
            if hasattr(self, 'entities'):
                if set(normalized.keys()) != set(self.entities):
                    self.errors.append("Normalized entities coverage mismatch")
                    logger.error("  ✗ Coverage mismatch!")
                else:
                    logger.info("  ✓ All entities covered")
            
            # Check structure
            required_keys = ['original', 'normalized', 'expanded']
            sample_count = min(10, len(normalized))
            for entity, data in list(normalized.items())[:sample_count]:
                for key in required_keys:
                    if key not in data:
                        self.errors.append(f"Missing key '{key}' for entity: {entity}")
                        break
            
            logger.info("  ✓ Structure correct")
            
            # Check expansion rate
            expanded_count = sum(1 for d in normalized.values() 
                               if d['normalized'] != d['expanded'])
            exp_rate = expanded_count / len(normalized) * 100
            
            logger.info(f"  ✓ Expansion rate: {exp_rate:.1f}%")
            
            if exp_rate < 3 or exp_rate > 50:
                self.warnings.append(f"Unusual expansion rate: {exp_rate:.1f}%")
            
        except Exception as e:
            self.errors.append(f"Cannot load normalized: {e}")
            logger.error(f"  ✗ Error loading: {e}")
    
    def _validate_umls_data(self):
        """Validate UMLS data from Task 1.4"""
        logger.info("\n5. VALIDATING UMLS DATA...")
        
        concepts_file = self.umls_cache_dir / 'umls_concepts.pkl'
        aliases_file = self.umls_cache_dir / 'umls_aliases.pkl'
        stats_file = self.umls_cache_dir / 'umls_stats.json'
        
        if not all([concepts_file.exists(), aliases_file.exists(), stats_file.exists()]):
            return
        
        try:
            # Load concepts
            with open(concepts_file, 'rb') as f:
                concepts = pickle.load(f)
            
            logger.info(f"  ✓ Loaded {len(concepts):,} concepts")
            
            # Load aliases
            with open(aliases_file, 'rb') as f:
                aliases = pickle.load(f)
            
            logger.info(f"  ✓ Loaded {len(aliases):,} aliases")
            
            # Load stats
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            logger.info("  ✓ Loaded stats")
            
            # Validate counts
            if len(concepts) < 4000000:
                self.errors.append(f"Too few concepts: {len(concepts):,}")
                logger.error("  ✗ Too few concepts!")
            else:
                logger.info("  ✓ Concepts count OK (>4M)")
            
            if len(aliases) < 10000000:
                self.errors.append(f"Too few aliases: {len(aliases):,}")
                logger.error("  ✗ Too few aliases!")
            else:
                logger.info("  ✓ Aliases count OK (>10M)")
            
            # Check structure (use dataclass attributes)
            sample_cui = list(concepts.keys())[0]
            sample_concept = concepts[sample_cui]
            
            required_attrs = ['cui', 'preferred_name', 'aliases', 'semantic_types', 'definitions']
            for attr in required_attrs:
                if not hasattr(sample_concept, attr):
                    self.errors.append(f"Missing attribute in concept: {attr}")
            
            logger.info("  ✓ Concept structure correct")
            
            # Check test CUI
            test_cui = 'C0011860'
            if test_cui not in concepts:
                self.errors.append("Test CUI C0011860 not found")
                logger.error("  ✗ Test CUI not found!")
            else:
                logger.info(f"  ✓ Test CUI found: {concepts[test_cui].preferred_name}")
            
            # Validate stats
            required_stats = ['total_concepts', 'total_aliases', 
                             'concepts_with_semantic_types', 'concepts_with_definitions']
            for stat in required_stats:
                if stat not in stats:
                    self.errors.append(f"Missing stat: {stat}")
            
            logger.info("  ✓ Stats complete")
            
        except Exception as e:
            self.errors.append(f"Cannot load UMLS data: {e}")
            logger.error(f"  ✗ Error loading: {e}")
    
    def _cross_validate_kg(self):
        """Cross-validate with KG file"""
        logger.info("\n6. CROSS-VALIDATION...")
        
        if not self.kg_path or not Path(self.kg_path).exists():
            self.warnings.append("KG file not provided for cross-validation")
            return
        
        try:
            kg_entities = set()
            with open(self.kg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) == 3:
                        kg_entities.add(parts[0].strip())
                        kg_entities.add(parts[2].strip())
            
            if hasattr(self, 'entities'):
                coverage = len(set(self.entities) & kg_entities) / len(kg_entities) * 100
                logger.info(f"  KG coverage: {coverage:.1f}%")
                
                if coverage < 99.0:
                    self.warnings.append(f"KG coverage low: {coverage:.1f}%")
                else:
                    logger.info("  ✓ KG coverage excellent")
        
        except Exception as e:
            self.warnings.append(f"Cannot validate KG coverage: {e}")
    
    def _print_summary(self):
        """Print validation summary"""
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\nFiles checked: 6/6")
        logger.info(f"Errors: {len(self.errors)}")
        logger.info(f"Warnings: {len(self.warnings)}")
        
        if self.errors:
            logger.error("\n❌ ERRORS FOUND:")
            for i, error in enumerate(self.errors, 1):
                logger.error(f"  {i}. {error}")
        
        if self.warnings:
            logger.warning("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                logger.warning(f"  {i}. {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("\n" + "=" * 70)
            logger.info("✅ STAGE 1 VALIDATION PASSED - PERFECT!")
            logger.info("=" * 70)
            logger.info("\n🎉 Stage 1 hoàn tất! Sẵn sàng chuyển sang Stage 2.")
        
        elif not self.errors:
            logger.info("\n" + "=" * 70)
            logger.info("✅ STAGE 1 VALIDATION PASSED - WITH WARNINGS")
            logger.info("=" * 70)
            logger.info("\n👍 Stage 1 passed nhưng có một số warnings không nghiêm trọng.")
            logger.info("Bạn có thể tiếp tục Stage 2.")
        
        else:
            logger.error("\n" + "=" * 70)
            logger.error("❌ STAGE 1 VALIDATION FAILED")
            logger.error("=" * 70)
            logger.error("\n⚠️  Vui lòng fix các errors trước khi sang Stage 2!")
