#!/usr/bin/env python3
"""
Standalone validation script for Stage 1
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from gfmrag.umls_mapping.validation import Stage1Validator
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

def main():
    """Run Stage 1 validation"""
    
    # Configure paths (adjust these to match your setup)
    preprocessing_dir = Path('./tmp/umls_mapping/stage31_preprocessing')
    umls_cache_dir = Path('./tmp/umls_mapping/umls_cache')  # Or wherever UMLS is cached
    kg_path = './data/kg_clean.txt'  # Optional
    
    # Create validator
    validator = Stage1Validator(
        preprocessing_dir=preprocessing_dir,
        umls_cache_dir=umls_cache_dir,
        kg_path=kg_path if Path(kg_path).exists() else None
    )
    
    # Run validation
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
