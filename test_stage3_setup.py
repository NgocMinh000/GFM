#!/usr/bin/env python3
"""
Test script to verify Stage 3 UMLS Mapping setup
Checks that all components are importable and UMLS data is ready
"""

import sys
from pathlib import Path

def check_imports():
    """Check all required imports"""
    print("=" * 80)
    print("Checking imports...")
    print("=" * 80)

    try:
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
        )
        print("✓ All core components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_umls_data():
    """Check UMLS data files"""
    print("\n" + "=" * 80)
    print("Checking UMLS data files...")
    print("=" * 80)

    umls_dir = Path("data/umls/META")
    required_files = ["MRCONSO.RRF", "MRSTY.RRF"]
    optional_files = ["MRDEF.RRF"]

    all_good = True

    for filename in required_files:
        filepath = umls_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename}: {size_mb:.1f} MB")
        else:
            print(f"✗ {filename}: NOT FOUND")
            all_good = False

    for filename in optional_files:
        filepath = umls_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ {filename} (optional): {size_mb:.1f} MB")
        else:
            print(f"⚠  {filename} (optional): Not found - pipeline will skip definitions")

    if not all_good:
        print("\n" + "!" * 80)
        print("MISSING REQUIRED UMLS FILES!")
        print("!" * 80)
        print("\nPlease download UMLS Metathesaurus from:")
        print("https://www.nlm.nih.gov/research/umls/")
        print("\nExtract and place files in: data/umls/META/")
        print("Required files:")
        print("  - MRCONSO.RRF (~5GB)")
        print("  - MRSTY.RRF (~100MB)")
        print("\nSee STAGE3_UMLS_MAPPING_README.md for details")

    return all_good

def check_dependencies():
    """Check Python dependencies"""
    print("\n" + "=" * 80)
    print("Checking dependencies...")
    print("=" * 80)

    required_packages = [
        "sentence_transformers",
        "sklearn",
        "faiss",
        "tqdm",
        "hydra",
        "omegaconf",
    ]

    optional_packages = [
        "matplotlib",
        "seaborn",
    ]

    all_good = True

    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}: NOT INSTALLED")
            all_good = False

    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"⚠  {package} (optional): Not installed - visualizations will be skipped")

    if not all_good:
        print("\n" + "!" * 80)
        print("MISSING REQUIRED PACKAGES!")
        print("!" * 80)
        print("\nInstall with:")
        print("pip install sentence-transformers scikit-learn faiss-cpu tqdm hydra-core")

    return all_good

def check_stage2_output():
    """Check if Stage 2 output exists"""
    print("\n" + "=" * 80)
    print("Checking Stage 2 output...")
    print("=" * 80)

    # Check for kg_clean.txt in tmp/entity_resolution/
    kg_path = Path("tmp/entity_resolution/kg_clean.txt")

    if kg_path.exists():
        size_kb = kg_path.stat().st_size / 1024
        print(f"✓ Found kg_clean.txt: {kg_path} ({size_kb:.1f} KB)")

        # Check format (should be comma-separated)
        with open(kg_path, 'r') as f:
            first_line = f.readline().strip()
            if ',' in first_line and first_line.count(',') >= 2:
                print(f"✓ Format looks correct (comma-separated)")
                print(f"  Sample: {first_line[:80]}...")
            else:
                print(f"⚠  Warning: Format might be incorrect")
                print(f"  Expected: entity1,relation,entity2")
                print(f"  Found: {first_line[:80]}...")
        return True
    else:
        print(f"✗ kg_clean.txt not found at: {kg_path}")
        print("\nStage 3 requires output from Stage 2 Entity Resolution")
        print("Expected path: tmp/entity_resolution/kg_clean.txt")
        print("Expected format: copper,is a,transition metal (comma-separated)")
        print("\nRun Stage 2 first:")
        print("  python -m gfmrag.workflow.stage2_entity_resolution stage=2")
        return False

def check_config():
    """Check config file"""
    print("\n" + "=" * 80)
    print("Checking config file...")
    print("=" * 80)

    config_path = Path("gfmrag/workflow/config/stage3_umls_mapping.yaml")
    if config_path.exists():
        print(f"✓ Config file exists: {config_path}")
        return True
    else:
        print(f"✗ Config file not found: {config_path}")
        return False

def main():
    print("\n")
    print("=" * 80)
    print("Stage 3 UMLS Mapping - Setup Verification")
    print("=" * 80)
    print()

    results = {
        "Imports": check_imports(),
        "Dependencies": check_dependencies(),
        "Config": check_config(),
        "UMLS Data": check_umls_data(),
        "Stage 2 Output": check_stage2_output(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")

    print()

    if all(results.values()):
        print("🎉 All checks passed! Ready to run Stage 3 pipeline.")
        print("\nRun with:")
        print("  python -m gfmrag.workflow.stage3_umls_mapping")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\nSee STAGE3_UMLS_MAPPING_README.md for setup instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
