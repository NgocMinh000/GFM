#!/usr/bin/env python3
"""
Test script to verify visualization setup and show what will be plotted

Usage:
    python check_visualization.py
"""

import sys
from pathlib import Path

def check_matplotlib():
    """Check if matplotlib/seaborn are installed"""
    print("=" * 80)
    print("Checking Visualization Dependencies")
    print("=" * 80)

    missing = []

    # Check matplotlib
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__}")
    except ImportError:
        print("✗ matplotlib NOT INSTALLED")
        missing.append("matplotlib")

    # Check seaborn
    try:
        import seaborn
        print(f"✓ seaborn {seaborn.__version__}")
    except ImportError:
        print("✗ seaborn NOT INSTALLED")
        missing.append("seaborn")

    # Check numpy
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
    except ImportError:
        print("✗ numpy NOT INSTALLED")
        missing.append("numpy")

    if missing:
        print("\n" + "!" * 80)
        print("MISSING DEPENDENCIES")
        print("!" * 80)
        print(f"\nPlease install: {', '.join(missing)}")
        print("\nInstall command:")
        print("  pip install matplotlib seaborn numpy")
        print("\nOr with conda:")
        print("  conda install matplotlib seaborn numpy")
        return False

    print("\n✅ All visualization dependencies installed!")
    return True


def show_stage2_plots():
    """Show what Stage 2 plots will be generated"""
    print("\n" + "=" * 80)
    print("Stage 2 Entity Resolution - Visualization Details")
    print("=" * 80)

    plots = {
        "type_distribution.png": {
            "type": "Bar Chart",
            "data": "Entity types from stage0_entity_types.json",
            "shows": "Number of entities per type (drug, disease, procedure, etc.)",
            "x_axis": "Entity Type",
            "y_axis": "Count",
            "example": "drug: 200, disease: 150, procedure: 100"
        },
        "tier_distribution.png": {
            "type": "Bar Chart + Pie Chart",
            "data": "Tier information from stage0_entity_types.json",
            "shows": "Distribution across 3 tiers (Tier 1: Keywords, Tier 2: SapBERT, Tier 3: LLM)",
            "x_axis": "Tier",
            "y_axis": "Count / Percentage",
            "example": "Tier 1: 60%, Tier 2: 30%, Tier 3: 10%"
        },
        "confidence_distribution.png": {
            "type": "Histogram + Box Plot",
            "data": "Confidence scores from stage0_entity_types.json",
            "shows": "Distribution of confidence scores with mean/median",
            "x_axis": "Confidence Score (0-1)",
            "y_axis": "Frequency",
            "example": "Mean: 0.82, Median: 0.85"
        },
        "cluster_size_distribution.png": {
            "type": "Histogram + Statistics Panel",
            "data": "Cluster information from stage1b_synonym_clusters.json",
            "shows": "Size distribution of synonym clusters",
            "x_axis": "Cluster Size",
            "y_axis": "Frequency",
            "example": "500 singletons, 20 clusters of size 2-5"
        },
        "embedding_similarity_heatmap.png": {
            "type": "Heatmap (50x50)",
            "data": "SapBERT embeddings from stage1_embeddings.npy",
            "shows": "Cosine similarity between entity embeddings",
            "colors": "Red (high similarity) to Blue (low similarity)",
            "example": "Shows entity grouping quality"
        }
    }

    for i, (filename, details) in enumerate(plots.items(), 1):
        print(f"\n{i}. {filename}")
        print(f"   Type: {details['type']}")
        print(f"   Data: {details['data']}")
        print(f"   Shows: {details['shows']}")
        if 'x_axis' in details:
            print(f"   X-axis: {details['x_axis']}")
            print(f"   Y-axis: {details['y_axis']}")
        elif 'colors' in details:
            print(f"   Colors: {details['colors']}")
        print(f"   Example: {details['example']}")

    print(f"\n📁 Output location: tmp/entity_resolution/visualizations/")


def show_stage3_plots():
    """Show what Stage 3 plots will be generated"""
    print("\n" + "=" * 80)
    print("Stage 3 UMLS Mapping - Visualization Details")
    print("=" * 80)

    plots = {
        "candidate_reduction_funnel.png": {
            "type": "Funnel Chart",
            "data": "Candidate counts from pipeline_metrics.json",
            "shows": "Candidate reduction: Stage 3.2 (128) → 3.3 (64) → 3.4 (32)",
            "purpose": "Understand filtering process"
        },
        "confidence_distribution.png": {
            "type": "Histogram",
            "data": "Final confidence scores from final_umls_mappings.json",
            "shows": "Distribution across High/Medium/Low confidence tiers",
            "purpose": "Assess mapping quality"
        },
        "score_progression.png": {
            "type": "Line Chart",
            "data": "Scores across stages from pipeline_metrics.json",
            "shows": "How scores evolve from Stage 3.2 → 3.6",
            "purpose": "Evaluate refinement effectiveness"
        },
        "stage_timing.png": {
            "type": "Bar Chart",
            "data": "Stage durations from pipeline_metrics.json",
            "shows": "Time taken by each stage",
            "purpose": "Identify bottlenecks"
        },
        "cluster_statistics.png": {
            "type": "Histogram",
            "data": "Cluster sizes from preprocessing",
            "shows": "Distribution of synonym cluster sizes",
            "purpose": "Check clustering quality"
        },
        "metric_heatmap.png": {
            "type": "Heatmap",
            "data": "All metrics from pipeline_metrics.json",
            "shows": "Overview of all pipeline metrics",
            "purpose": "Quick health check"
        },
        "quality_metrics.png": {
            "type": "Multi-panel Chart",
            "data": "Quality metrics (if gold standard available)",
            "shows": "Precision, recall, confidence calibration",
            "purpose": "Evaluation against ground truth"
        }
    }

    for i, (filename, details) in enumerate(plots.items(), 1):
        print(f"\n{i}. {filename}")
        print(f"   Type: {details['type']}")
        print(f"   Data: {details['data']}")
        print(f"   Shows: {details['shows']}")
        print(f"   Purpose: {details['purpose']}")

    print(f"\n📁 Output location: tmp/umls_mapping/visualizations/")


def check_output_files():
    """Check if required output files exist"""
    print("\n" + "=" * 80)
    print("Checking Required Output Files")
    print("=" * 80)

    # Stage 2 files
    print("\nStage 2 Entity Resolution:")
    stage2_files = [
        "tmp/entity_resolution/stage0_entity_types.json",
        "tmp/entity_resolution/stage1_embeddings.npy",
        "tmp/entity_resolution/stage1_entity_ids.json",
        "tmp/entity_resolution/stage1b_synonym_clusters.json"
    ]

    stage2_exists = 0
    for filepath in stage2_files:
        path = Path(filepath)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {filepath} ({size_kb:.1f} KB)")
            stage2_exists += 1
        else:
            print(f"  ✗ {filepath} (not found)")

    # Stage 3 files
    print("\nStage 3 UMLS Mapping:")
    stage3_files = [
        "tmp/umls_mapping/pipeline_metrics.json",
        "tmp/umls_mapping/final_umls_mappings.json"
    ]

    stage3_exists = 0
    for filepath in stage3_files:
        path = Path(filepath)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  ✓ {filepath} ({size_kb:.1f} KB)")
            stage3_exists += 1
        else:
            print(f"  ✗ {filepath} (not found)")

    print(f"\n📊 Stage 2: {stage2_exists}/{len(stage2_files)} files found")
    print(f"📊 Stage 3: {stage3_exists}/{len(stage3_files)} files found")

    if stage2_exists == 0:
        print("\n⚠️  Run Stage 2 first: python -m gfmrag.workflow.stage2_entity_resolution")

    if stage3_exists == 0:
        print("\n⚠️  Run Stage 3 first: python -m gfmrag.workflow.stage3_umls_mapping")


def main():
    print("\n")
    print("=" * 80)
    print("VISUALIZATION CHECKER - Stage 2 & Stage 3")
    print("=" * 80)
    print()

    # 1. Check dependencies
    has_deps = check_matplotlib()

    # 2. Show what will be plotted
    show_stage2_plots()
    show_stage3_plots()

    # 3. Check output files
    check_output_files()

    # 4. Summary
    print("\n" + "=" * 80)
    print("SUMMARY & NEXT STEPS")
    print("=" * 80)

    if not has_deps:
        print("\n1. Install visualization dependencies:")
        print("   pip install matplotlib seaborn numpy")
        print("\n2. Run pipelines:")
        print("   python -m gfmrag.workflow.stage2_entity_resolution")
        print("   python -m gfmrag.workflow.stage3_umls_mapping")
        print("\n3. Check visualizations:")
        print("   ls tmp/entity_resolution/visualizations/")
        print("   ls tmp/umls_mapping/visualizations/")
    else:
        print("\n✅ Dependencies installed!")
        print("\nAfter running pipelines, visualizations will be at:")
        print("  - tmp/entity_resolution/visualizations/")
        print("  - tmp/umls_mapping/visualizations/")

    print()


if __name__ == "__main__":
    main()
