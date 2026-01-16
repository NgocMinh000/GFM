#!/bin/bash
# Quick-start script for Stage 3 UMLS Mapping
# Usage: bash run_stage3_umls_mapping.sh

set -e  # Exit on error

echo "========================================="
echo "Stage 3: UMLS Mapping Pipeline"
echo "========================================="
echo ""

# Check setup first
echo "Step 1: Verifying setup..."
python test_stage3_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup verification failed!"
    echo "Please fix the issues above before running the pipeline."
    echo "See STAGE3_UMLS_MAPPING_README.md for setup instructions."
    exit 1
fi

echo ""
echo "========================================="
echo "Step 2: Running UMLS Mapping Pipeline..."
echo "========================================="
echo ""

# Run the pipeline
python -m gfmrag.workflow.stage3_umls_mapping "$@"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ Stage 3 Complete!"
    echo "========================================="
    echo ""
    echo "Output files:"
    echo "  - tmp/umls_mapping/final_umls_mappings.json"
    echo "  - tmp/umls_mapping/umls_mapping_triples.txt"
    echo "  - tmp/umls_mapping/pipeline_metrics.json"
    echo ""
    echo "Add triples to KG:"
    echo "  cat tmp/umls_mapping/umls_mapping_triples.txt >> tmp/kg_construction/*/hotpotqa/kg_final.txt"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed!"
    echo "Check logs above for errors."
    exit 1
fi
