#!/bin/bash
#
# Script to run Stage 2: Entity Resolution
# Activates conda environment and runs Stage 2 workflow
#

set -e  # Exit on error

echo "======================================================"
echo "🚀 Starting Stage 2: Advanced Entity Resolution"
echo "======================================================"
echo ""

# Check if Stage 1 output exists
if [ ! -f "./data/hotpotqa/processed/stage1/kg.txt" ]; then
    echo "❌ Error: Stage 1 output not found!"
    echo "Please run Stage 1 first:"
    echo "  bash run_stage1.sh"
    exit 1
fi

echo "✅ Found Stage 1 output: kg.txt"
echo ""

# Activate conda environment
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Conda environment already active: $CONDA_DEFAULT_ENV"
else
    echo "🔧 Activating conda environment: gfm-rag"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gfm-rag
fi

echo ""
echo "📂 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"
echo ""

# Run Stage 2
echo "⚙️  Running Stage 2 workflow..."
echo "   Input: ./data/hotpotqa/processed/stage1/kg.txt"
echo "   Output: ./tmp/entity_resolution/kg_clean.txt"
echo ""
echo "Pipeline stages:"
echo "   [0] Type Inference"
echo "   [1] SapBERT Embedding"
echo "   [2] FAISS Blocking"
echo "   [3] Multi-Feature Scoring"
echo "   [4] Adaptive Thresholding"
echo "   [5] Clustering & Canonicalization"
echo ""

python -m gfmrag.workflow.stage2_entity_resolution

echo ""
echo "======================================================"
echo "✅ Stage 2 Completed Successfully!"
echo "======================================================"
echo ""
echo "📊 Output files:"
echo "   • ./tmp/entity_resolution/kg_clean.txt (cleaned KG with SYNONYM_OF edges)"
echo "   • ./tmp/entity_resolution/stage0_types.json"
echo "   • ./tmp/entity_resolution/stage1_embeddings.npy"
echo "   • ./tmp/entity_resolution/stage2_candidates.json"
echo "   • ./tmp/entity_resolution/stage3_scores.json"
echo "   • ./tmp/entity_resolution/stage4_equivalents.json"
echo "   • ./tmp/entity_resolution/stage5_clusters.json"
echo ""
echo "🎉 Entity resolution pipeline completed!"
echo ""
