#!/bin/bash
#
# Quick test script to verify Stage 2 configuration and paths
#

set -e

echo "🔍 Stage 2 Configuration Check"
echo "======================================"
echo ""

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ No conda environment active!"
    echo "   Please run: conda activate gfm-rag"
    exit 1
else
    echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
fi

echo ""

# Check Python and key packages
echo "🐍 Python Environment:"
echo "   Python: $(python --version)"
python -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>&1 || echo "   ❌ PyTorch not found"
python -c "import transformers; print(f'   Transformers: {transformers.__version__}')" 2>&1 || echo "   ❌ Transformers not found"
python -c "import hydra; print(f'   Hydra: {hydra.__version__}')" 2>&1 || echo "   ❌ Hydra not found"

echo ""

# Check input files
echo "📂 Input Files:"
if [ -f "./data/hotpotqa/processed/stage1/kg.txt" ]; then
    SIZE=$(du -h ./data/hotpotqa/processed/stage1/kg.txt | cut -f1)
    LINES=$(wc -l < ./data/hotpotqa/processed/stage1/kg.txt)
    echo "   ✅ kg.txt exists ($SIZE, $LINES triples)"
else
    echo "   ❌ kg.txt not found at: ./data/hotpotqa/processed/stage1/kg.txt"
    echo "      Run Stage 1 first: bash run_stage1.sh"
    exit 1
fi

echo ""

# Check output directory
echo "📂 Output Directory:"
if [ -d "./tmp/entity_resolution" ]; then
    echo "   ✅ Output dir exists: ./tmp/entity_resolution"
    echo "   Files:"
    ls -lh ./tmp/entity_resolution/ 2>&1 | tail -10 || echo "   (empty)"
else
    echo "   ℹ️  Output dir will be created: ./tmp/entity_resolution"
fi

echo ""

# Check config file
echo "⚙️  Configuration:"
if [ -f "./gfmrag/workflow/config/stage2_entity_resolution.yaml" ]; then
    echo "   ✅ Config file exists"
    echo "   Key settings:"
    grep "kg_input_path:" ./gfmrag/workflow/config/stage2_entity_resolution.yaml
    grep "device:" ./gfmrag/workflow/config/stage2_entity_resolution.yaml | head -1
    grep "k_neighbors:" ./gfmrag/workflow/config/stage2_entity_resolution.yaml
else
    echo "   ❌ Config file not found"
    exit 1
fi

echo ""
echo "======================================"
echo "✅ All checks passed!"
echo ""
echo "Ready to run Stage 2:"
echo "  python -m gfmrag.workflow.stage2_entity_resolution"
echo ""
