#!/bin/bash
#
# Script to run Stage 1: KG Construction
# Activates conda environment and runs Stage 1 workflow
#

set -e  # Exit on error

echo "=================================================="
echo "🚀 Starting Stage 1: Knowledge Graph Construction"
echo "=================================================="
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your API credentials:"
    echo "  cp .env.example .env"
    echo "  nano .env  # Edit and add your API keys"
    exit 1
fi

# Check if API key is configured
if grep -q "your-yescale-api-key-here" .env || grep -q "your-.*-key-here" .env; then
    echo "⚠️  Warning: .env file contains placeholder values"
    echo "Please edit .env and add your actual API keys"
    exit 1
fi

echo "✅ Found .env file with API credentials"
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

# Run Stage 1
echo "⚙️  Running Stage 1 workflow..."
echo "   Input: ./data/hotpotqa/raw/dataset_corpus.json"
echo "   Output: ./data/hotpotqa/processed/stage1/kg.txt"
echo ""

python -m gfmrag.workflow.stage1_index_dataset

echo ""
echo "=================================================="
echo "✅ Stage 1 Completed Successfully!"
echo "=================================================="
echo ""
echo "📊 Output files:"
echo "   • ./data/hotpotqa/processed/stage1/kg.txt"
echo "   • ./data/hotpotqa/processed/stage1/document2entities.json"
echo ""
echo "➡️  Next step: Run Stage 2 entity resolution"
echo "   bash run_stage2.sh"
echo ""
