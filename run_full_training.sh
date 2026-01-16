#!/bin/bash
# Complete End-to-End Training Pipeline
# Runs all 3 days: Data prep → Training → Calibration → Testing
#
# Usage:
#   ./run_full_training.sh

set -e  # Exit on error

echo "================================================================================"
echo "PHASE 2 - COMPLETE TRAINING PIPELINE"
echo "================================================================================"
echo ""

# Configuration
MODEL_NAME="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
CONFIG_FILE="gfmrag/umls_mapping/training/config/training_config.yaml"
OUTPUT_DIR="models/cross_encoder_finetuned"
CALIBRATION_DIR="models/calibration"
RESULTS_DIR="results/phase2"

# Data paths
MEDMENTIONS_PATH="data/MedMentions/full"
FAISS_INDEX="tmp/umls_faiss_index"
SPLITS_CACHE="tmp/training/medmentions_splits.pkl"

# Check prerequisites
echo "[Prerequisites Check]"
echo "Checking environment..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Please activate your conda environment."
    exit 1
fi
echo "✓ Python found: $(python --version)"

# Check CUDA (optional but recommended)
if command -v nvidia-smi &> /dev/null; then
    echo "✓ CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "⚠ CUDA not available - training will be VERY slow on CPU"
fi

# Check required files
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found"

if [ ! -d "$MEDMENTIONS_PATH" ]; then
    echo "ERROR: MedMentions data not found at: $MEDMENTIONS_PATH"
    echo "Please download MedMentions dataset first (see TRAINING_GUIDE.md)"
    exit 1
fi
echo "✓ MedMentions data found"

if [ ! -d "$FAISS_INDEX" ]; then
    echo "ERROR: FAISS index not found at: $FAISS_INDEX"
    echo "Please build FAISS index first (see TRAINING_GUIDE.md Step 3)"
    exit 1
fi
echo "✓ FAISS index found"

echo ""
echo "================================================================================"
echo "DAY 1: DATA PREPARATION"
echo "================================================================================"
echo ""

# Data preparation is handled automatically by the trainer
# Just verify that data loader works
echo "Testing data loader..."
python -c "
from gfmrag.umls_mapping.training.data_loader import load_medmentions
from gfmrag.umls_mapping.umls_loader import UMLSLoader

print('Loading UMLS...')
umls = UMLSLoader()
print(f'UMLS concepts: {len(umls.concepts):,}')

print('Testing MedMentions loader...')
train, val, test = load_medmentions(
    data_path='$MEDMENTIONS_PATH',
    umls_loader=umls,
    cache_path='$SPLITS_CACHE',
)
print(f'✓ Train: {len(train):,}')
print(f'✓ Val:   {len(val):,}')
print(f'✓ Test:  {len(test):,}')
"

echo ""
echo "✓ Day 1 complete - Data preparation ready"
echo ""

echo "================================================================================"
echo "DAY 2: MODEL TRAINING"
echo "================================================================================"
echo ""

echo "Starting training..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo ""
echo "This will take ~6 hours on V100 GPU (3-5 days on CPU)"
echo ""

# Run training
python -m gfmrag.umls_mapping.training.cross_encoder_trainer \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR"

# Check if training succeeded
if [ ! -f "$OUTPUT_DIR/checkpoint-best/pytorch_model.bin" ]; then
    echo "ERROR: Training failed - checkpoint not found"
    exit 1
fi

echo ""
echo "✓ Day 2 complete - Model trained successfully"
echo ""

# Evaluate on test set
echo "Evaluating model on test set..."
python -m gfmrag.umls_mapping.training.evaluate \
    --model_path "$OUTPUT_DIR/checkpoint-best" \
    --test_data "$SPLITS_CACHE" \
    --output_dir "$RESULTS_DIR/evaluation"

echo ""

echo "================================================================================"
echo "DAY 3: CALIBRATION & INTEGRATION"
echo "================================================================================"
echo ""

echo "Step 1: Calibration and threshold tuning..."
python -m gfmrag.umls_mapping.training.calibrate_and_tune \
    --model_path "$OUTPUT_DIR/checkpoint-best" \
    --val_data "$SPLITS_CACHE" \
    --output_dir "$CALIBRATION_DIR" \
    --calibration_method compare \
    --threshold_objective f1

# Check if calibration succeeded
if [ ! -f "$CALIBRATION_DIR/adaptive_tuner.pkl" ]; then
    echo "ERROR: Calibration failed - tuner not found"
    exit 1
fi

echo ""
echo "✓ Calibration complete"
echo ""

echo "Step 2: Integration testing..."
python -m gfmrag.umls_mapping.training.integration_test \
    --model_path "$OUTPUT_DIR/checkpoint-best" \
    --calibrator_path "$CALIBRATION_DIR/platt_scaler.pkl" \
    --threshold_tuner_path "$CALIBRATION_DIR/adaptive_tuner.pkl" \
    --output "$RESULTS_DIR/integration_test.json"

echo ""
echo "✓ Day 3 complete - Integration tested successfully"
echo ""

echo "================================================================================"
echo "PHASE 2 COMPLETE! 🎉"
echo "================================================================================"
echo ""
echo "Generated artifacts:"
echo "  Model:           $OUTPUT_DIR/checkpoint-best/"
echo "  Calibrator:      $CALIBRATION_DIR/platt_scaler.pkl"
echo "  Thresholds:      $CALIBRATION_DIR/adaptive_tuner.pkl"
echo "  Evaluation:      $RESULTS_DIR/evaluation/"
echo "  Integration:     $RESULTS_DIR/integration_test.json"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. Check evaluation plots: $RESULTS_DIR/evaluation/*.png"
echo "  3. Integrate into production (see DAY3_INTEGRATION_GUIDE.md)"
echo ""
echo "To use in production:"
echo "  Update your Stage 3 pipeline to use FineTunedCrossEncoderReranker"
echo "  with paths to the generated artifacts above."
echo ""
