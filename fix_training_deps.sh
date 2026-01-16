#!/bin/bash
# Fix dependency conflicts for Stage 3 training
# This script resolves version conflicts between gfmrag and training dependencies

echo "=========================================="
echo "Fixing Stage 3 Training Dependencies"
echo "=========================================="
echo ""
echo "⚠️  This will upgrade some packages to resolve conflicts"
echo ""

# Uninstall conflicting versions
echo "Step 1: Uninstalling old conflicting packages..."
pip uninstall -y transformers huggingface-hub accelerate numpy

# Reinstall with correct versions
echo ""
echo "Step 2: Installing compatible versions..."
pip install "transformers>=4.46.1" \
            "huggingface-hub>=0.20.0" \
            "numpy>=1.24.0,<2.0.0" \
            "accelerate>=0.20.0"

# Install remaining training dependencies
echo ""
echo "Step 3: Installing training dependencies..."
pip install datasets>=2.14.0 \
            tensorboard>=2.14.0 \
            scikit-learn>=1.3.0 \
            matplotlib>=3.7.0 \
            seaborn>=0.12.0 \
            scipy>=1.10.0 \
            faiss-cpu

echo ""
echo "=========================================="
echo "✅ Dependency fix complete!"
echo "=========================================="
echo ""
echo "Verify installation:"
echo "  python -c 'import transformers; print(\"transformers:\", transformers.__version__)'"
echo "  python -c 'import tensorboard; print(\"tensorboard:\", tensorboard.__version__)'"
echo ""
echo "Now run training:"
echo "  python -m gfmrag.umls_mapping.training.cross_encoder_trainer"
