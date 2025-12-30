#!/bin/bash
################################################################################
# Quick UMLS Download Script (RRF files only)
#
# This script downloads UMLS and extracts only the 3 required RRF files,
# saving disk space and time.
#
# Usage:
#   export UTS_USER="your_username"
#   export UTS_PASS="your_password"
#   bash scripts/quick_download_umls.sh
#
# Or:
#   bash scripts/quick_download_umls.sh
#   (will prompt for credentials)
################################################################################

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
UMLS_DIR="$PROJECT_ROOT/data/umls"
UMLS_VERSION="2024AA"
DOWNLOAD_URL="https://download.nlm.nih.gov/umls/kss/$UMLS_VERSION/umls-$UMLS_VERSION-full.zip"

# Get credentials
if [ -z "$UTS_USER" ] || [ -z "$UTS_PASS" ]; then
    echo "Enter your UTS credentials (https://uts.nlm.nih.gov)"
    read -p "Username: " UTS_USER
    read -sp "Password: " UTS_PASS
    echo ""
fi

# Create directory
mkdir -p "$UMLS_DIR"
cd "$UMLS_DIR"

echo "📥 Downloading UMLS $UMLS_VERSION..."

# Download
curl -u "$UTS_USER:$UTS_PASS" \
     -C - \
     --progress-bar \
     -o "umls-$UMLS_VERSION-full.zip" \
     "$DOWNLOAD_URL"

echo "✅ Download complete"
echo "📦 Extracting RRF files only (this saves time)..."

# Extract only the required RRF files
unzip -j "umls-$UMLS_VERSION-full.zip" "*/META/MRCONSO.RRF" -d . 2>/dev/null || \
    (unzip -q "umls-$UMLS_VERSION-full.zip" && \
     find . -name "MRCONSO.RRF" -exec cp {} . \;)

unzip -j "umls-$UMLS_VERSION-full.zip" "*/META/MRSTY.RRF" -d . 2>/dev/null || \
    find . -name "MRSTY.RRF" -exec cp {} . \;

unzip -j "umls-$UMLS_VERSION-full.zip" "*/META/MRDEF.RRF" -d . 2>/dev/null || \
    find . -name "MRDEF.RRF" -exec cp {} . \;

echo "✅ Extraction complete"

# Cleanup
echo "🧹 Cleaning up..."
rm -f "umls-$UMLS_VERSION-full.zip"
find . -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null

# Verify
echo ""
echo "✅ Files ready:"
ls -lh MRCONSO.RRF MRSTY.RRF MRDEF.RRF 2>/dev/null || echo "❌ Some files missing"

echo ""
echo "📍 Location: $UMLS_DIR"
echo ""
echo "Next: Run pipeline with:"
echo "  cd $PROJECT_ROOT"
echo "  python run_umls_pipeline.py"
