#!/bin/bash
# Download PrimeKG UMLS-MONDO mapping file

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PrimeKG UMLS-MONDO Mapping Downloader${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TARGET_DIR="${1:-./primekg_analysis}"
mkdir -p "$TARGET_DIR"

echo -e "${YELLOW}Target directory: $TARGET_DIR${NC}"
echo ""

# Check if file already exists
if [ -f "$TARGET_DIR/umls_mondo.csv" ]; then
    echo -e "${GREEN}✓ umls_mondo.csv already exists${NC}"
    echo -e "${YELLOW}Preview (first 5 lines):${NC}"
    head -5 "$TARGET_DIR/umls_mondo.csv"
    echo ""
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Using existing file."
        exit 0
    fi
fi

# Try direct download from raw GitHub
echo -e "${YELLOW}Attempting to download umls_mondo.csv from GitHub...${NC}"

GITHUB_RAW_URL="https://raw.githubusercontent.com/mims-harvard/PrimeKG/main/datasets/data/umls/umls_mondo.csv"

if wget -q --spider "$GITHUB_RAW_URL" 2>/dev/null; then
    echo -e "${GREEN}✓ URL accessible${NC}"
    wget -O "$TARGET_DIR/umls_mondo.csv" "$GITHUB_RAW_URL" 2>&1 | grep -E '(saved|failed|error)' || true

    if [ -f "$TARGET_DIR/umls_mondo.csv" ] && [ -s "$TARGET_DIR/umls_mondo.csv" ]; then
        echo -e "${GREEN}✓ Downloaded successfully!${NC}"

        # Show file info
        echo ""
        echo -e "${BLUE}File info:${NC}"
        ls -lh "$TARGET_DIR/umls_mondo.csv"

        # Show preview
        echo ""
        echo -e "${YELLOW}Preview (first 5 lines):${NC}"
        head -5 "$TARGET_DIR/umls_mondo.csv"

        # Count mappings
        echo ""
        LINE_COUNT=$(wc -l < "$TARGET_DIR/umls_mondo.csv")
        echo -e "${GREEN}Total mappings: $((LINE_COUNT - 1))${NC}"

        echo ""
        echo -e "${GREEN}✅ Download complete!${NC}"
        echo ""
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Use with converter:"
        echo "   python primekg_to_umls_triples.py kg.csv output.txt --mapping $TARGET_DIR/umls_mondo.csv"
        echo ""
        exit 0
    else
        echo -e "${RED}✗ Download failed or file is empty${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Direct download not available (network restrictions)${NC}"
fi

# Alternative: Clone repo
echo ""
echo -e "${YELLOW}Alternative: Clone PrimeKG repository...${NC}"
read -p "Clone full PrimeKG repo? (~100MB, y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$TARGET_DIR"

    if [ -d "PrimeKG" ]; then
        echo -e "${YELLOW}PrimeKG directory exists, pulling latest...${NC}"
        cd PrimeKG
        git pull
    else
        echo -e "${YELLOW}Cloning PrimeKG repository...${NC}"
        git clone https://github.com/mims-harvard/PrimeKG.git
        cd PrimeKG
    fi

    # Copy mapping file
    if [ -f "datasets/data/umls/umls_mondo.csv" ]; then
        cp datasets/data/umls/umls_mondo.csv ../
        echo -e "${GREEN}✓ Copied umls_mondo.csv${NC}"
        echo ""
        head -5 ../umls_mondo.csv
        echo ""
        echo -e "${GREEN}✅ Setup complete!${NC}"
    else
        echo -e "${RED}✗ Mapping file not found in repo${NC}"
        exit 1
    fi
else
    echo ""
    echo -e "${RED}Download cancelled.${NC}"
    echo ""
    echo -e "${YELLOW}Manual download options:${NC}"
    echo "1. Visit: https://github.com/mims-harvard/PrimeKG/tree/main/datasets/data/umls"
    echo "2. Download umls_mondo.csv manually"
    echo "3. Save to: $TARGET_DIR/umls_mondo.csv"
    echo ""
    echo "Or use filter strategy (no mapping needed):"
    echo "  python primekg_to_umls_triples.py kg.csv output.txt --strategy filter"
    echo ""
    exit 1
fi
