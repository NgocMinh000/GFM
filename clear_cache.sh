#!/bin/bash
# Script: clear_cache.sh - Xóa cache của GFM-RAG workflow

set -e  # Exit on error

echo "================================================"
echo "GFM-RAG Cache Clearing Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/user/GFM"

# Cache directories
KG_CACHE="$BASE_DIR/gfmrag/workflow/tmp/kg_construction"
QA_CACHE="$BASE_DIR/gfmrag/workflow/tmp/qa_construction"
TMP_DIR="$BASE_DIR/tmp"
OUTPUT_DIR="$BASE_DIR/gfmrag/workflow/outputs"

# Function to remove directory
remove_dir() {
    local dir=$1
    local name=$2

    if [ -d "$dir" ]; then
        echo -e "${YELLOW}Found:${NC} $name"
        echo "  Location: $dir"

        # Count files
        file_count=$(find "$dir" -type f | wc -l)
        dir_size=$(du -sh "$dir" 2>/dev/null | cut -f1)

        echo "  Files: $file_count"
        echo "  Size: $dir_size"

        read -p "  Remove this cache? (y/N): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$dir"
            echo -e "${GREEN}✓ Removed${NC}"
        else
            echo -e "${YELLOW}⊘ Skipped${NC}"
        fi
        echo ""
    else
        echo -e "${YELLOW}⊘ Not found:${NC} $name ($dir)"
        echo ""
    fi
}

# Function to clean pycache
clean_pycache() {
    echo -e "${YELLOW}Cleaning Python cache...${NC}"

    pycache_count=$(find "$BASE_DIR" -type d -name "__pycache__" | wc -l)

    if [ $pycache_count -gt 0 ]; then
        echo "  Found $pycache_count __pycache__ directories"
        read -p "  Remove all __pycache__? (y/N): " -n 1 -r
        echo

        if [[ $REPLY =~ ^[Yy]$ ]]; then
            find "$BASE_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
            find "$BASE_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
            find "$BASE_DIR" -type f -name "*.pyo" -delete 2>/dev/null || true
            echo -e "${GREEN}✓ Cleaned Python cache${NC}"
        else
            echo -e "${YELLOW}⊘ Skipped${NC}"
        fi
    else
        echo "  No __pycache__ found"
    fi
    echo ""
}

# Main script
echo "This script will help you clear cached data from GFM-RAG workflow."
echo ""
echo "Cache locations:"
echo "  1. KG Construction cache: $KG_CACHE"
echo "  2. QA Construction cache: $QA_CACHE"
echo "  3. Temporary files: $TMP_DIR"
echo "  4. Workflow outputs: $OUTPUT_DIR"
echo "  5. Python cache (__pycache__)"
echo ""
echo "================================================"
echo ""

# 1. KG Construction cache
remove_dir "$KG_CACHE" "KG Construction Cache"

# 2. QA Construction cache
remove_dir "$QA_CACHE" "QA Construction Cache"

# 3. Temporary directory (if exists at root)
remove_dir "$TMP_DIR" "Root Temporary Directory"

# 4. Workflow outputs
if [ -d "$OUTPUT_DIR" ]; then
    echo -e "${YELLOW}Found:${NC} Workflow Outputs"
    echo "  Location: $OUTPUT_DIR"

    # Show subdirectories
    echo "  Contents:"
    ls -1 "$OUTPUT_DIR" 2>/dev/null | head -5 | while read line; do
        echo "    - $line"
    done

    read -p "  Remove workflow outputs? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$OUTPUT_DIR"
        echo -e "${GREEN}✓ Removed${NC}"
    else
        echo -e "${YELLOW}⊘ Skipped${NC}"
    fi
    echo ""
fi

# 5. Python cache
clean_pycache

echo "================================================"
echo -e "${GREEN}Cache clearing completed!${NC}"
echo ""
echo "Next steps:"
echo "  1. To run workflow from scratch:"
echo "     python -m gfmrag.workflow.stage1_index_dataset"
echo ""
echo "  2. Or force recompute by setting force=True in config:"
echo "     python -m gfmrag.workflow.stage1_index_dataset \\"
echo "       kg_constructor.force=True \\"
echo "       qa_constructor.force=True"
echo ""
echo "================================================"
