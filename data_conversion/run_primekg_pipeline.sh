#!/bin/bash
# PrimeKG to UMLS CUI Triples - One-Command Pipeline

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  PrimeKG to UMLS CUI Triples - Complete Pipeline${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""
echo -e "${BLUE}Pipeline Steps:${NC}"
echo -e "  1. Download kg.csv from Harvard Dataverse (~1.5GB)"
echo -e "  2. Download umls_mondo.csv from GitHub (~500KB)"
echo -e "  3. Convert MONDO IDs → UMLS CUIs (reverse mapping)"
echo -e "  4. Generate UMLS CUI-based triples"
echo -e "  5. Validate output"
echo ""
echo -e "${YELLOW}Estimated time: 10-20 minutes (depending on network)${NC}"
echo ""

# Parse arguments
OUTPUT_DIR="./primekg_output"
DATA_DIR="./primekg_data"
STRATEGY="map"
SKIP_DOWNLOAD=""
KEEP_UNMAPPED=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="--skip-download"
            shift
            ;;
        --keep-unmapped)
            KEEP_UNMAPPED="--keep-unmapped"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output-dir DIR     Output directory (default: ./primekg_output)"
            echo "  --data-dir DIR       Data directory (default: ./primekg_data)"
            echo "  --strategy STRATEGY  filter|map (default: map)"
            echo "  --skip-download      Skip download if files exist"
            echo "  --keep-unmapped      Keep entities without UMLS CUI"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0                              # Full auto mode"
            echo "  $0 --skip-download              # Use existing files"
            echo "  $0 --strategy filter            # UMLS only (fast)"
            echo "  $0 --keep-unmapped              # Keep all entities"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Show configuration
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Output directory: ${OUTPUT_DIR}"
echo -e "  Data directory:   ${DATA_DIR}"
echo -e "  Strategy:         ${STRATEGY}"
[ -n "$SKIP_DOWNLOAD" ] && echo -e "  Skip download:    Yes" || echo -e "  Skip download:    No"
[ -n "$KEEP_UNMAPPED" ] && echo -e "  Keep unmapped:    Yes" || echo -e "  Keep unmapped:    No"
echo ""

# Confirmation
if [ -z "$SKIP_DOWNLOAD" ]; then
    echo -e "${YELLOW}This will download ~1.5GB of data. Continue? (y/N):${NC} "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""
fi

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo -e "${GREEN}✓ Using Python: $PYTHON_CMD${NC}"
echo ""

# Check dependencies
echo -e "${BLUE}Checking dependencies...${NC}"
if ! $PYTHON_CMD -c "import requests, tqdm, pandas" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install requests tqdm pandas || {
        echo -e "${RED}Failed to install dependencies${NC}"
        exit 1
    }
fi
echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Run pipeline
echo -e "${CYAN}================================================================${NC}"
echo -e "${CYAN}  Starting Pipeline...${NC}"
echo -e "${CYAN}================================================================${NC}"
echo ""

$PYTHON_CMD primekg_pipeline.py \
    --output-dir "$OUTPUT_DIR" \
    --data-dir "$DATA_DIR" \
    --strategy "$STRATEGY" \
    $SKIP_DOWNLOAD \
    $KEEP_UNMAPPED

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${GREEN}  ✅ PIPELINE COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo -e "${BLUE}Output file:${NC}"
    echo -e "  ${OUTPUT_DIR}/primekg_umls_triples.txt"
    echo ""
    echo -e "${BLUE}Quick preview:${NC}"
    if [ -f "${OUTPUT_DIR}/primekg_umls_triples.txt" ]; then
        head -10 "${OUTPUT_DIR}/primekg_umls_triples.txt"
        echo ""
        LINE_COUNT=$(wc -l < "${OUTPUT_DIR}/primekg_umls_triples.txt")
        FILE_SIZE=$(du -h "${OUTPUT_DIR}/primekg_umls_triples.txt" | cut -f1)
        echo -e "${GREEN}Total triples: ${LINE_COUNT}${NC}"
        echo -e "${GREEN}File size: ${FILE_SIZE}${NC}"
    fi
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo -e "  1. Copy to GFM data directory:"
    echo -e "     ${YELLOW}cp ${OUTPUT_DIR}/primekg_umls_triples.txt /home/user/GFM/data/kg.txt${NC}"
    echo ""
    echo -e "  2. Run Stage 1 (Index KG):"
    echo -e "     ${YELLOW}cd /home/user/GFM${NC}"
    echo -e "     ${YELLOW}python -m gfmrag.workflow.stage1_index_dataset${NC}"
    echo ""
    echo -e "  3. Run Stage 2 (Entity Resolution):"
    echo -e "     ${YELLOW}python -m gfmrag.workflow.stage2_entity_resolution${NC}"
    echo ""
    echo -e "  4. Run Stage 3 (UMLS Mapping):"
    echo -e "     ${YELLOW}python -m gfmrag.workflow.stage3_umls_mapping${NC}"
    echo ""
    echo -e "${CYAN}================================================================${NC}"
else
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${RED}  ❌ PIPELINE FAILED${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting:${NC}"
    echo -e "  - Check network connection"
    echo -e "  - Verify disk space (need ~2GB)"
    echo -e "  - Check firewall/proxy settings"
    echo -e "  - Review error messages above"
    echo ""
    echo -e "${YELLOW}Manual download:${NC}"
    echo -e "  wget -O ${DATA_DIR}/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620"
    echo -e "  wget -O ${DATA_DIR}/umls_mondo.csv https://raw.githubusercontent.com/mims-harvard/PrimeKG/main/datasets/data/umls/umls_mondo.csv"
    echo ""
    echo -e "  Then run: $0 --skip-download"
    echo ""
    echo -e "${CYAN}================================================================${NC}"
fi

exit $EXIT_CODE
