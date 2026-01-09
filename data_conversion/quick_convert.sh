#!/bin/bash
# Quick conversion script for CSV to triples

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  CSV to Triples Converter - Quick Run${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No input file specified${NC}"
    echo ""
    echo "Usage:"
    echo "  ./quick_convert.sh <input.csv> [output.txt]"
    echo ""
    echo "Examples:"
    echo "  ./quick_convert.sh input/my_data.csv"
    echo "  ./quick_convert.sh input/my_data.csv output/my_triples.txt"
    echo "  ./quick_convert.sh input/my_data.csv output/my_triples.txt --add-metadata"
    echo ""
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-output/triples.txt}"
EXTRA_ARGS="${@:3}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}"
    exit 1
fi

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"
if ! python3 -c "import pandas, tqdm" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install pandas tqdm
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"
echo ""

# Show input info
echo -e "${BLUE}Input file:${NC} $INPUT_FILE"
echo -e "${BLUE}Output file:${NC} $OUTPUT_FILE"
if [ -n "$EXTRA_ARGS" ]; then
    echo -e "${BLUE}Extra args:${NC} $EXTRA_ARGS"
fi
echo ""

# Preview input
echo -e "${YELLOW}Preview of input CSV (first 3 rows):${NC}"
head -4 "$INPUT_FILE"
echo ""

# Run conversion
echo -e "${GREEN}Running conversion...${NC}"
python csv_to_triples.py "$INPUT_FILE" "$OUTPUT_FILE" $EXTRA_ARGS

# Show output preview
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo -e "${YELLOW}Preview of output (first 10 lines):${NC}"
    head -10 "$OUTPUT_FILE"
    echo ""

    LINE_COUNT=$(wc -l < "$OUTPUT_FILE")
    echo -e "${GREEN}✓ Conversion complete!${NC}"
    echo -e "${GREEN}  Output: $OUTPUT_FILE ($LINE_COUNT triples)${NC}"
else
    echo -e "${RED}Error: Output file not created${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Next steps:${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. Verify output:"
echo "   head -20 $OUTPUT_FILE"
echo ""
echo "2. Copy to GFM data directory:"
echo "   cp $OUTPUT_FILE /home/user/GFM/data/kg.txt"
echo ""
echo "3. Run Stage 1 (Index KG):"
echo "   cd /home/user/GFM"
echo "   python -m gfmrag.workflow.stage1_index_dataset"
echo ""
