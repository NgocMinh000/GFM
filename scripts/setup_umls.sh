#!/bin/bash
################################################################################
# UMLS Download and Setup Script for Remote Server
#
# Usage:
#   bash scripts/setup_umls.sh
#
# This script will:
#   1. Prompt for UTS credentials
#   2. Download UMLS Full Release
#   3. Extract required RRF files
#   4. Clean up temporary files
#   5. Verify installation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
UMLS_DIR="$PROJECT_ROOT/data/umls"
UMLS_VERSION="${UMLS_VERSION:-2024AA}"  # Default to 2024AA, can override
DOWNLOAD_URL="https://download.nlm.nih.gov/umls/kss/$UMLS_VERSION/umls-$UMLS_VERSION-full.zip"

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_dependencies() {
    print_header "Checking Dependencies"

    local missing_deps=0

    # Check for required commands
    for cmd in curl unzip; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is not installed"
            missing_deps=1
        else
            print_success "$cmd is available"
        fi
    done

    if [ $missing_deps -eq 1 ]; then
        print_error "Missing dependencies. Please install them first:"
        echo "  Ubuntu/Debian: sudo apt-get install curl unzip"
        echo "  CentOS/RHEL:   sudo yum install curl unzip"
        exit 1
    fi

    echo ""
}

check_disk_space() {
    print_header "Checking Disk Space"

    # Check available space (in GB)
    local available=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
    local required=45  # 8GB download + 40GB extracted (with margin)

    print_info "Available space: ${available}GB"
    print_info "Required space:  ${required}GB"

    if [ "$available" -lt "$required" ]; then
        print_warning "Low disk space! You need at least ${required}GB"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Sufficient disk space available"
    fi

    echo ""
}

get_credentials() {
    print_header "UTS Credentials"

    print_info "You need a UTS (UMLS Terminology Services) account"
    print_info "Register at: https://uts.nlm.nih.gov/uts/signup-login"
    echo ""

    read -p "Enter UTS Username: " UTS_USER
    read -sp "Enter UTS Password: " UTS_PASS
    echo ""

    if [ -z "$UTS_USER" ] || [ -z "$UTS_PASS" ]; then
        print_error "Username and password are required"
        exit 1
    fi

    echo ""
}

download_umls() {
    print_header "Downloading UMLS $UMLS_VERSION"

    mkdir -p "$UMLS_DIR"
    cd "$UMLS_DIR"

    local zip_file="umls-$UMLS_VERSION-full.zip"

    # Check if already downloaded
    if [ -f "$zip_file" ]; then
        print_warning "Found existing $zip_file"
        read -p "Re-download? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$zip_file"
        else
            print_info "Using existing download"
            return 0
        fi
    fi

    print_info "Downloading from: $DOWNLOAD_URL"
    print_info "This may take 30-60 minutes depending on your connection..."
    echo ""

    # Download with curl (supports resume with -C -)
    if curl -u "$UTS_USER:$UTS_PASS" \
            -C - \
            --progress-bar \
            -o "$zip_file" \
            "$DOWNLOAD_URL"; then
        print_success "Download completed"
    else
        print_error "Download failed!"
        print_info "Try running the script again - it will resume from where it stopped"
        exit 1
    fi

    echo ""
}

extract_files() {
    print_header "Extracting UMLS Files"

    cd "$UMLS_DIR"
    local zip_file="umls-$UMLS_VERSION-full.zip"

    if [ ! -f "$zip_file" ]; then
        print_error "Download file not found: $zip_file"
        exit 1
    fi

    print_info "Extracting $zip_file..."
    print_info "This may take 10-20 minutes..."

    # Extract main zip
    unzip -q -o "$zip_file"
    print_success "Main archive extracted"

    # Find and extract mmsys.zip
    local mmsys_path=$(find . -name "mmsys.zip" | head -1)

    if [ -z "$mmsys_path" ]; then
        print_error "mmsys.zip not found in archive"
        exit 1
    fi

    local extract_dir=$(dirname "$mmsys_path")
    cd "$extract_dir"

    print_info "Extracting mmsys.zip..."
    unzip -q -o mmsys.zip
    print_success "UMLS data extracted"

    echo ""
}

copy_rrf_files() {
    print_header "Copying Required RRF Files"

    cd "$UMLS_DIR"

    # Find META directory
    local meta_dir=$(find . -type d -name "META" | head -1)

    if [ -z "$meta_dir" ]; then
        print_error "META directory not found"
        exit 1
    fi

    print_info "Found META directory: $meta_dir"

    # Copy required files
    local files=("MRCONSO.RRF" "MRSTY.RRF" "MRDEF.RRF")

    for file in "${files[@]}"; do
        local src="$meta_dir/$file"
        local dst="$UMLS_DIR/$file"

        if [ -f "$src" ]; then
            print_info "Copying $file..."
            cp "$src" "$dst"
            print_success "$file copied ($(du -h "$dst" | cut -f1))"
        else
            print_error "$file not found in $meta_dir"
            exit 1
        fi
    done

    echo ""
}

cleanup() {
    print_header "Cleaning Up"

    cd "$UMLS_DIR"

    print_warning "This will delete temporary files to save disk space"
    print_info "Files to delete:"
    echo "  - umls-$UMLS_VERSION-full.zip (~8GB)"
    echo "  - Extracted directories (~35GB)"
    echo ""

    read -p "Proceed with cleanup? (Y/n): " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        # Remove zip file
        if [ -f "umls-$UMLS_VERSION-full.zip" ]; then
            print_info "Removing zip file..."
            rm -f "umls-$UMLS_VERSION-full.zip"
            print_success "Zip file removed"
        fi

        # Remove extracted directories (keep only RRF files)
        print_info "Removing extracted directories..."
        find "$UMLS_DIR" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
        print_success "Temporary directories removed"

        # Show space saved
        print_success "Cleanup completed!"
    else
        print_info "Skipping cleanup - temporary files kept"
    fi

    echo ""
}

verify_installation() {
    print_header "Verifying Installation"

    cd "$UMLS_DIR"

    local files=("MRCONSO.RRF" "MRSTY.RRF" "MRDEF.RRF")
    local all_ok=1

    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            local size=$(du -h "$file" | cut -f1)
            local lines=$(wc -l < "$file")
            print_success "$file: $size, $lines lines"
        else
            print_error "$file: NOT FOUND"
            all_ok=0
        fi
    done

    echo ""

    if [ $all_ok -eq 1 ]; then
        print_success "All required files are present!"
        echo ""
        print_info "Files location: $UMLS_DIR"
        echo ""
        print_header "Next Steps"
        echo ""
        echo "1. Prepare your Knowledge Graph file:"
        echo "   cp /path/to/your/kg.txt $PROJECT_ROOT/data/kg_clean.txt"
        echo ""
        echo "2. Run the UMLS mapping pipeline:"
        echo "   cd $PROJECT_ROOT"
        echo "   python run_umls_pipeline.py"
        echo ""
        echo "3. Or run a test with Stage 0 only:"
        echo "   python run_umls_pipeline.py --stages stage0_umls_loading"
        echo ""
    else
        print_error "Installation incomplete - some files are missing"
        exit 1
    fi
}

# Main execution
main() {
    clear
    print_header "UMLS Download and Setup for Remote Server"
    echo ""
    print_info "Project root: $PROJECT_ROOT"
    print_info "UMLS directory: $UMLS_DIR"
    print_info "UMLS version: $UMLS_VERSION"
    echo ""

    # Run setup steps
    check_dependencies
    check_disk_space
    get_credentials
    download_umls
    extract_files
    copy_rrf_files
    cleanup
    verify_installation

    print_success "Setup completed successfully! 🎉"
}

# Run main function
main
