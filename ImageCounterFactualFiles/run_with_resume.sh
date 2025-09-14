#!/bin/bash

# Wrapper script for Vizfiletest.py with automatic resume functionality
# This script automatically detects if a progress file exists and resumes processing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to get the correct data directory for a model type
get_correct_data_dir() {
    local model_type="$1"
    case "$model_type" in
        "artstyle")
            echo "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/test"
            ;;
        "century")
            echo "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/test_century"
            ;;
        "portrait")
            echo "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Wrapper script for Vizfiletest.py with automatic resume functionality"
    echo ""
    echo "Required arguments:"
    echo "  --model_type TYPE     Model type: artstyle, century, or portrait"
    echo "  --data_dir DIR        Directory containing images for batch analysis"
    echo ""
    echo "IMPORTANT: Data directory must match the configured path for each model type:"
    echo "  - artstyle:   /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/dataset_resized_train_val_test_combi/test"
    echo "  - century:    /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/CenturyWiseLabelsInsteadofArtists/test_century"
    echo "  - portrait:   /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test"
    echo ""
    echo "Optional arguments:"
    echo "  --image_path PATH     Path to single image for analysis"
    echo "  --output DIR          Output directory (default: ImageCounterfactualExplanations)"
    echo "  --attention_threshold FLOAT  Threshold for high attention (default: 0.2)"
    echo "  --dpi INT             DPI for output images (default: 150)"
    echo "  --device DEVICE       Device: 'auto', 'cpu', or 'cuda' (default: 'cpu')"
    echo "  --verbose             Enable verbose output"
    echo "  --save_raw_only       Save only raw attention maps (no overlays)"
    echo "  --csv_format FORMAT   CSV format: 'detailed' or 'summary' (default: 'detailed')"
    echo "  --force               Force reprocessing even if files exist"
    echo "  --progress_file FILE  Custom progress file path"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model_type portrait --data_dir /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test"
    echo "  $0 --model_type portrait --data_dir /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test --verbose --save_raw_only"
    echo "  $0 --model_type portrait --data_dir /data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/PortraitDataset/test --csv_format detailed --force"
    echo ""
    echo "The script will automatically:"
    echo "  - Detect existing progress files"
    echo "  - Resume from where it left off"
    echo "  - Skip already processed files"
    echo "  - Handle interruptions gracefully"
}

# Parse command line arguments
MODEL_TYPE=""
DATA_DIR=""
IMAGE_PATH=""
OUTPUT_DIR=""
ATTENTION_THRESHOLD=""
DPI=""
DEVICE=""
VERBOSE=""
SAVE_RAW_ONLY=""
CSV_FORMAT=""
FORCE=""
PROGRESS_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --image_path)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --attention_threshold)
            ATTENTION_THRESHOLD="$2"
            shift 2
            ;;
        --dpi)
            DPI="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --save_raw_only)
            SAVE_RAW_ONLY="--save_raw_only"
            shift
            ;;
        --csv_format)
            CSV_FORMAT="$2"
            shift 2
            ;;
        --force)
            FORCE="--force_reprocess"
            shift 2
            ;;
        --progress_file)
            PROGRESS_FILE="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "$MODEL_TYPE" ]]; then
    print_error "Missing required argument: --model_type"
    show_usage
    exit 1
fi

# Check that either data_dir or image_path is provided
if [[ -z "$DATA_DIR" && -z "$IMAGE_PATH" ]]; then
    print_error "Missing required argument: Either --data_dir or --image_path must be specified"
    show_usage
    exit 1
fi

# Check if both are provided (not allowed)
if [[ -n "$DATA_DIR" && -n "$IMAGE_PATH" ]]; then
    print_error "Error: Cannot specify both --data_dir and --image_path. Use one or the other."
    show_usage
    exit 1
fi

# Validate data directory matches the configured path for the model type
if [[ -n "$DATA_DIR" ]]; then
    CORRECT_DATA_DIR=$(get_correct_data_dir "$MODEL_TYPE")
    if [[ "$DATA_DIR" != "$CORRECT_DATA_DIR" ]]; then
        print_error "Data directory mismatch!"
        print_error "For model type '$MODEL_TYPE', the data directory must be:"
        print_error "  $CORRECT_DATA_DIR"
        print_error "You provided: $DATA_DIR"
        echo ""
        print_error "Please use the correct data directory for your model type."
        exit 1
    fi
fi

# Check if Vizfiletest.py exists
if [[ ! -f "Vizfiletest.py" ]]; then
    print_error "Vizfiletest.py not found in current directory"
    exit 1
fi

# Check if data directory exists
if [[ ! -d "$DATA_DIR" ]]; then
    print_error "Data directory does not exist: $DATA_DIR"
    exit 1
fi

# Build the command
CMD="python Vizfiletest.py --model_type $MODEL_TYPE"

if [[ -n "$DATA_DIR" ]]; then
    CMD="$CMD --data_dir $DATA_DIR"
fi

if [[ -n "$IMAGE_PATH" ]]; then
    CMD="$CMD --image_path $IMAGE_PATH"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

if [[ -n "$ATTENTION_THRESHOLD" ]]; then
    CMD="$CMD --attention_threshold $ATTENTION_THRESHOLD"
fi

if [[ -n "$DPI" ]]; then
    CMD="$CMD --dpi $DPI"
fi

if [[ -n "$DEVICE" ]]; then
    CMD="$CMD --device $DEVICE"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

if [[ -n "$SAVE_RAW_ONLY" ]]; then
    CMD="$CMD $SAVE_RAW_ONLY"
fi

if [[ -n "$CSV_FORMAT" ]]; then
    CMD="$CMD --csv_format $CSV_FORMAT"
fi

if [[ -n "$FORCE" ]]; then
    CMD="$CMD $FORCE"
fi

if [[ -n "$PROGRESS_FILE" ]]; then
    CMD="$CMD --progress_file $PROGRESS_FILE"
fi

# Auto-detect progress file if not specified
if [[ -z "$PROGRESS_FILE" ]]; then
    if [[ -n "$DATA_DIR" ]]; then
        DATA_DIR_NAME=$(basename "$DATA_DIR")
        DATE=$(date +%Y%m%d)
        AUTO_PROGRESS_FILE="progress_${MODEL_TYPE}_${DATA_DIR_NAME}_${DATE}.pkl"
        
        if [[ -f "$AUTO_PROGRESS_FILE" ]]; then
            print_status "Found existing progress file: $AUTO_PROGRESS_FILE"
            print_status "Automatically resuming from previous run..."
            CMD="$CMD --resume"
        else
            print_status "No existing progress file found. Starting fresh..."
        fi
    else
        # For single image processing, no progress file needed
        print_status "Single image processing - no progress file needed"
    fi
else
    if [[ -f "$PROGRESS_FILE" ]]; then
        print_status "Found specified progress file: $PROGRESS_FILE"
        print_status "Automatically resuming from previous run..."
        CMD="$CMD --resume"
    else
        print_status "Specified progress file not found. Starting fresh..."
    fi
fi

# Show what will be executed
print_status "Executing command:"
echo "$CMD"
echo ""

# Execute the command
print_status "Starting image analysis..."
if eval $CMD; then
    print_success "Image analysis completed successfully!"
    
    # Show progress file information
    if [[ -z "$PROGRESS_FILE" ]]; then
        if [[ -f "$AUTO_PROGRESS_FILE" ]]; then
            print_status "Progress saved to: $AUTO_PROGRESS_FILE"
            print_status "You can resume later using: --resume"
        fi
    else
        if [[ -f "$PROGRESS_FILE" ]]; then
            print_status "Progress saved to: $PROGRESS_FILE"
            print_status "You can resume later using: --resume"
        fi
    fi
else
    print_error "Image analysis failed with exit code $?"
    print_warning "You can resume from where you left off using: --resume"
    exit 1
fi
