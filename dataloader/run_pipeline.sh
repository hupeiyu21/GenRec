#!/bin/bash

# MM-RQVAE Data Processing Pipeline
# This script combines all data processing steps into one unified pipeline

set -e  # Exit on any error

# Default parameters
DATASET="Baby"
INPUT_PATH="../datasets"
OUTPUT_PATH="../datasets"
GPU_ID="0"
MODEL_PATH="/root/Qwen/models--Qwen--Qwen3-Embedding-8B/snapshots/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/"
PLM_NAME="qwen"
IMAGE_ROOT="../datasets/amazon14/Images"
MODEL_CACHE_DIR="../cache_models/clip"

# Function to print usage
print_usage() {
    echo "Usage: $0 [OPTIONS] [STEPS]"
    echo ""
    echo "OPTIONS:"
    echo "  --dataset DATASET          Dataset name (default: Baby)"
    echo "  --input-path PATH          Input data path (default: ../datasets)"
    echo "  --output-path PATH         Output data path (default: ../datasets)"
    echo "  --gpu-id ID                GPU device ID (default: 1)"
    echo "  --model-path PATH          Text embedding model path"
    echo "  --plm-name NAME            PLM name for file naming (default: qwen)"
    echo "  --image-root PATH          Image directory path"
    echo "  --model-cache-dir PATH     Model cache directory"
    echo "  --skip-download            Skip download step"
    echo "  --skip-images              Skip image processing step"
    echo "  --skip-process             Skip data processing step"
    echo "  --skip-text-emb            Skip text embedding generation"
    echo "  --skip-image-emb           Skip image embedding generation"
    echo "  --help                     Show this help message"
    echo ""
    echo "STEPS (if not specified, runs all steps):"
    echo "  download                   Download raw dataset"
    echo "  images                     Download and process images"
    echo "  process                    Process interaction data"
    echo "  text-emb                   Generate text embeddings"
    echo "  image-emb                  Generate image embeddings"
    echo "  fusion-emb                 Fusion embeddings"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 --dataset Beauty                    # Run all steps for Beauty dataset"
    echo "  $0 --dataset Baby download process     # Run only download and process steps"
    echo "  $0 --skip-images --skip-image-emb     # Skip image-related steps"
}

# Function to print step header
print_step() {
    echo ""
    echo "=========================================="
    echo "STEP: $1"
    echo "=========================================="
    echo ""
}

# Function to check if step should be run
should_run_step() {
    local step=$1
    local skip_flag=""
    
    case $step in
        "download") skip_flag="SKIP_DOWNLOAD" ;;
        "images") skip_flag="SKIP_IMAGES" ;;
        "process") skip_flag="SKIP_PROCESS" ;;
        "text-emb") skip_flag="SKIP_TEXT_EMB" ;;
        "image-emb") skip_flag="SKIP_IMAGE_EMB" ;;
        "fusion-emb") skip_flag="SKIP_FUSION_EMB" ;;
    esac
    
    if [[ -n "${!skip_flag}" ]]; then
        return 1  # Skip this step
    fi
    
    if [[ ${#STEPS[@]} -eq 0 ]]; then
        return 0  # Run all steps if no specific steps specified
    fi
    
    for s in "${STEPS[@]}"; do
        if [[ "$s" == "$step" ]]; then
            return 0  # Run this step
        fi
    done
    
    return 1  # Skip this step
}

# Parse command line arguments
STEPS=()
SKIP_DOWNLOAD=""
SKIP_IMAGES=""
SKIP_PROCESS=""
SKIP_TEXT_EMB=""
SKIP_IMAGE_EMB=""
SKIP_FUSION_EMB=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --input-path)
            INPUT_PATH="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --plm-name)
            PLM_NAME="$2"
            shift 2
            ;;
        --image-root)
            IMAGE_ROOT="$2"
            shift 2
            ;;
        --model-cache-dir)
            MODEL_CACHE_DIR="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD="true"
            shift
            ;;
        --skip-images)
            SKIP_IMAGES="true"
            shift
            ;;
        --skip-process)
            SKIP_PROCESS="true"
            shift
            ;;
        --skip-text-emb)
            SKIP_TEXT_EMB="true"
            shift
            ;;
        --skip-image-emb)
            SKIP_IMAGE_EMB="true"
            shift
            ;;
        --help)
            print_usage
            exit 0
            ;;
        download|images|process|text-emb|image-emb|fusion-emb)
            STEPS+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Print configuration
echo "=========================================="
echo "MM-RQVAE Data Processing Pipeline"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Input Path: $INPUT_PATH"
echo "Output Path: $OUTPUT_PATH"
echo "GPU ID: $GPU_ID"
echo "Model Path: $MODEL_PATH"
echo "PLM Name: $PLM_NAME"
echo "Image Root: $IMAGE_ROOT"
echo "Model Cache Dir: $MODEL_CACHE_DIR"
echo ""

if [[ ${#STEPS[@]} -gt 0 ]]; then
    echo "Running specific steps: ${STEPS[*]}"
else
    echo "Running all steps"
fi

echo ""

# Step 1: Download Dataset
if should_run_step "download"; then
    print_step "Download Dataset"
    echo "Downloading $DATASET dataset..."
    python amazon14_data_download.py --category "$DATASET"
    echo "Download completed!"
fi

# Step 2: Load Images
if should_run_step "images"; then
    print_step "Load Images"
    echo "Processing images for $DATASET dataset..."
    python load_all_figures.py --dataset "$DATASET"
    echo "Image processing completed!"
fi

# Step 3: Process Data
if should_run_step "process"; then
    print_step "Process Data"
    echo "Processing interaction data for $DATASET dataset..."
    python amazon18_data_process.py \
        --dataset "$DATASET" \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_PATH"
    echo "Data processing completed!"
fi

# Step 4: Generate Text Embeddings
if should_run_step "text-emb"; then
    print_step "Generate Text Embeddings"
    echo "Generating text embeddings for $DATASET dataset..."
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python amazon_text_emb.py --dataset "$DATASET" \
        --model_name_or_path "$MODEL_PATH" \
        --plm_name "$PLM_NAME"
    echo "Text embedding generation completed!"
fi

# Step 5: Generate Image Embeddings
if should_run_step "image-emb"; then
    print_step "Generate Image Embeddings"
    echo "Generating image embeddings for $DATASET dataset..."
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python clip_feature.py \
        --image_root "$IMAGE_ROOT" \
        --save_root "$OUTPUT_PATH" \
        --model_cache_dir "$MODEL_CACHE_DIR" \
        --dataset "$DATASET"
    echo "Image embedding generation completed!"
fi

# Step 6: Fusion Embeddings
if should_run_step "fusion-emb"; then
    print_step "Fusion Embeddings"
    echo "Fusing embeddings for $DATASET dataset..."
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python fusion_embeddings.py \
        --method cross-attention \
        --dataset_name $DATASET \
        --epochs 20 \
        --embed_dim 768 \
        --nhead 4
fi

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Output directory: $OUTPUT_PATH"
echo ""

# Check if output files exist
echo "Checking output files..."
if [[ -d "$OUTPUT_PATH/$DATASET" ]]; then
    echo "✓ Processed data directory exists"
    ls -la "$OUTPUT_PATH/$DATASET/"
else
    echo "✗ Processed data directory not found"
fi

echo ""
echo "Pipeline finished!"


