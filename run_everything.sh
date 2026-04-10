#!/bin/bash

# Default arguments
EPOCHS=15
MODELS=("resnet18" "resnet50" "mobilenet_v3_small" "convnext_tiny" "efficientnet_v2_s" "vit_b_16")
SEEDS=(42 43 44)
DATA_DIR="datasets"
LOG_DIR="logs"
RESULTS_DIR="results"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift ;;
        --data-dir) DATA_DIR="$2"; shift ;;
        --log-dir) LOG_DIR="$2"; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "   Starting Automated Evaluation Suite    "
echo "=========================================="

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

for CLASSES in 2 3; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            RUN_ID="run_s${SEED}"
            echo ">>> Running Classes: $CLASSES | Model: $MODEL | Seed: $SEED | Epochs: $EPOCHS"
            
            # Call the experiment pipeline
            python experiment_pipeline.py \
                --data-dir "$DATA_DIR" \
                --epochs "$EPOCHS" \
                --model-name "$MODEL" \
                --num-classes "$CLASSES" \
                --seed "$SEED" \
                --log-dir "$LOG_DIR" \
                --results-dir "$RESULTS_DIR" \
                --run-id "$RUN_ID"
            
            if [ $? -ne 0 ]; then
                echo -e "\033[0;33mWARNING: Run for Model $MODEL with Seed $SEED exited with a non-zero status.\033[0m"
            fi
        done
    done
done

echo "=========================================="
echo "   Aggregating Results and Plotting       "
echo "=========================================="

python aggregate_results.py

echo "Done! Check the '$RESULTS_DIR' folder for tables and plots."
