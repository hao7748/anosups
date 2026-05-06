#!/bin/bash

# Script to run anosups experiments for different datasets and record execution time
# Usage: ./run_anosups_experiments.sh

# Set the path to the Python script
PYTHON_SCRIPT="main_anosups.py"

# Create logs directory if it doesn't exist
LOG_DIR="./experiment_logs"
mkdir -p ${LOG_DIR}

# Get current timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_DIR}/all_experiments_${TIMESTAMP}.log"

# Define datasets to run
DATASETS=("01" "02" "hazelnut")

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MASTER_LOG}
}

# Function to run experiment for a specific dataset
run_experiment() {
    local dataset=$1
    local start_time=$(date +%s)

    log_message "=========================================="
    log_message "Starting experiment for dataset: ${dataset}"
    log_message "Start time: $(date +'%Y-%m-%d %H:%M:%S %Z')"
    log_message "=========================================="

    # Run the Python script and capture output
    python ${PYTHON_SCRIPT} --data ${dataset} 2>&1 | tee -a ${MASTER_LOG}

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))

    log_message "=========================================="
    if [ ${exit_code} -eq 0 ]; then
        log_message "[OK] Experiment for dataset: ${dataset} COMPLETED successfully"
    else
        log_message "[FAIL] Experiment for dataset: ${dataset} FAILED with exit code ${exit_code}"
    fi
    log_message "End time: $(date +'%Y-%m-%d %H:%M:%S %Z')"
    log_message "Total duration: ${hours}h ${minutes}m ${seconds}s (${duration} seconds)"
    log_message "=========================================="
    echo "" >> ${MASTER_LOG}

    # Return duration for summary
    echo ${duration}
}

# Function to run all datasets
run_all_experiments() {
    log_message "Starting ANOSUPS experiments for all datasets"
    log_message "Master log file: ${MASTER_LOG}"
    log_message "Python script: ${PYTHON_SCRIPT}"
    echo "" >> ${MASTER_LOG}

    # Check if Python script exists
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        log_message "[ERROR] Python script ${PYTHON_SCRIPT} not found!"
        exit 1
    fi

    # Check conda environment
    if command -v conda &> /dev/null; then
        log_message "Conda found. Activating anosups environment..."
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate anosups
        if [ $? -ne 0 ]; then
            log_message "[WARN] Could not activate 'anosups' environment. Using base environment."
        else
            log_message "[OK] Activated conda environment: anosups"
        fi
    else
        log_message "[WARN] Conda not found. Using default Python environment."
    fi

    echo "" >> ${MASTER_LOG}

    # Array to store durations
    declare -A durations

    # Run experiments for each dataset
    for dataset in "${DATASETS[@]}"; do
        duration=$(run_experiment ${dataset})
        durations[${dataset}]=${duration}
        echo "" >> ${MASTER_LOG}
    done

    # Print summary
    log_message ""
    log_message "========== EXPERIMENT SUMMARY =========="
    log_message "Total datasets processed: ${#DATASETS[@]}"
    log_message ""
    log_message "Dataset-wise execution times:"
    for dataset in "${DATASETS[@]}"; do
        local dur=${durations[${dataset}]}
        local hours=$((dur / 3600))
        local minutes=$(((dur % 3600) / 60))
        local seconds=$((dur % 60))
        log_message "  - ${dataset}: ${hours}h ${minutes}m ${seconds}s (${dur} seconds)"
    done
    log_message ""
    log_message "All results saved in: ${MASTER_LOG}"
    log_message "=========================================="
}

# Function to run single dataset
run_single_dataset() {
    local dataset=$1
    local valid_datasets="01|02|hazelnut"

    if [[ ! ${dataset} =~ ^(${valid_datasets})$ ]]; then
        echo "[ERROR] Invalid dataset '${dataset}'. Valid options: 01, 02, hazelnut"
        exit 1
    fi

    log_message "Running experiment for single dataset: ${dataset}"
    run_experiment ${dataset}
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments - run all datasets
    run_all_experiments
elif [ $# -eq 1 ]; then
    # One argument - run specific dataset
    run_single_dataset $1
else
    echo "Usage: $0 [dataset]"
    echo "  dataset: 01, 02, hazelnut (optional, run all if not specified)"
    echo ""
    echo "Examples:"
    echo "  $0              # Run experiments for all datasets"
    echo "  $0 01           # Run experiment only for dataset '01'"
    echo "  $0 hazelnut     # Run experiment only for dataset 'hazelnut'"
    exit 1
fi

exit 0
