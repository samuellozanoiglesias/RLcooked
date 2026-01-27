#!/bin/bash

# SINDy Analysis Launcher Script
# This script launches SINDy analysis for all trainings of a given map and checkpoint
# It runs both combined analysis (all simulations together) and individual analysis (each simulation separately)

# USE: 
# nohup ./launch_sindy_analysis.sh <MAP_NR> [OPTIONS] > log_sindy.out 2>&1 &

# Default parameters
MAP_NR="baseline_division_of_labor"
CHECKPOINT_NUMBER="final"
CLUSTER="cuenca"
GAME_VERSION="classic"
INTENT_VERSION="v3.1"
COOPERATIVE=True
SINDY_THRESHOLD=0.01
POLYNOMIAL_DEGREE=2
SMOOTH_DATA=True
SAVE_PLOTS=True

# Analysis modes
RUN_COMBINED=true
RUN_INDIVIDUAL=true
MAX_PARALLEL_JOBS=5  # Limit parallel jobs to avoid overwhelming the system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "SINDy Analysis Launcher - Run SINDy analysis on multiple trainings"
    echo ""
    echo "Options:"
    echo "  --map_nr MAP                 Map name (default: baseline_division_of_labor)"
    echo "  --checkpoint CHECKPOINT      Checkpoint number (default: final)"
    echo "  --cluster CLUSTER           Cluster name (default: cuenca)"
    echo "  --game_version VERSION      Game version (default: classic)"
    echo "  --intent_version VERSION    Intent version (default: v3.1)"
    echo "  --cooperative BOOL          Cooperative mode (default: True)"
    echo "  --threshold FLOAT           SINDy threshold (default: 0.01)"
    echo "  --degree INT                Polynomial degree (default: 2)"
    echo "  --no-combined               Skip combined analysis"
    echo "  --no-individual             Skip individual analysis"
    echo "  --combined-only             Run only combined analysis"
    echo "  --individual-only           Run only individual analysis"
    echo "  --max-jobs INT              Maximum parallel jobs (default: 5)"
    echo "  --training-pattern PATTERN  Pattern to match training IDs (default: all)"
    echo "  --help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                                    # Run with defaults"
    echo "  $0 --map_nr encouraged_division_of_labor             # Different map"
    echo "  $0 --checkpoint 100 --combined-only                 # Only combined analysis"
    echo "  $0 --training-pattern '*2025-09-13*'                # Specific training pattern"
    echo "  $0 --threshold 0.05 --degree 3                      # Custom SINDy parameters"
}

# Parse command line arguments
TRAINING_PATTERN="*"
while [[ $# -gt 0 ]]; do
    case $1 in
        --map_nr)
            MAP_NR="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_NUMBER="$2"
            shift 2
            ;;
        --cluster)
            CLUSTER="$2"
            shift 2
            ;;
        --game_version)
            GAME_VERSION="$2"
            shift 2
            ;;
        --intent_version)
            INTENT_VERSION="$2"
            shift 2
            ;;
        --cooperative)
            COOPERATIVE="$2"
            shift 2
            ;;
        --threshold)
            SINDY_THRESHOLD="$2"
            shift 2
            ;;
        --degree)
            POLYNOMIAL_DEGREE="$2"
            shift 2
            ;;
        --no-combined)
            RUN_COMBINED=false
            shift
            ;;
        --no-individual)
            RUN_INDIVIDUAL=false
            shift
            ;;
        --combined-only)
            RUN_COMBINED=true
            RUN_INDIVIDUAL=false
            shift
            ;;
        --individual-only)
            RUN_COMBINED=false
            RUN_INDIVIDUAL=true
            shift
            ;;
        --max-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        --training-pattern)
            TRAINING_PATTERN="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Validate parameters
if [[ $MAX_PARALLEL_JOBS -lt 1 ]]; then
    echo -e "${RED}Error: max-jobs must be at least 1${NC}"
    exit 1
fi

if [[ "$RUN_COMBINED" == false && "$RUN_INDIVIDUAL" == false ]]; then
    echo -e "${RED}Error: At least one analysis mode must be enabled${NC}"
    exit 1
fi

# Set up directories
COOPERATIVE_DIR=$([ "$COOPERATIVE" == "True" ] && echo "cooperative" || echo "competitive")
BASE_DATA_DIR="/data/samuel_lozano/cooked/${GAME_VERSION}/${INTENT_VERSION}/map_${MAP_NR}/${COOPERATIVE_DIR}"
SCRIPT_DIR="/home/samuel_lozano/cooked"
LOG_DIR="${SCRIPT_DIR}/cuenca/sindy_logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Print configuration
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}SINDy Analysis Launcher${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Map: ${GREEN}${MAP_NR}${NC}"
echo -e "Checkpoint: ${GREEN}${CHECKPOINT_NUMBER}${NC}"
echo -e "Cluster: ${GREEN}${CLUSTER}${NC}"
echo -e "Game Version: ${GREEN}${GAME_VERSION}${NC}"
echo -e "Intent Version: ${GREEN}${INTENT_VERSION}${NC}"
echo -e "Cooperative: ${GREEN}${COOPERATIVE}${NC}"
echo -e "SINDy Threshold: ${GREEN}${SINDY_THRESHOLD}${NC}"
echo -e "Polynomial Degree: ${GREEN}${POLYNOMIAL_DEGREE}${NC}"
echo -e "Training Pattern: ${GREEN}${TRAINING_PATTERN}${NC}"
echo -e "Combined Analysis: ${GREEN}${RUN_COMBINED}${NC}"
echo -e "Individual Analysis: ${GREEN}${RUN_INDIVIDUAL}${NC}"
echo -e "Max Parallel Jobs: ${GREEN}${MAX_PARALLEL_JOBS}${NC}"
echo -e "Base Data Directory: ${GREEN}${BASE_DATA_DIR}${NC}"
echo -e "Log Directory: ${GREEN}${LOG_DIR}${NC}"
echo ""

# Check if base directory exists
if [[ ! -d "$BASE_DATA_DIR" ]]; then
    echo -e "${RED}Error: Base data directory does not exist: ${BASE_DATA_DIR}${NC}"
    exit 1
fi

# Find all training directories
echo -e "${YELLOW}Searching for training directories...${NC}"
TRAINING_DIRS=()
for training_dir in "$BASE_DATA_DIR"/Training_${TRAINING_PATTERN}; do
    if [[ -d "$training_dir" ]]; then
        training_id=$(basename "$training_dir" | sed 's/Training_//')
        simulations_dir="${training_dir}/simulations_${CHECKPOINT_NUMBER}"
        if [[ -d "$simulations_dir" ]]; then
            TRAINING_DIRS+=("$training_id")
            echo -e "  Found: ${GREEN}${training_id}${NC}"
        else
            echo -e "  ${YELLOW}Warning: No simulations_${CHECKPOINT_NUMBER} found for ${training_id}${NC}"
        fi
    fi
done

if [[ ${#TRAINING_DIRS[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No training directories found matching pattern '${TRAINING_PATTERN}'${NC}"
    exit 1
fi

echo -e "${GREEN}Found ${#TRAINING_DIRS[@]} training directories${NC}"
echo ""

# Function to run SINDy analysis
run_sindy_analysis() {
    local training_id="$1"
    local mode="$2"  # "combined" or simulation_id
    local log_suffix="$3"
    
    local log_file="${LOG_DIR}/sindy_${MAP_NR}_${training_id}_${CHECKPOINT_NUMBER}_${log_suffix}.log"
    local cmd="cd ${SCRIPT_DIR} && python sindy_analysis_pipeline.py"
    cmd+=" --map_nr ${MAP_NR}"
    cmd+=" --training_id ${training_id}"
    cmd+=" --checkpoint_number ${CHECKPOINT_NUMBER}"
    cmd+=" --cluster ${CLUSTER}"
    cmd+=" --game_version ${GAME_VERSION}"
    cmd+=" --intent_version ${INTENT_VERSION}"
    cmd+=" --cooperative ${COOPERATIVE}"
    cmd+=" --sindy_threshold ${SINDY_THRESHOLD}"
    cmd+=" --polynomial_degree ${POLYNOMIAL_DEGREE}"
    cmd+=" --smooth_data ${SMOOTH_DATA}"
    cmd+=" --save_plots ${SAVE_PLOTS}"
    
    if [[ "$mode" == "combined" ]]; then
        cmd+=" --combine_simulations"
    else
        cmd+=" --simulation_id ${mode}"
    fi
    
    echo -e "${BLUE}Starting: ${training_id} (${mode})${NC}"
    echo "Command: $cmd" > "$log_file"
    echo "Started at: $(date)" >> "$log_file"
    echo "----------------------------------------" >> "$log_file"
    
    # Run the command and capture both stdout and stderr
    if eval "$cmd" >> "$log_file" 2>&1; then
        echo -e "${GREEN}✓ Completed: ${training_id} (${mode})${NC}"
        echo "Completed at: $(date)" >> "$log_file"
    else
        echo -e "${RED}✗ Failed: ${training_id} (${mode})${NC}"
        echo "Failed at: $(date)" >> "$log_file"
    fi
}

# Function to get simulation IDs for a training
get_simulation_ids() {
    local training_id="$1"
    local simulations_dir="${BASE_DATA_DIR}/Training_${training_id}/simulations_${CHECKPOINT_NUMBER}"
    local sim_ids=()
    
    if [[ -d "$simulations_dir" ]]; then
        for sim_dir in "$simulations_dir"/simulation_*; do
            if [[ -d "$sim_dir" ]]; then
                local sim_id=$(basename "$sim_dir" | sed 's/simulation_//')
                # Check if required files exist
                if [[ -f "${sim_dir}/simulation.csv" ]]; then
                    sim_ids+=("$sim_id")
                fi
            fi
        done
    fi
    
    echo "${sim_ids[@]}"
}

# Job management
declare -a RUNNING_JOBS=()

# Function to wait for jobs to complete
wait_for_jobs() {
    local max_jobs="$1"
    
    while [[ ${#RUNNING_JOBS[@]} -ge $max_jobs ]]; do
        local new_jobs=()
        for job in "${RUNNING_JOBS[@]}"; do
            if kill -0 "$job" 2>/dev/null; then
                new_jobs+=("$job")
            fi
        done
        RUNNING_JOBS=("${new_jobs[@]}")
        sleep 1
    done
}

# Function to add job to tracking
add_job() {
    local job_pid="$1"
    RUNNING_JOBS+=("$job_pid")
}

# Main execution
echo -e "${BLUE}Starting SINDy analysis...${NC}"
echo ""

# Run combined analysis
if [[ "$RUN_COMBINED" == true ]]; then
    echo -e "${YELLOW}=== Running Combined Analysis ===${NC}"
    for training_id in "${TRAINING_DIRS[@]}"; do
        wait_for_jobs "$MAX_PARALLEL_JOBS"
        run_sindy_analysis "$training_id" "combined" "combined" &
        add_job $!
    done
    
    # Wait for all combined analyses to complete
    while [[ ${#RUNNING_JOBS[@]} -gt 0 ]]; do
        wait_for_jobs 0
        sleep 1
    done
    echo -e "${GREEN}Combined analysis completed for all trainings${NC}"
    echo ""
fi

# Run individual analysis
if [[ "$RUN_INDIVIDUAL" == true ]]; then
    echo -e "${YELLOW}=== Running Individual Analysis ===${NC}"
    for training_id in "${TRAINING_DIRS[@]}"; do
        echo -e "${YELLOW}Processing training: ${training_id}${NC}"
        
        # Get simulation IDs for this training
        sim_ids=($(get_simulation_ids "$training_id"))
        
        if [[ ${#sim_ids[@]} -eq 0 ]]; then
            echo -e "${YELLOW}  No simulations found for ${training_id}${NC}"
            continue
        fi
        
        echo -e "  Found ${#sim_ids[@]} simulations"
        
        # Process each simulation
        for sim_id in "${sim_ids[@]}"; do
            wait_for_jobs "$MAX_PARALLEL_JOBS"
            run_sindy_analysis "$training_id" "$sim_id" "sim_${sim_id}" &
            add_job $!
        done
    done
    
    # Wait for all individual analyses to complete
    while [[ ${#RUNNING_JOBS[@]} -gt 0 ]]; do
        wait_for_jobs 0
        sleep 1
    done
    echo -e "${GREEN}Individual analysis completed for all trainings${NC}"
    echo ""
fi

# Final summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Analysis Complete!${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Processed trainings: ${GREEN}${#TRAINING_DIRS[@]}${NC}"
echo -e "Log files location: ${GREEN}${LOG_DIR}${NC}"
echo ""
echo -e "${YELLOW}To check results:${NC}"
echo "  # View logs:"
echo "  ls -la ${LOG_DIR}/sindy_${MAP_NR}_*_${CHECKPOINT_NUMBER}_*.log"
echo ""
echo "  # Check for errors:"
echo "  grep -l \"Failed\\|Error\" ${LOG_DIR}/sindy_${MAP_NR}_*_${CHECKPOINT_NUMBER}_*.log"
echo ""
echo "  # View combined results:"
if [[ "$RUN_COMBINED" == true ]]; then
    for training_id in "${TRAINING_DIRS[@]}"; do
        echo "  ls -la ${BASE_DATA_DIR}/Training_${training_id}/simulations_${CHECKPOINT_NUMBER}/sindy_models_combined/"
    done
fi
echo ""
echo -e "${GREEN}SINDy Analysis Launcher completed successfully!${NC}"
