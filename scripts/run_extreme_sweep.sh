#!/bin/bash

# Extreme hyperparameter sweep for PPO optimization
# For pushing beyond current plateau to theoretical maximum
# Use this if you have additional CUDA devices available

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting EXTREME PPO Hyperparameter Sweep${NC}"
echo -e "${BLUE}Pushing for theoretical maximum of 9.5${NC}"

# Base experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXP_NAME="ppo_extreme_${TIMESTAMP}"

# Create logs directory
mkdir -p logs/${BASE_EXP_NAME}

# Function to run experiment
run_experiment() {
    local cuda_id=$1
    local exp_suffix=$2
    local params=$3
    local description=$4
    
    echo -e "${YELLOW}Starting EXTREME experiment on CUDA:${cuda_id} - ${description}${NC}"
    
    # Create log file
    local log_file="logs/${BASE_EXP_NAME}/cuda_${cuda_id}_${exp_suffix}.log"
    
    # Run the experiment
    python ppo.py \
        --cuda ${cuda_id} \
        --exp_name "${BASE_EXP_NAME}_${exp_suffix}" \
        ${params} \
        > ${log_file} 2>&1 &
    
    local pid=$!
    echo -e "${GREEN}EXTREME Experiment ${exp_suffix} started on CUDA:${cuda_id} with PID:${pid}${NC}"
    echo "${pid}" > "logs/${BASE_EXP_NAME}/cuda_${cuda_id}_${exp_suffix}.pid"
}

# Configuration 1: Very high entropy for maximum exploration
run_experiment 0 "ultra_entropy" \
    "--num_minibatches 1024 --ent_coef 5e-3 --update_epochs 30 --gae_lambda 0.99 --clip_coef 0.05 --vf_coef 1.5 --hidden_dims 2048 2048 2048 1024 --target_kl 0.005" \
    "Ultra high entropy exploration"

# Configuration 2: Very long rollouts with small batches
run_experiment 1 "long_rollout" \
    "--num_minibatches 2048 --ent_coef 1e-3 --update_epochs 15 --gae_lambda 0.995 --clip_coef 0.1 --vf_coef 1.0 --num_steps 50 --hidden_dims 1024 1024 1024 --target_kl 0.01" \
    "Very long rollouts"

# Configuration 3: Massive network with conservative updates
run_experiment 2 "mega_network" \
    "--num_minibatches 512 --ent_coef 1e-4 --update_epochs 50 --gae_lambda 0.99 --clip_coef 0.05 --vf_coef 2.0 --hidden_dims 4096 4096 2048 1024 --target_kl 0.005" \
    "Massive network conservative"

# Configuration 4: No clipping but with target KL
run_experiment 3 "no_clip_kl" \
    "--num_minibatches 1024 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 100.0 --vf_coef 1.0 --hidden_dims 2048 2048 2048 --target_kl 0.001" \
    "No clipping with strict KL"

# Configuration 5: High frequency updates
run_experiment 4 "high_freq" \
    "--num_minibatches 4096 --ent_coef 1e-3 --update_epochs 5 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0 --num_steps 5 --hidden_dims 2048 2048 1024" \
    "High frequency small updates"

# Configuration 6: Different activation function
run_experiment 5 "swish_act" \
    "--num_minibatches 1024 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0 --hidden_dims 2048 2048 2048 1024 --act_fn swish --target_kl 0.01" \
    "Swish activation function"

# Configuration 7: Ultra conservative clipping
run_experiment 6 "ultra_conservative" \
    "--num_minibatches 2048 --ent_coef 1e-5 --update_epochs 100 --gae_lambda 0.999 --clip_coef 0.01 --vf_coef 0.1 --hidden_dims 1024 1024 1024 --target_kl 0.0001" \
    "Ultra conservative updates"

# Configuration 8: Aggressive learning with high capacity
run_experiment 7 "aggressive" \
    "--num_minibatches 256 --ent_coef 1e-2 --update_epochs 10 --gae_lambda 0.9 --clip_coef 0.5 --vf_coef 3.0 --hidden_dims 3072 3072 2048 1024 --max_grad_norm 2.0" \
    "Aggressive high-capacity learning"

echo -e "\n${GREEN}All EXTREME experiments launched!${NC}"
echo -e "${BLUE}These configurations push hyperparameters to extremes${NC}"
echo -e "${BLUE}Monitor with: watch -n 30 './analyze_results.sh'${NC}"
echo -e "${BLUE}Stop all: ./kill_sweep.sh${NC}" 