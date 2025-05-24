#!/bin/bash

# Hyperparameter sweep script for PPO optimization
# Each configuration runs on a different CUDA device

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PPO Hyperparameter Sweep${NC}"
echo -e "${BLUE}Targeting theoretical maximum of 9.5 from current plateau of 5${NC}"

# Base experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_EXP_NAME="ppo_sweep_${TIMESTAMP}"

# Create logs directory
mkdir -p logs/${BASE_EXP_NAME}

# Function to run experiment
run_experiment() {
    local cuda_id=$1
    local exp_suffix=$2
    local params=$3
    local description=$4
    
    echo -e "${YELLOW}Starting experiment on CUDA:${cuda_id} - ${description}${NC}"
    
    # Create log file
    local log_file="logs/${BASE_EXP_NAME}/cuda_${cuda_id}_${exp_suffix}.log"
    
    # Run the experiment
    python ppo.py \
        --cuda ${cuda_id} \
        --exp_name "${BASE_EXP_NAME}_${exp_suffix}" \
        ${params} \
        > ${log_file} 2>&1 &
    
    local pid=$!
    echo -e "${GREEN}Experiment ${exp_suffix} started on CUDA:${cuda_id} with PID:${pid}${NC}"
    echo "${pid}" > "logs/${BASE_EXP_NAME}/cuda_${cuda_id}_${exp_suffix}.pid"
}

# Configuration 1: Baseline with increased minibatches
run_experiment 0 "baseline_plus" \
    "--num_minibatches 512" \
    "Baseline + 512 minibatches"

# Configuration 2: Better exploration (higher entropy)
run_experiment 1 "high_entropy" \
    "--num_minibatches 512 --ent_coef 1e-3" \
    "High entropy exploration"

# Configuration 3: More update epochs
run_experiment 2 "more_epochs" \
    "--num_minibatches 512 --ent_coef 1e-3 --update_epochs 20" \
    "More update epochs"

# Configuration 4: Better GAE and clipping
run_experiment 3 "better_gae" \
    "--num_minibatches 512 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1" \
    "Improved GAE and clipping"

# Configuration 5: Higher value function coefficient
run_experiment 4 "high_vf_coef" \
    "--num_minibatches 512 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0" \
    "Higher value function coefficient"

# Configuration 6: Gradient clipping and longer rollouts
run_experiment 5 "grad_clip_rollout" \
    "--num_minibatches 512 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0 --max_grad_norm 1.0 --num_steps 20" \
    "Better gradient clipping and longer rollouts"

# Configuration 7: Larger network capacity
run_experiment 6 "large_network" \
    "--num_minibatches 512 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0 --hidden_dims 2048 2048 2048 1024" \
    "Larger network capacity"

# Configuration 8: Full optimization with target KL and no LR annealing
run_experiment 7 "target_kl" \
    "--num_minibatches 1024 --ent_coef 1e-3 --update_epochs 20 --gae_lambda 0.99 --clip_coef 0.1 --vf_coef 1.0 --hidden_dims 2048 2048 2048 1024 --target_kl 0.01" \
    "Full optimization with target KL"

# Wait a moment for all to start
sleep 5

# Monitor function
monitor_experiments() {
    echo -e "\n${BLUE}Monitoring experiments...${NC}"
    echo -e "${BLUE}Use 'tail -f logs/${BASE_EXP_NAME}/cuda_X_*.log' to follow specific experiments${NC}"
    echo -e "${BLUE}Use './kill_sweep.sh' to stop all experiments${NC}"
    
    while true; do
        echo -e "\n${YELLOW}=== Experiment Status $(date) ===${NC}"
        
        for i in {0..7}; do
            local pid_file="logs/${BASE_EXP_NAME}/cuda_${i}_*.pid"
            if ls ${pid_file} 1> /dev/null 2>&1; then
                local pid=$(cat ${pid_file})
                if ps -p ${pid} > /dev/null 2>&1; then
                    echo -e "${GREEN}CUDA:${i} - RUNNING (PID:${pid})${NC}"
                else
                    echo -e "${RED}CUDA:${i} - FINISHED/CRASHED${NC}"
                fi
            else
                echo -e "${RED}CUDA:${i} - NO PID FILE${NC}"
            fi
        done
        
        # Check if any experiments are still running
        local running_count=0
        for i in {0..7}; do
            local pid_file="logs/${BASE_EXP_NAME}/cuda_${i}_*.pid"
            if ls ${pid_file} 1> /dev/null 2>&1; then
                local pid=$(cat ${pid_file})
                if ps -p ${pid} > /dev/null 2>&1; then
                    ((running_count++))
                fi
            fi
        done
        
        if [ ${running_count} -eq 0 ]; then
            echo -e "\n${GREEN}All experiments completed!${NC}"
            break
        fi
        
        sleep 60  # Check every minute
    done
}

# Create kill script
cat > kill_sweep.sh << 'EOF'
#!/bin/bash
echo "Killing all sweep experiments..."
for pid_file in logs/*/cuda_*.pid; do
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            echo "Killing PID: $pid"
            kill $pid
        fi
        rm "$pid_file"
    fi
done
echo "All experiments killed."
EOF
chmod +x kill_sweep.sh

# Create results analysis script
cat > analyze_results.sh << 'EOF'
#!/bin/bash
echo "=== Hyperparameter Sweep Results Analysis ==="
echo

for log_file in logs/*/cuda_*.log; do
    if [ -f "$log_file" ]; then
        echo "=== $(basename $log_file) ==="
        
        # Extract final return if available
        final_return=$(grep -o "returns: [0-9.-]*" "$log_file" | tail -1 | cut -d' ' -f2)
        max_return=$(grep -o "returns: [0-9.-]*" "$log_file" | cut -d' ' -f2 | sort -n | tail -1)
        
        if [ ! -z "$final_return" ]; then
            echo "Final return: $final_return"
            echo "Max return: $max_return"
        else
            echo "No return data found"
        fi
        
        # Check for any errors
        error_count=$(grep -i "error\|exception\|traceback" "$log_file" | wc -l)
        echo "Errors: $error_count"
        echo
    fi
done

echo "=== Best Performing Configurations ==="
for log_file in logs/*/cuda_*.log; do
    if [ -f "$log_file" ]; then
        max_return=$(grep -o "returns: [0-9.-]*" "$log_file" | cut -d' ' -f2 | sort -n | tail -1)
        if [ ! -z "$max_return" ]; then
            echo "$(basename $log_file): $max_return"
        fi
    fi
done | sort -k2 -n | tail -5
EOF
chmod +x analyze_results.sh

echo -e "\n${GREEN}All experiments launched!${NC}"
echo -e "${BLUE}Logs directory: logs/${BASE_EXP_NAME}/${NC}"
echo -e "${BLUE}Monitor progress: ./analyze_results.sh${NC}"
echo -e "${BLUE}Kill all experiments: ./kill_sweep.sh${NC}"

# Start monitoring
monitor_experiments 