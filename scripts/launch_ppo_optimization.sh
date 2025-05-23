#!/bin/bash

# Master PPO Optimization Launcher
# Systematic approach to reach theoretical maximum of 9.5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}======================================${NC}"
echo -e "${PURPLE}  PPO Optimization Suite${NC}"
echo -e "${PURPLE}  Target: 9.5 (Current: ~5.0)${NC}"
echo -e "${PURPLE}======================================${NC}"

# Check GPU availability
check_gpus() {
    echo -e "\n${BLUE}Checking available GPUs...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name total used; do
            utilization=$(echo "scale=1; $used / $total * 100" | bc -l 2>/dev/null || echo "N/A")
            echo -e "GPU ${idx}: ${name} (${utilization}% memory used)"
        done
    else
        echo -e "${RED}nvidia-smi not found. Cannot check GPU status.${NC}"
    fi
}

# Make scripts executable
chmod +x run_hyperparameter_sweep.sh
chmod +x run_extreme_sweep.sh

# Menu function
show_menu() {
    echo -e "\n${GREEN}Choose optimization strategy:${NC}"
    echo "1. Standard Hyperparameter Sweep (8 configs, CUDA 0-7)"
    echo "2. Extreme Hyperparameter Sweep (8 configs, CUDA 0-7)"
    echo "3. Both Standard + Extreme (16 configs, CUDA 0-15)"
    echo "4. Custom GPU range for Standard sweep"
    echo "5. Check GPU status only"
    echo "6. Analyze existing results"
    echo "7. Kill all running experiments"
    echo "8. Exit"
}

# Custom GPU range function
run_custom_standard() {
    echo -e "\n${YELLOW}Enter CUDA device range for standard sweep:${NC}"
    read -p "Start GPU (default 0): " start_gpu
    read -p "End GPU (default 7): " end_gpu
    
    start_gpu=${start_gpu:-0}
    end_gpu=${end_gpu:-7}
    
    echo -e "${GREEN}Running standard sweep on GPU ${start_gpu}-${end_gpu}${NC}"
    
    # Modify the script temporarily
    sed "s/for i in {0..7}/for i in {${start_gpu}..${end_gpu}}/g" run_hyperparameter_sweep.sh > temp_sweep.sh
    sed -i "s/run_experiment [0-7]/run_experiment \$((${start_gpu}+\${i}))/g" temp_sweep.sh
    chmod +x temp_sweep.sh
    
    ./temp_sweep.sh
    rm temp_sweep.sh
}

# Results monitoring
monitor_results() {
    echo -e "\n${BLUE}Setting up results monitoring...${NC}"
    
    # Create enhanced monitoring script
    cat > enhanced_monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== PPO Optimization Results Monitor ==="
    echo "$(date)"
    echo
    
    # Find latest log directory
    latest_dir=$(ls -td logs/*/ 2>/dev/null | head -1)
    if [ -z "$latest_dir" ]; then
        echo "No log directories found."
        sleep 10
        continue
    fi
    
    echo "Monitoring: $latest_dir"
    echo
    
    # Check each GPU
    for log_file in ${latest_dir}cuda_*.log; do
        if [ -f "$log_file" ]; then
            gpu_id=$(basename "$log_file" | grep -o 'cuda_[0-9]*' | cut -d'_' -f2)
            config=$(basename "$log_file" | sed 's/cuda_[0-9]*_//;s/.log//')
            
            # Get latest return value
            latest_return=$(tail -20 "$log_file" | grep -o "returns: [0-9.-]*" | tail -1 | cut -d' ' -f2)
            max_return=$(grep -o "returns: [0-9.-]*" "$log_file" | cut -d' ' -f2 | sort -n | tail -1)
            
            # Check if process is running
            pid_file="${latest_dir}cuda_${gpu_id}_${config}.pid"
            if [ -f "$pid_file" ]; then
                pid=$(cat "$pid_file")
                if ps -p $pid > /dev/null 2>&1; then
                    status="RUNNING"
                else
                    status="FINISHED"
                fi
            else
                status="UNKNOWN"
            fi
            
            printf "GPU %-2s %-20s %-10s Latest: %-6s Max: %-6s\n" \
                   "$gpu_id" "$config" "$status" "${latest_return:-N/A}" "${max_return:-N/A}"
        fi
    done
    
    echo
    echo "Press Ctrl+C to exit monitoring"
    sleep 10
done
EOF
    chmod +x enhanced_monitor.sh
    
    echo -e "${GREEN}Starting enhanced monitor...${NC}"
    ./enhanced_monitor.sh
}

# Main execution
check_gpus

while true; do
    show_menu
    read -p "Enter your choice (1-8): " choice
    
    case $choice in
        1)
            echo -e "\n${GREEN}Launching Standard Hyperparameter Sweep...${NC}"
            ./run_hyperparameter_sweep.sh
            ;;
        2)
            echo -e "\n${GREEN}Launching Extreme Hyperparameter Sweep...${NC}"
            ./run_extreme_sweep.sh
            ;;
        3)
            echo -e "\n${GREEN}Launching Both Standard + Extreme Sweeps...${NC}"
            echo -e "${YELLOW}Standard sweep on CUDA 0-7, Extreme on CUDA 8-15${NC}"
            
            # Modify extreme sweep to use CUDA 8-15
            sed 's/run_experiment [0-7]/run_experiment $((8+&))/g' run_extreme_sweep.sh > temp_extreme.sh
            chmod +x temp_extreme.sh
            
            ./run_hyperparameter_sweep.sh &
            sleep 5
            ./temp_extreme.sh &
            
            wait
            rm temp_extreme.sh
            ;;
        4)
            run_custom_standard
            ;;
        5)
            check_gpus
            ;;
        6)
            ./analyze_results.sh 2>/dev/null || echo "No results to analyze yet"
            ;;
        7)
            ./kill_sweep.sh 2>/dev/null || echo "No experiments to kill"
            ;;
        8)
            echo -e "${GREEN}Exiting...${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
    
    if [[ $choice =~ ^[1-4]$ ]]; then
        echo -e "\n${BLUE}Would you like to monitor results? (y/n)${NC}"
        read -p "Monitor: " monitor_choice
        if [[ $monitor_choice =~ ^[Yy]$ ]]; then
            monitor_results
        fi
    fi
done 