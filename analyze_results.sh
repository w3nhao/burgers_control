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
