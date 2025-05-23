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
