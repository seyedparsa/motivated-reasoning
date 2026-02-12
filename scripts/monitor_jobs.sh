#!/bin/bash
# Monitor all eval_llm jobs by model/dataset/bias combination
# Tracks the most recent job for each of the 36 combinations
#
# Usage: ./monitor_jobs_full.sh [-r|--running]
#   -r, --running   Show only running jobs

RUNNING_ONLY=false
if [[ "$1" == "-r" || "$1" == "--running" ]]; then
    RUNNING_ONLY=true
fi

echo "=== Job Monitor ($(date '+%H:%M:%S')) ==="
echo ""

# Define all combinations
models=(qwen-3-8b llama-3.1-8b gemma-3-4b)
datasets=(arc-challenge mmlu aqua commonsense_qa)
biases=(expert self metadata)

# Get list of running job IDs
running_jobs=$(squeue -u $USER -h -o "%i" 2>/dev/null | tr '\n' '|' | sed 's/|$//')

# Temporary files
tmpfile_running=$(mktemp)
tmpfile_done=$(mktemp)
tmpfile_failed=$(mktemp)
tmpfile_missing=$(mktemp)

# Process each combination
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for bias in "${biases[@]}"; do
            name="${model}_${dataset}_${bias}"

            # Find the best .err file: prefer running jobs, then most recently modified
            errfile=""
            for f in $(ls -t eval_llm_${name}_*.err 2>/dev/null); do
                fid=$(echo "$f" | grep -oE '[0-9]{7}')
                if [ -n "$running_jobs" ] && echo "$running_jobs" | grep -q "$fid"; then
                    errfile="$f"
                    break
                fi
            done
            # If no running job, use most recently modified file
            [ -z "$errfile" ] && errfile=$(ls -t eval_llm_${name}_*.err 2>/dev/null | head -1)

            if [ -z "$errfile" ]; then
                echo "$name" >> "$tmpfile_missing"
                continue
            fi

            # Extract job ID
            job_id=$(echo "$errfile" | grep -oE '[0-9]{7}')

            # Check job status
            if [ -n "$running_jobs" ] && echo "$running_jobs" | grep -q "$job_id"; then
                # RUNNING - get progress
                last_line=$(tail -c 500 "$errfile" 2>/dev/null | tr '\r' '\n' | grep "Labeling\|it/s\]" | tail -1)

                if [ -n "$last_line" ]; then
                    progress=$(echo "$last_line" | grep -oE '[0-9]+/[0-9]+' | head -1)
                    eta=$(echo "$last_line" | sed -n 's/.*<\([^,]*\),.*/\1/p')
                    pct=$(echo "$last_line" | grep -oE '[0-9]+%' | head -1)

                    if [ -n "$eta" ] && [ -n "$progress" ]; then
                        # Convert ETA to seconds for sorting
                        if echo "$eta" | grep -qE '^[0-9]+:[0-9]+:[0-9]+$'; then
                            hours=$((10#$(echo "$eta" | cut -d: -f1)))
                            mins=$((10#$(echo "$eta" | cut -d: -f2)))
                            secs=$((10#$(echo "$eta" | cut -d: -f3)))
                            total_secs=$((hours * 3600 + mins * 60 + secs))
                        elif echo "$eta" | grep -qE '^[0-9]+:[0-9]+$'; then
                            mins=$((10#$(echo "$eta" | cut -d: -f1)))
                            secs=$((10#$(echo "$eta" | cut -d: -f2)))
                            total_secs=$((mins * 60 + secs))
                        else
                            total_secs=0
                        fi
                        printf "%06d|%8s|%6s|%8s|%s|%s\n" "$total_secs" "$eta" "$pct" "$progress" "$name" "$job_id" >> "$tmpfile_running"
                    else
                        # Running but no progress yet
                        printf "%06d|%8s|%6s|%8s|%s|%s\n" "999999" "starting" "-" "-" "$name" "$job_id" >> "$tmpfile_running"
                    fi
                else
                    # Running but no progress output yet
                    printf "%06d|%8s|%6s|%8s|%s|%s\n" "999999" "starting" "-" "-" "$name" "$job_id" >> "$tmpfile_running"
                fi
            elif grep -q "Traceback\|Error:" "$errfile" 2>/dev/null; then
                # FAILED - get error type
                error_type=$(grep -oE '(RateLimitError|APIError|Timeout|KeyError|ValueError|RuntimeError|CUDA|OOM|OutOfMemory)' "$errfile" | tail -1)
                [ -z "$error_type" ] && error_type="Unknown"
                echo "$error_type|$name|$job_id" >> "$tmpfile_failed"
            else
                # Check if completed (look for 100% in progress or successful output)
                outfile="${errfile%.err}.out"
                if [ -f "$outfile" ] && grep -q "Saved\|saved\|100%" "$outfile" 2>/dev/null; then
                    # Get final count
                    final=$(tail -c 500 "$errfile" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+' | tail -1)
                    [ -z "$final" ] && final="done"
                    echo "$final|$name|$job_id" >> "$tmpfile_done"
                elif tail -c 500 "$errfile" 2>/dev/null | tr '\r' '\n' | grep -qE '100%|completed'; then
                    final=$(tail -c 500 "$errfile" 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+' | tail -1)
                    [ -z "$final" ] && final="done"
                    echo "$final|$name|$job_id" >> "$tmpfile_done"
                else
                    # Unknown state - might still be starting or completed without marker
                    echo "?|$name|$job_id" >> "$tmpfile_done"
                fi
            fi
        done
    done
done

# Display RUNNING jobs
running_count=0
if [ -s "$tmpfile_running" ]; then
    running_count=$(wc -l < "$tmpfile_running")
    echo "RUNNING ($running_count jobs):"
    echo "ETA      | Prog  | Done     | Job ID  | Job"
    echo "---------|-------|----------|---------|------------------------------------"
    sort -rn "$tmpfile_running" | cut -d'|' -f2- | while IFS='|' read eta pct progress name job_id; do
        printf "%8s | %5s | %8s | %7s | %s\n" "$eta" "$pct" "$progress" "$job_id" "$name"
    done
    echo ""
fi

# Display COMPLETED jobs
done_count=0
if [ -s "$tmpfile_done" ]; then
    done_count=$(wc -l < "$tmpfile_done")
    if [ "$RUNNING_ONLY" = false ]; then
        echo "COMPLETED ($done_count jobs):"
        echo "Final    | Job ID  | Job"
        echo "---------|---------|------------------------------------"
        sort -t'|' -k2 "$tmpfile_done" | while IFS='|' read final name job_id; do
            printf "%8s | %7s | %s\n" "$final" "$job_id" "$name"
        done
        echo ""
    fi
fi

# Display FAILED jobs
failed_count=0
if [ -s "$tmpfile_failed" ]; then
    failed_count=$(wc -l < "$tmpfile_failed")
    if [ "$RUNNING_ONLY" = false ]; then
        echo "FAILED ($failed_count jobs):"
        echo "Error          | Job ID  | Job"
        echo "---------------|---------|------------------------------------"
        sort -t'|' -k2 "$tmpfile_failed" | while IFS='|' read error name job_id; do
            printf "%-14s | %7s | %s\n" "$error" "$job_id" "$name"
        done
        echo ""
    fi
fi

# Display MISSING jobs (no job file found)
missing_count=0
if [ -s "$tmpfile_missing" ]; then
    missing_count=$(wc -l < "$tmpfile_missing")
    if [ "$RUNNING_ONLY" = false ]; then
        echo "NOT SUBMITTED ($missing_count jobs):"
        sort "$tmpfile_missing" | while read name; do
            echo "  $name"
        done
        echo ""
    fi
fi

# Summary
total=$((running_count + done_count + failed_count))
if [ "$RUNNING_ONLY" = false ]; then
    echo "=== Summary ==="
    echo "Running:       $running_count"
    echo "Completed:     $done_count"
    echo "Failed:        $failed_count"
    echo "Not submitted: $missing_count"
    echo "Total tracked: $total / 36"
else
    echo "($running_count running, $done_count done, $failed_count failed)"
fi

# Cleanup
rm -f "$tmpfile_running" "$tmpfile_done" "$tmpfile_failed" "$tmpfile_missing"
