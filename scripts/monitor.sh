#!/bin/bash
# Job monitor - shows status of recent jobs
# Usage: ./scripts/monitor.sh [options]
#   -s, --since JOB_ID   Only show jobs >= JOB_ID (default: last submitted batch)
#   -n, --last N         Only show last N jobs
#   -w, --watch          Continuous monitoring (refresh every 5s)
#   -a, --all            Show all jobs (no limit)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINCE_JOB=""
LAST_N=""
WATCH_MODE=false
SHOW_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--since) SINCE_JOB="$2"; shift 2 ;;
        -n|--last) LAST_N="$2"; shift 2 ;;
        -w|--watch) WATCH_MODE=true; shift ;;
        -a|--all) SHOW_ALL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Default: use last submitted batch if no filter specified
if [ -z "$SINCE_JOB" ] && [ -z "$LAST_N" ] && [ "$SHOW_ALL" = false ]; then
    if [ -f "${SCRIPT_DIR}/.last_submit" ]; then
        SINCE_JOB=$(cat "${SCRIPT_DIR}/.last_submit")
    else
        LAST_N=20  # fallback
    fi
fi

monitor_once() {
    if [ "$WATCH_MODE" = true ]; then
        clear 2>/dev/null || true
    fi
    echo "=== Job Monitor ($(date '+%Y-%m-%d %H:%M:%S')) ==="

    # Get running/pending job IDs from squeue (exclude CG)
    running_jobs=$(squeue -u $USER -h -t R,PD -o "%i" 2>/dev/null | tr '\n' ' ')

    # Get stuck CG jobs (TIME 0:00 means they never ran)
    stuck_jobs=$(squeue -u $USER -h -t CG -o "%i %M" 2>/dev/null | awk '$2=="0:00" {print $1}' | tr '\n' ' ')

    # Build list of job files, sorted by job ID (descending)
    if [ "$SHOW_ALL" = true ] || [ -z "$LAST_N" ]; then
        job_files=$(ls -1 job_*.err 2>/dev/null | sed 's/.*_\([0-9]\{7\}\)\.err/\1 &/' | sort -rn | cut -d' ' -f2)
    else
        job_files=$(ls -1 job_*.err 2>/dev/null | sed 's/.*_\([0-9]\{7\}\)\.err/\1 &/' | sort -rn | head -${LAST_N} | cut -d' ' -f2)
    fi

    # Filter by --since if specified
    if [ -n "$SINCE_JOB" ]; then
        job_files=$(echo "$job_files" | while read f; do
            jid=$(echo "$f" | grep -oE '[0-9]{7}')
            [ "$jid" -ge "$SINCE_JOB" ] && echo "$f"
        done)
    fi

    running_count=0
    completed_count=0
    failed_count=0
    stuck_count=0

    # Arrays for output
    running_lines=""
    completed_lines=""
    failed_lines=""
    stuck_lines=""

    for errfile in $job_files; do
        [ -z "$errfile" ] && continue
        job_id=$(echo "$errfile" | grep -oE '[0-9]{7}')
        # Parse job name: job__MODEL__DATASET__BIAS__PROBE_JOBID.err (new format)
        # or: job_MODEL_DATASET_BIAS_PROBE_JOBID.err (old format)
        job_name=$(echo "$errfile" | sed "s/_${job_id}.err//")

        # Check if new format (uses __ delimiter)
        if echo "$job_name" | grep -q "__"; then
            # New format: job__MODEL__DATASET__BIAS__PROBE
            job_name=$(echo "$job_name" | sed 's/^job__//')
            model=$(echo "$job_name" | cut -d'_' -f1-2 | sed 's/__.*$//' | cut -d'_' -f1)
            # Split by __ and extract parts
            model=$(echo "$job_name" | awk -F'__' '{print $1}')
            dataset=$(echo "$job_name" | awk -F'__' '{print $2}')
            bias=$(echo "$job_name" | awk -F'__' '{print $3}')
            probe=$(echo "$job_name" | awk -F'__' '{print $4}')
        else
            # Old format: job_MODEL_DATASET_BIAS_PROBE
            job_name=$(echo "$job_name" | sed 's/^job_//')
            model=$(echo "$job_name" | cut -d'_' -f1)
            dataset=$(echo "$job_name" | cut -d'_' -f2)
            bias=$(echo "$job_name" | cut -d'_' -f3)
            probe=$(echo "$job_name" | cut -d'_' -f4-)
        fi

        # Check if stuck (CG with TIME 0:00)
        if echo " $stuck_jobs " | grep -q " $job_id "; then
            stuck_lines="${stuck_lines}${job_id}|${model}|${dataset}|${bias}|${probe}|NodeFail\n"
            stuck_count=$((stuck_count + 1))

        # Check if running
        elif echo " $running_jobs " | grep -q " $job_id "; then
            # RUNNING - get progress
            outfile="${errfile%.err}.out"
            progress=""
            if [ -f "$outfile" ]; then
                last_progress=$(tail -c 3000 "$outfile" 2>/dev/null | grep -E "\[PROGRESS\]" | tail -1 | sed 's/.*\[PROGRESS\] //')
                [ -n "$last_progress" ] && progress="$last_progress"
            fi
            if [ -z "$progress" ]; then
                # Try tqdm from stderr
                last_tqdm=$(tail -c 3000 "$errfile" 2>/dev/null | tr '\r' '\n' | grep -E "[0-9]+%" | tail -1)
                if [ -n "$last_tqdm" ]; then
                    pct=$(echo "$last_tqdm" | grep -oE '[0-9]+%' | tail -1)
                    progress="$pct"
                fi
            fi
            [ -z "$progress" ] && progress="starting"

            running_lines="${running_lines}${job_id}|${model}|${dataset}|${bias}|${probe}|${progress}\n"
            running_count=$((running_count + 1))

        elif grep -qE "Traceback|Error:|CUDA|OOM|Killed|EOFError" "$errfile" 2>/dev/null; then
            # FAILED
            error_type=$(grep -oE '(EOFError|RuntimeError|FileNotFoundError|ValueError|KeyError|CUDA|OOM)' "$errfile" 2>/dev/null | tail -1)
            [ -z "$error_type" ] && error_type="Error"
            failed_lines="${failed_lines}${job_id}|${model}|${dataset}|${bias}|${probe}|${error_type}\n"
            failed_count=$((failed_count + 1))

        else
            # Check for completion
            outfile="${errfile%.err}.out"
            if grep -qE "\[DONE\]|Done!|Upserted" "$outfile" 2>/dev/null; then
                completed_lines="${completed_lines}${job_id}|${model}|${dataset}|${bias}|${probe}\n"
                completed_count=$((completed_count + 1))
            else
                # No completion marker and not running = failed
                failed_lines="${failed_lines}${job_id}|${model}|${dataset}|${bias}|${probe}|Unknown\n"
                failed_count=$((failed_count + 1))
            fi
        fi
    done

    # Handle stuck CG jobs that have no .err files
    for stuck_id in $stuck_jobs; do
        [ -z "$stuck_id" ] && continue
        # Skip if already processed (has .err file)
        if ls job_*${stuck_id}.err >/dev/null 2>&1; then
            continue
        fi
        # Get job name from squeue
        job_info=$(squeue -j "$stuck_id" -h -o "%j" 2>/dev/null)
        if [ -n "$job_info" ]; then
            # Parse job name
            if echo "$job_info" | grep -q "__"; then
                job_name=$(echo "$job_info" | sed 's/^job__//')
                model=$(echo "$job_name" | awk -F'__' '{print $1}')
                dataset=$(echo "$job_name" | awk -F'__' '{print $2}')
                bias=$(echo "$job_name" | awk -F'__' '{print $3}')
                probe=$(echo "$job_name" | awk -F'__' '{print $4}')
            else
                job_name=$(echo "$job_info" | sed 's/^job_//')
                model=$(echo "$job_name" | cut -d'_' -f1)
                dataset=$(echo "$job_name" | cut -d'_' -f2)
                bias=$(echo "$job_name" | cut -d'_' -f3)
                probe=$(echo "$job_name" | cut -d'_' -f4-)
            fi
            stuck_lines="${stuck_lines}${stuck_id}|${model}|${dataset}|${bias}|${probe}|NodeFail\n"
            stuck_count=$((stuck_count + 1))
        fi
    done

    echo ""

    # Display RUNNING
    if [ $running_count -gt 0 ]; then
        echo "RUNNING ($running_count):"
        printf "  %-8s %-12s %-15s %-10s %-12s %s\n" "JOB_ID" "MODEL" "DATASET" "BIAS" "PROBE" "PROGRESS"
        printf "  %-8s %-12s %-15s %-10s %-12s %s\n" "-------" "-----" "-------" "----" "-----" "--------"
        echo -e "$running_lines" | while IFS='|' read jid model dataset bias probe progress; do
            [ -z "$jid" ] && continue
            printf "  %-8s %-12s %-15s %-10s %-12s %s\n" "$jid" "$model" "$dataset" "$bias" "$probe" "$progress"
        done
        echo ""
    fi

    # Display COMPLETED
    if [ $completed_count -gt 0 ]; then
        echo "COMPLETED ($completed_count):"
        echo -e "$completed_lines" | while IFS='|' read jid model dataset bias probe; do
            [ -z "$jid" ] && continue
            printf "  %s: %s / %s / %s / %s\n" "$jid" "$model" "$dataset" "$bias" "$probe"
        done
        echo ""
    fi

    # Display FAILED
    if [ $failed_count -gt 0 ]; then
        echo "FAILED ($failed_count):"
        echo -e "$failed_lines" | while IFS='|' read jid model dataset bias probe error; do
            [ -z "$jid" ] && continue
            printf "  %s: %s / %s / %s / %s (%s)\n" "$jid" "$model" "$dataset" "$bias" "$probe" "$error"
        done
        echo ""
    fi

    # Display STUCK (CG jobs that never ran)
    if [ $stuck_count -gt 0 ]; then
        echo "STUCK ($stuck_count):"
        echo -e "$stuck_lines" | while IFS='|' read jid model dataset bias probe error; do
            [ -z "$jid" ] && continue
            printf "  %s: %s / %s / %s / %s (%s)\n" "$jid" "$model" "$dataset" "$bias" "$probe" "$error"
        done
        echo ""
    fi

    # Summary
    echo "=== Summary: $running_count running, $completed_count completed, $failed_count failed, $stuck_count stuck ==="
}

if [ "$WATCH_MODE" = true ]; then
    while true; do
        monitor_once
        sleep 5
    done
else
    monitor_once
fi
