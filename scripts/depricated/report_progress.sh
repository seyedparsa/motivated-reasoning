#!/bin/bash

# Declare associative arrays to store progress by model and status
declare -A model_finished
declare -A model_in_progress
declare -A model_errored
declare -A model_failed

# Function to extract short job name from filename
# Formats: 
#   - probe_<model>_<dataset>_<bias>_<probe_type>_<job_id>.out
#   - bias_probe_<model>_<dataset>_<bias>_<job_id>.out
# Returns: <dataset>/<bias>/<probe_type> or <dataset>/<bias>
extract_job_name() {
    local base_name="$1"
    # Remove prefixes and job ID suffix
    local job_part=$(echo "$base_name" | sed 's/^\(bias_\)\?probe_//' | sed 's/_[0-9]*$//')
    
    # Known patterns to identify boundaries
    # Bias types: expert, self, metadata
    # Probe types: bias, has-switched, will-switch
    # Model names typically end with version like: _3_4b, _3_8b, _3_1_8b
    
    # Find the bias type (expert, self, or metadata)
    local bias=""
    local probe=""
    local dataset=""
    
    # Check for probe type first (at the end)
    if echo "$job_part" | grep -qE '_(bias|has-switched|will-switch)$'; then
        probe=$(echo "$job_part" | grep -oE '_(bias|has-switched|will-switch)$' | sed 's/^_//')
        job_part=$(echo "$job_part" | sed 's/_[^_]*$//')
    fi
    
    # Find bias type
    if echo "$job_part" | grep -qE '_(expert|self|metadata)$'; then
        bias=$(echo "$job_part" | grep -oE '_(expert|self|metadata)$' | sed 's/^_//')
        dataset=$(echo "$job_part" | sed 's/_[^_]*$//')
    else
        # Fallback: take last part as bias
        bias=$(echo "$job_part" | awk -F'_' '{print $NF}')
        dataset=$(echo "$job_part" | sed 's/_[^_]*$//')
    fi
    
    # Remove model name (everything before dataset)
    # Model names typically have version numbers like _3_4b, _3_8b, _3_1_8b
    # Remove patterns that look like model versions: <word>_<number>_<number>[b]? or <word>_<number>_<number>_<number>[b]?
    # Keep removing until we hit a known dataset name or until we've removed enough parts
    # Common datasets: aqua, mmlu, arc_challenge, commonsense_qa
    # Try to remove model name by looking for version-like patterns
    dataset=$(echo "$dataset" | sed -E 's/^[^_]+_[0-9]+(_[0-9]+)?[b]?_//' | sed -E 's/^[^_]+_[0-9]+(_[0-9]+)?[b]?_//')
    
    # If still starts with version-like pattern, remove one more time
    if echo "$dataset" | grep -qE '^[0-9]+[b]?_'; then
        dataset=$(echo "$dataset" | sed -E 's/^[0-9]+[b]?_//')
    fi
    
    # Build result
    if [ -n "$probe" ]; then
        echo "$dataset/$bias/$probe"
    else
        echo "$dataset/$bias"
    fi
}

# Function to format job label with optional job ID
format_job_label() {
    local job_id="$1"
    local job_name="$2"

    if [ -n "$job_id" ]; then
        echo "[$job_id] $job_name"
    else
        echo "$job_name"
    fi
}

# Loop through all .out files
for out_file in *.out; do
    # Skip if file doesn't exist (e.g., no .out files)
    [ -f "$out_file" ] || continue
    
    # Get the base name of the file
    base_name="${out_file%.out}"
    err_file="$base_name.err"
    
    # Get the model name from the .out file
    model_name=$(grep "Loading model:" "$out_file" | sed 's/Loading model: //' | head -n 1)
    
    # If no model name found, skip this file
    [ -z "$model_name" ] && continue
    
    # Extract short job name
    job_name=$(extract_job_name "$base_name")
    job_id=$(echo "$base_name" | grep -oE '[0-9]+$' | tail -1)
    job_label=$(format_job_label "$job_id" "$job_name")
    
    # Check for errors in .err file if it exists
    if [ -f "$err_file" ]; then
        if grep -q "Error" "$err_file"; then
            last_line=$(tail -n 1 "$err_file")
            if echo "$last_line" | grep -q "OutOfMemoryError"; then
                error_msg="OOM"
            else
                error_msg="Error"
            fi
            # Add to errored jobs
            current="${model_errored[$model_name]}"
            new_entry="  • $job_label: $error_msg"
            if [ -z "$current" ]; then
                model_errored["$model_name"]="$new_entry"
            else
                model_errored["$model_name"]="$current
$new_entry"
            fi
            continue
        fi
    fi
    
    # Look for "Layers to probe:" line in the output
    layers_line=$(grep "Layers to probe:" "$out_file" | head -n 1)
    
    if [ -n "$layers_line" ]; then
        # Extract the list of layers from the line
        # Format: "Layers to probe: [0, 1, 2, 9, 17, 25, 32, 33, 34]"
        layers_str=$(echo "$layers_line" | sed 's/.*Layers to probe: \[\(.*\)\].*/\1/')
        
        # Count total layers by counting commas and adding 1
        # Remove spaces and count commas
        layers_clean=$(echo "$layers_str" | tr -d ' ')
        if [ -z "$layers_clean" ]; then
            total_layers=0
        else
            total_layers=$(echo "$layers_clean" | awk -F',' '{print NF}')
        fi
        
        # Count completed layers by looking for "Layer X:" patterns (universal probes)
        # Look for patterns like "Layer 0: RFM Val" which indicates universal probe completion
        universal_layers=$(grep -E "Layer [0-9]+: RFM Val" "$out_file" | sed 's/.*Layer \([0-9]\+\):.*/\1/' | sort -u)
        
        # Count step-specific progress: unique (layer, step) combinations
        # Also extract unique layers from step-specific results
        step_results=$(grep -E "Layer [0-9]+, Step [0-9]+: RFM (Val )?(Acc|AUC)" "$out_file" | sed 's/.*Layer \([0-9]\+\), Step \([0-9]\+\):.*/\1:\2/' | sort -u)
        step_layers=$(grep -E "Layer [0-9]+, Step [0-9]+: RFM (Val )?(Acc|AUC)" "$out_file" | sed 's/.*Layer \([0-9]\+\), Step.*/\1/' | sort -u)
        
        # Combine both universal and step-specific layers
        all_processed_layers=$(echo -e "${universal_layers}\n${step_layers}" | sort -u | grep -v '^$')
        
        # Count how many unique layers have been processed
        if [ -n "$all_processed_layers" ]; then
            completed_layers=$(echo "$all_processed_layers" | wc -l | tr -d ' ')
        else
            completed_layers=0
        fi
        
        completed_steps=0
        if [ -n "$step_results" ]; then
            completed_steps=$(echo "$step_results" | wc -l | tr -d ' ')
        fi
        
        # Build progress string
        if [ "$total_layers" -gt 0 ]; then
            percentage=$((completed_layers * 100 / total_layers))
            
            # Determine status and format progress
            if [ "$percentage" -eq 100 ]; then
                # Finished - just show job name
                current="${model_finished[$model_name]}"
                new_entry="  • $job_label"
                if [ -z "$current" ]; then
                    model_finished["$model_name"]="$new_entry"
                else
                    model_finished["$model_name"]="$current
$new_entry"
                fi
            else
                # Check if job is still running
                is_running=0
                if [ -n "$job_id" ] && [ -n "${running_job_ids[$job_id]}" ]; then
                    is_running=1
                fi
                
                # Only mark as in-progress if job is actually running
                if [ "$is_running" -eq 1 ]; then
                    # In progress
                    if [ "$completed_steps" -gt 0 ]; then
                        progress="$completed_layers/$total_layers layers ($percentage%, $completed_steps steps)"
                    else
                        progress="$completed_layers/$total_layers layers ($percentage%)"
                    fi
                    current="${model_in_progress[$model_name]}"
                    new_entry="  • $job_label: $progress"
                    if [ -z "$current" ]; then
                        model_in_progress["$model_name"]="$new_entry"
                    else
                        model_in_progress["$model_name"]="$current
$new_entry"
                    fi
                else
                    # Job not running but not 100% - likely failed or was interrupted
                    if [ "$completed_steps" -gt 0 ]; then
                        progress="$completed_layers/$total_layers layers ($percentage%, $completed_steps steps)"
                    else
                        progress="$completed_layers/$total_layers layers ($percentage%)"
                    fi
                    current="${model_failed[$model_name]}"
                    new_entry="  • $job_label: $progress (stopped)"
                    if [ -z "$current" ]; then
                        model_failed["$model_name"]="$new_entry"
                    else
                        model_failed["$model_name"]="$current
$new_entry"
                    fi
                fi
            fi
        else
            # No layer info - check if job is still running
            is_running=0
            if [ -n "$job_id" ] && [ -n "${running_job_ids[$job_id]}" ]; then
                is_running=1
            fi
            
            if [ "$is_running" -eq 1 ]; then
                current="${model_in_progress[$model_name]}"
                new_entry="  • $job_label: Started (no layer info)"
                if [ -z "$current" ]; then
                    model_in_progress["$model_name"]="$new_entry"
                else
                    model_in_progress["$model_name"]="$current
$new_entry"
                fi
            fi
        fi
    else
        # No layers line found, try to detect if it's a probe training job
        # Look for any probe-related output or if it's a probe job file
        if echo "$base_name" | grep -qE '^(bias_)?probe_' || grep -q -E "(Layer [0-9]+:|train_probes|probe|Loading model)" "$out_file"; then
            # Check if job is still running
            is_running=0
            if [ -n "$job_id" ] && [ -n "${running_job_ids[$job_id]}" ]; then
                is_running=1
            fi
            
            # Only add to in-progress if job is actually running
            if [ "$is_running" -eq 1 ]; then
                # Check for other progress indicators
                last_line=$(tail -n 1 "$out_file" 2>/dev/null)
                if echo "$last_line" | grep -q -E '[0-9]+%'; then
                    progress=$(echo "$last_line" | grep -o -E '[0-9]+%' | tail -n 1)
                    current="${model_in_progress[$model_name]}"
                    new_entry="  • $job_label: $progress"
                else
                    # Job started but no progress info yet
                    current="${model_in_progress[$model_name]}"
                    new_entry="  • $job_label: Started"
                fi
                if [ -z "$current" ]; then
                    model_in_progress["$model_name"]="$new_entry"
                else
                    # Check if this job is already in the list to avoid duplicates
                    if ! echo "$current" | grep -F -q "$job_label"; then
                        model_in_progress["$model_name"]="$current
$new_entry"
                    fi
                fi
            fi
        fi
    fi
done

# Also check the job queue for running/queued jobs that don't have .out files yet
# Get all running/queued probe jobs
declare -A processed_job_ids
declare -A running_job_ids
# Mark all job IDs we've already processed (from .out files)
for out_file in *.out; do
    [ -f "$out_file" ] || continue
    base_name="${out_file%.out}"
    # Extract job ID from filename (last number before .out)
    job_id=$(echo "$base_name" | grep -oE '[0-9]+$' | tail -1)
    if [ -n "$job_id" ]; then
        processed_job_ids["$job_id"]="$base_name"
    fi
done

# Get all currently running/queued job IDs from squeue
if command -v squeue >/dev/null 2>&1; then
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*JOBID ]] && continue
        job_id=$(echo "$line" | awk '{print $1}' | tr -d '[:space:]')
        [[ "$job_id" =~ ^[0-9]+$ ]] && running_job_ids["$job_id"]=1
    done < <(squeue -u "${USER}" -o "%.18i" 2>/dev/null | grep -E '^[[:space:]]*[0-9]')
fi

# Get jobs from squeue and process them
# Use process substitution to avoid subshell issues with associative arrays
if command -v squeue >/dev/null 2>&1; then
    while IFS= read -r line; do
        # Skip header line
        [[ "$line" =~ ^[[:space:]]*JOBID ]] && continue
        
        # Extract job ID
        job_id=$(echo "$line" | awk '{print $1}' | tr -d '[:space:]')
        
        # Skip if empty or not a number
        [[ ! "$job_id" =~ ^[0-9]+$ ]] && continue
        
        # Check if we've already processed this job (has .out file)
        if [ -n "${processed_job_ids[$job_id]}" ]; then
            continue
        fi
        
        # Extract job name (may be truncated)
        job_name_partial=$(echo "$line" | awk '{for(i=3;i<=NF-2;i++) printf "%s ", $i; print ""}' | sed 's/[[:space:]]*$//')
        
        # Skip non-probe jobs
        if ! echo "$job_name_partial" | grep -qE '^(bias_)?probe_'; then
            continue
        fi
        
        # Try to find the full job name by matching job ID with files
        job_base=""
        for file in *.out *.err; do
            [ -f "$file" ] || continue
            if echo "$file" | grep -qE "${job_id}\.(out|err)$"; then
                job_base="${file%.out}"
                job_base="${job_base%.err}"
                break
            fi
        done
        
        # If we found a file, use it; otherwise construct from partial name
        if [ -z "$job_base" ]; then
            # Try to find files that start with the partial name
            for file in *.out *.err; do
                [ -f "$file" ] || continue
                file_base="${file%.out}"
                file_base="${file_base%.err}"
                if echo "$file_base" | grep -qE "^${job_name_partial}"; then
                    job_base="$file_base"
                    break
                fi
            done
        fi
        
        # Try to get model name
        model_name=""
        if [ -n "$job_base" ] && [ -f "${job_base}.out" ]; then
            model_name=$(grep "Loading model:" "${job_base}.out" | sed 's/Loading model: //' | head -n 1)
        fi
        
        # If still no model name, try to extract from job name
        if [ -z "$model_name" ]; then
            name_to_check="${job_base:-$job_name_partial}"
            if echo "$name_to_check" | grep -qE '^probe_'; then
                model_part=$(echo "$name_to_check" | sed 's/^probe_//' | sed 's/_[0-9]*$//')
                if echo "$model_part" | grep -qE '^gemma'; then
                    model_name="gemma-3-4b"
                elif echo "$model_part" | grep -qE '^llama'; then
                    model_name="llama-3.1-8b"
                elif echo "$model_part" | grep -qE '^qwen'; then
                    model_name="qwen-3-8b"
                fi
            elif echo "$name_to_check" | grep -qE '^bias_probe_'; then
                model_part=$(echo "$name_to_check" | sed 's/^bias_probe_//' | sed 's/_[0-9]*$//')
                if echo "$model_part" | grep -qE '^qwen'; then
                    model_name="qwen-3-8b"
                fi
            fi
        fi
        
        # Skip if we can't determine model name
        [ -z "$model_name" ] && continue
        
        # Extract short job name
        job_name=$(extract_job_name "${job_base:-$job_name_partial}")
        job_label=$(format_job_label "$job_id" "$job_name")
        
        # Check job status from squeue
        job_status=$(echo "$line" | awk '{print $(NF-1)}')
        
        # Add to in-progress jobs
        current="${model_in_progress[$model_name]}"
        if [ "$job_status" = "R" ]; then
            new_entry="  • $job_label: Running (no output yet)"
        else
            new_entry="  • $job_label: Queued"
        fi
        
        if [ -z "$current" ]; then
            model_in_progress["$model_name"]="$new_entry"
        else
            # Check if this job is already in the list
            if ! echo "$current" | grep -F -q "$job_label"; then
                model_in_progress["$model_name"]="$current
$new_entry"
            fi
        fi
    done < <(squeue -u "${USER}" -o "%.18i %.20P %.50j %.2t %.10M" 2>/dev/null | grep -E 'probe_|bias_probe_' | grep -v '^[[:space:]]*JOBID')
fi

# Print the categorized progress report
# Collect all unique model names
declare -A all_models
for model in "${!model_finished[@]}" "${!model_in_progress[@]}" "${!model_errored[@]}" "${!model_failed[@]}"; do
    all_models["$model"]=1
done

# Print each model's status
for model_name in $(printf '%s\n' "${!all_models[@]}" | sort); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $model_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Print finished jobs
    if [ -n "${model_finished[$model_name]}" ]; then
        finished_count=$(echo "${model_finished[$model_name]}" | grep -c '^  •' || echo "0")
        echo ""
        echo "✅ Finished ($finished_count):"
        echo "${model_finished[$model_name]}"
    fi
    
    # Print in-progress jobs
    if [ -n "${model_in_progress[$model_name]}" ]; then
        in_progress_count=$(echo "${model_in_progress[$model_name]}" | grep -c '^  •' || echo "0")
        echo ""
        echo "🔄 In Progress ($in_progress_count):"
        echo "${model_in_progress[$model_name]}"
    fi
    
    # Print failed/incomplete jobs
    if [ -n "${model_failed[$model_name]}" ]; then
        failed_count=$(echo "${model_failed[$model_name]}" | grep -c '^  •' || echo "0")
        echo ""
        echo "⚠️  Failed/Incomplete ($failed_count):"
        echo "${model_failed[$model_name]}"
    fi
    
    # Print errored jobs
    if [ -n "${model_errored[$model_name]}" ]; then
        errored_count=$(echo "${model_errored[$model_name]}" | grep -c '^  •' || echo "0")
        echo ""
        echo "❌ Errors ($errored_count):"
        echo "${model_errored[$model_name]}"
    fi
    
    echo ""
done
