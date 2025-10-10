#!/bin/bash

# Declare an associative array to store progress by model
declare -A model_progress

# Loop through all .err files
for err_file in *.err; do
    # Get the base name of the file
    base_name="${err_file%.err}"
    out_file="$base_name.out"

    # Check if the .out file exists
    if [ -f "$out_file" ]; then
        # Get the model name from the .out file
        model_name=$(grep "Loading model:" "$out_file" | sed 's/Loading model: //')

        # Get the last line of the .err file
        last_line=$(tail -n 1 "$err_file")

        # Check for Error
        if grep -q "Error" "$err_file"; then
            if echo "$last_line" | grep -q "OutOfMemoryError"; then
                progress="Error: OOM"
            else
                progress="Error"
            fi
        else
            # Extract the percentage from the last line
            progress=$(echo "$last_line" | grep -o -E '[0-9]+%' | tail -n 1)
            
            # If progress is not found, check if the job is 100% complete
            if [ -z "$progress" ]; then
                if grep -q "100%" "$err_file"; then
                    progress="100%"
                else
                    progress="N/A"
                fi
            fi
        fi

        # Append the job progress to the model's entry in the associative array
        if [ -n "$model_name" ]; then
            # Using a temporary variable to handle the multiline string
            current_progress="${model_progress[$model_name]}"
            new_progress="- $base_name: $progress"
            
            if [ -z "$current_progress" ]; then
                model_progress["$model_name"]="$new_progress"
            else
                model_progress["$model_name"]="$current_progress
$new_progress"
            fi
        fi
    fi
done

# Print the categorized progress report
for model_name in "${!model_progress[@]}"; do
    echo "### $model_name"
    echo "${model_progress[$model_name]}"
done