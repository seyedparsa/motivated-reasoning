#!/bin/bash

for err_file in $(grep -l -E "error|Error|Upload|Creating|OutOfMemoryError" *.err); do
  out_file="${err_file%.err}.out"
  echo "Deleting $err_file and $out_file"
  rm "$err_file" "$out_file"
done
