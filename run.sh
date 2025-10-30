#!/bin/bash

source activate autocfd

case_name=Couette_flow-laminar
export CONFIG_FILE_PATH=Benchmark/${case_name}.yaml
log_dir="log"
mkdir -p "$log_dir"
log_file="${log_dir}/run.log"

python src/main.py | tee "$log_file"