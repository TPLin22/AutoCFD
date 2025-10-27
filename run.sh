#!/bin/bash

source activate autocfd

case_name=Couette_flow-laminar
export CONFIG_FILE_PATH=Benchmark/${case_name}.yaml
python src/main.py