#!/bin/bash

set -euo pipefail

source venv/bin/activate
python src/dataset_builder.py  --max_workers 4 --input_dir ../data/RamanArunachalamData/input --output_dir ../data/RamanArunachalamData/vocal_data
#python src/dataset_builder.py  --max_workers 8 --input_dir ../data/simple-test/input --output_dir ../data/simple-test/vocal_data
deactivate
