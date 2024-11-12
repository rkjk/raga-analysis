#!/bin/bash

set -euo pipefail

source venv/bin/activate
python src/dataset_builder.py --input_dir ../data/simple-test --output_dir ../data/simple-test/pitch_data
deactivate