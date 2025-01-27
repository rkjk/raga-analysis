#!/bin/bash

set -euo pipefail

source venv/bin/activate
python src/dataset_builder.py  --max_workers 12 --input_dir ../data/simple-test/input --output_dir ../data/simple-test/pitch_data_midi
deactivate