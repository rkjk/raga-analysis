#!/bin/bash

set -euo pipefail

source venv/bin/activate
python pitch-detect.py
deactivate