#!/bin/bash

set -euo pipefail

source venv/bin/activate
python src/live.py
deactivate
