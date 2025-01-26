#!/bin/bash

source venv/bin/activate
pytest -rP tests/pyin_test.py
deactivate
