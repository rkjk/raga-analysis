#!/bin/bash

# Create virtual environment called venv
python3 -m venv venv

source venv/bin/activate

# install portaudio - a dependency of pyaudio
sudo apt-get update && sudo apt-get install -y portaudio19-dev

# Install python dependencies
pip install --upgrade pip
pip install -r requirements.txt

deactivate
