#!/bin/bash

# Target sample rate
TARGET_SR=44100

# Original sample rate to check
ORIGINAL_SR=48000

# Find all MP3 files in the current directory and subdirectories
find . -type f -iname "*.mp3" -print0 | while read -d $'\0' file; do
  # Get the current sample rate of the file
  current_sr=$(ffprobe -v error -select_streams a -show_entries stream=sample_rate -of csv=p=0 "$file")

  # Check if the current sample rate is 48 kHz
  if [ "$current_sr" -eq "$ORIGINAL_SR" ]; then
    # Resample the file to 44.1 kHz
    output_file="${file%.mp3}-resampled.mp3"
    sox "$file" -r $TARGET_SR "$output_file"
    echo "Resampled $file to $output_file"
  else
    echo "Skipping $file as it is not at 48 kHz"
  fi
done

