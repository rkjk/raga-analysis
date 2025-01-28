#!/bin/bash

URL=$1

OUT=$2

echo "Downloading $1 and saving at $2"

yt-dlp --extract-audio --audio-format mp3 --postprocessor-args "-ar 44100" $URL -o $OUT
