#!/bin/bash

URL=$1

OUT=$2

yt-dlp -f bestaudio --extract-audio --audio-format mp3 --postprocessor-args "-ar 44100" $URL -o $OUT
