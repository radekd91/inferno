#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

# Assign input and output folders from script arguments
INPUT_FOLDER="$1"
OUTPUT_FOLDER="$2"

# Find video files in the input folder and process each one
find "$INPUT_FOLDER" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.mov' -o -iname '*.flv' -o -iname '*.wmv' \) -print0 | while IFS= read -r -d $'\0' file; do
    # Create a corresponding output file path
    OUTPUT_FILE="${OUTPUT_FOLDER}/${file#$INPUT_FOLDER/}"
    OUTPUT_FILE_DIR=$(dirname "$OUTPUT_FILE")
    
    # Ensure the output directory exists
    mkdir -p "$OUTPUT_FILE_DIR"
    
    # Convert the video file to 25fps using ffmpeg
    ffmpeg -i "$file" -r 25 "$OUTPUT_FILE"
done
