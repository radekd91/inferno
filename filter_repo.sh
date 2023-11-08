#!/bin/bash

# Path to your text file
FILE_LIST_PATH="../inferno_files_filter_additional4.txt"

# Read the text file line by line
while IFS= read -r file_path; do
  # Use git filter-repo to remove the file from the history
  echo "$file_path"
  git filter-repo --invert-paths --path "$file_path"
done < "$FILE_LIST_PATH"

# After all files are processed, you can optionally run garbage collection to clean up
#git reflog expire --expire=now --all
#git gc --prune=now --aggressive
