#!/usr/bin/env bash
echo "Clearing datasets now..."

# Script to delete training run directories created by Julia script
# Directories follow the pattern: training-run-XX (where XX is zero-padded number)

# Set the data directory path
# Modify this path to match your actual data_dir variable from Julia
DATA_DIR="./data"  # Current directory - change this to your actual data_dir path

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Find and list training run directories
echo "Searching for training run directories in: $DATA_DIR"
TRAINING_DIRS=$(find "$DATA_DIR" -maxdepth 1 -type d -name "training-run-[0-9][0-9]" | sort)

if [ -z "$TRAINING_DIRS" ]; then
    echo "No training run directories found matching pattern 'training-run-XX'."
    exit 0
fi

# Display found directories
echo "Found the following training run directories:"
echo "$TRAINING_DIRS"
echo

# Prompt for confirmation
read -p "Are you sure you want to delete these directories? This action cannot be undone. (y/N): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Delete directories
    echo "Deleting training run directories..."
    echo "$TRAINING_DIRS" | while read -r dir; do
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            echo "Deleting: $dir"
            rm -rf "$dir"
            if [ $? -eq 0 ]; then
                echo "  ✓ Successfully deleted"
            else
                echo "  ✗ Failed to delete"
            fi
        fi
    done
    echo "Deletion process completed."
else
    echo "Operation cancelled."
    exit 0
fi


echo "Clearing dataset info files"

