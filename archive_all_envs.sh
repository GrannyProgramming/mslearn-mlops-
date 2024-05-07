#!/bin/bash

# Directory to store the archived environment files
ARCHIVE_DIR="/mnt/c/Users/Alex/Documents/myCode/ARCHIVED/conda-env-archived"

# Create the archive directory if it does not exist
mkdir -p "$ARCHIVE_DIR"

# List of environments to archive
ENVS="blog demo fast-api-health fastapi-health gpt hc_vault openai passive_inc pf py3.8 py3.9 pyautogen ubs_aml_pf"

# Loop through the environments and archive them
for ENV in $ENVS; do
    echo "Archiving $ENV..."
    conda env export --name $ENV > "$ARCHIVE_DIR/$ENV.yml"
    
    # Remove the environment after archiving
    conda env remove --name $ENV --yes
done

echo "All specified environments have been archived and removed."
