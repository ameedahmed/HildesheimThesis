#!/bin/bash
# Shell script to run the modified Vizfiletest.py with conda environment

# Change to the base directory
cd "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main"

# Activate conda environment (replace 'your_env_name' with your actual conda environment name)
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate your_env_name

# Check if conda environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment activated successfully"
    echo "Running modified Vizfiletest.py..."
    
    # Run the Python script
    python "/data/vifapi/ameed_ahmed_thesis/HildesheimThesis/CSWin_Transformer_main/ImageCounterfactualExplanations/Vizfiletest_no_overlap.py"
else
    echo "Failed to activate conda environment"
    echo "Please check your conda environment name and try again"
    exit 1
fi
