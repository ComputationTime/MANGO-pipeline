#!/bin/bash

# Environment set up (as done in BindCraft)
CONDA_BASE=$(conda info --base 2>/dev/null) || { echo -e "Error: conda is not installed or cannot be initialised."; exit 1; }
conda create --name MANGO_env python=3.11
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/MANGO_env


# Name of the directory where weights will be stored and Zenodo file download link
WEIGHTS_DIR="mango/trained_models/"
WEIGHTS_URL=""
clear

# True workhorse for downloading the stuff
echo "Creating directory and downloading model weights from Zenodo..."
mkdir -p "$WEIGHTS_DIR"
wget -O "$WEIGHTS_DIR/MANGO.zip" "$WEIGHTS_URL"
# mv "Banana.zip" "$WEIGHTS_DIR"
unzip -o "$WEIGHTS_DIR/MANGO.zip" -d "$WEIGHTS_DIR"
clear

echo "Weights successfully downloaded, installing MANGO..."
pip install -e .
clear

echo "Cleaning up excess files..."
rm "$WEIGHTS_DIR/MANGO.zip"
rm -r MANGO.egg-info/ build/

# Get a dummy pdb for users to start with
echo "Fetching a copy of 

echo "Lastly installing pyrosetta. Will do this for now while license is temporarily hard to find..."
pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release

echo "Successfully installed MANGO 🥭 and downloaded it's weights! Have fun 🐵"