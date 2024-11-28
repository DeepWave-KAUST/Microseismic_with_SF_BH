#!/bin/bash
# 
# Installer for packages
# 
# Run: ./install_env.sh
# 

echo 'Creating package environment'

# create conda env
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate forge
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'
