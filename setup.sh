#!/bin/bash

echo "Creating virtual environment..."
python3 -m venv venv

if [ -d "./venv/Scripts" ]; then
    source ./venv/Scripts/activate
else
    source ./venv/bin/activate
fi

echo "Installing Torch with CUDA 12.4 support..."
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing other dependencies and project logic..."
pip install -e .

python -m ipykernel install --user --name=ml_project --display-name "(Emotion_detection_grp_I)"

echo "Setup Complete"