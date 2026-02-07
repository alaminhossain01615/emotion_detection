@echo off
python -m venv venv
call venv\Scripts\activate

echo Installing Torch with CUDA 12.4 support...
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo Installing other dependencies and project logic...
pip install -e .

python -m ipykernel install --user --name=ml_project --display-name "(Emotion_detection_grp_I))"
echo Setup Complete.
pause