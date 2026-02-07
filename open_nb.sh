#!/bin/bash

DEFAULT_NB="Emotion_Detection_r.ipynb"

NB_NAME=${1:-$DEFAULT_NB}


FULL_PATH="notebooks/$NB_NAME"


if [ -f "./venv/Scripts/activate" ]; then
    source ./venv/Scripts/activate
elif [ -f "./venv/bin/activate" ]; then
    source ./venv/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

echo "Opening: $FULL_PATH"

jupyter notebook "$FULL_PATH"