@echo off
setlocal

set "DEFAULT_NOTEBOOK=Emotion_Detection_mlp.ipynb"

if "%~1"=="" (
    set "NB_NAME=%DEFAULT_NOTEBOOK%"
) else (
    set "NB_NAME=%~1"
)

set "FULL_PATH=notebooks\%NB_NAME%"

echo Activating environment...
call venv\Scripts\activate

echo Opening: %FULL_PATH%

jupyter notebook "%FULL_PATH%"
pause