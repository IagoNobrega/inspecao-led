@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Preparando o ambiente...
  python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
python desktop_app.py
pause
