@echo off
rem Demo en vivo: deteccion + clasificacion con los valores por defecto (camara auto, modelo actual).
rem Doble clic, o "demo" en cmd desde la raiz del repo (".\demo" en PowerShell). Acepta argumentos: demo --debug
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No existe .venv en %CD%. Crealo como en el README, seccion 4.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\inferencia_deteccion.py %*
if errorlevel 1 pause
