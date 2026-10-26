@echo off
rem Sesion de fotos: ESPACIO o clic = foto, N = nuevo objeto, Q = salir.
rem Guarda en D:\Proyecto-Residuos-ML\fotos_laptop\<fecha>\objNNN\ (fuera del repo y de dataset\).
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No existe .venv en %CD%. Crealo como en el README, seccion 4.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\capturar_sesion.py %*
if errorlevel 1 pause
