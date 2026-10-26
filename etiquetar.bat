@echo off
rem Etiqueta a mano los objetos de la ultima sesion de capturar.bat: clase (0-7) y split (T train, V val).
rem Guarda en D:\Proyecto-Residuos-ML\fotos_laptop\<fecha>\etiquetas.csv y retoma si ya existe. Otra fecha: etiquetar --fecha 2026-10-26
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No existe .venv en %CD%. Crealo como en el README, seccion 4.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python scripts\etiquetar_sesion.py %*
pause
