@echo off
rem Evalua todos los modelos de modelos\*.pt en dataset\val_camara (nunca en juliaca\test).
rem Resultados en reportes\<modelo>_val_camara.json y reportes\val_camara_por_foto.csv.
setlocal enabledelayedexpansion
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No existe .venv en %CD%. Crealo como en el README, seccion 4.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
set "PESOS="
for %%f in (modelos\*.pt) do set PESOS=!PESOS! "%%f"
if not defined PESOS (
    echo No hay modelos en modelos\*.pt
    pause
    exit /b 1
)
python scripts\evaluar.py --pesos !PESOS! --datos val_camara %*
pause
