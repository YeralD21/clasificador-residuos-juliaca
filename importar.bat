@echo off
rem Importa a dataset\juliaca la ultima sesion ya etiquetada (etiquetas.csv).
rem Sin argumentos (doble clic) SOLO muestra lo que haria (--dry-run). Para copiar de verdad: importar --importar
rem Otros argumentos pasan al script: importar --importar --fecha 2026-10-26 --recorte roi
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No existe .venv en %CD%. Crealo como en el README, seccion 4.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if /i "%~1"=="--importar" (
    python scripts\importar_sesion.py %2 %3 %4 %5 %6 %7 %8 %9
    if not errorlevel 1 python scripts\auditar.py
) else (
    python scripts\importar_sesion.py --dry-run %*
    echo.
    echo Para importar de verdad: importar --importar
)
pause
