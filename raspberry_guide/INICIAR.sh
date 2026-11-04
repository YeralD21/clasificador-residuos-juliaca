#!/usr/bin/env bash
# =====================================================================
#  Clasificador de residuos - Juliaca  |  TODO EN UNO
#  1) Instala lo necesario (solo la primera vez o si falta algo)
#  2) Abre la camara con el modelo
#  Uso:  bash INICIAR.sh            (opciones: --rapido, --camara 1,
#                                     --sin-ventana, --umbral 0.7)
# =====================================================================
cd "$(dirname "$0")"
VERSION="8.4.184"   # version de ultralytics con la que se entreno el modelo

pausa() { echo ""; read -r -p "Presiona Enter para cerrar..." _; }
fallo() { echo ""; echo "[ERROR] $1"; pausa; exit 1; }

instalar() {
  echo ""
  echo "Primera ejecucion: se instalara todo lo necesario."
  echo "Necesita internet y tarda unos 10-20 minutos. Puede pedir la clave de la Raspberry."

  echo ""
  echo "== 1/4 Verificando sistema =="
  ARCH=$(uname -m)
  echo "Arquitectura: $ARCH"
  if [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armv6l" ]; then
    fallo "El sistema es de 32 bits. PyTorch necesita Raspberry Pi OS de 64 bits.
        Reinstala la tarjeta SD con 'Raspberry Pi OS (64-bit)' usando Raspberry Pi Imager."
  fi
  python3 --version || fallo "No se encontro python3."

  echo "== 2/4 Paquetes del sistema =="
  sudo apt-get update || fallo "No se pudo actualizar (revisa el internet)."
  sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0 v4l-utils \
    || fallo "No se pudieron instalar los paquetes del sistema."

  echo "== 3/4 Entorno virtual + librerias =="
  rm -rf .venv
  python3 -m venv --system-site-packages .venv || fallo "No se pudo crear el entorno .venv"
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt \
    || fallo "Fallo la instalacion de librerias. Revisa el internet y vuelve a ejecutar: bash INICIAR.sh"

  echo "== 4/4 Comprobacion =="
  .venv/bin/python -c "import ultralytics, torch, cv2; print('ultralytics', ultralytics.__version__); print('torch      ', torch.__version__); print('opencv     ', cv2.__version__)" \
    || fallo "Las librerias no cargan correctamente."
  echo "Camaras detectadas:"
  v4l2-ctl --list-devices 2>/dev/null || echo "  (ninguna USB detectada)"
  echo ""
  echo "[OK] Instalacion completa."
}

echo "=============================================="
echo "  Clasificador de residuos - Juliaca"
echo "=============================================="

ls modelo/*.pt >/dev/null 2>&1 || fallo "No se encontro el modelo .pt en la carpeta 'modelo/'."
[ -f clasificar_camara.py ] || fallo "Falta el archivo clasificar_camara.py"
[ -f requirements.txt ]     || fallo "Falta el archivo requirements.txt"

# Hace falta instalar? (no existe el entorno o la version no coincide)
v=""
if [ -x .venv/bin/python ]; then
  v=$(.venv/bin/python -c "from importlib.metadata import version; print(version('ultralytics'))" 2>/dev/null)
fi
if [ "$v" = "$VERSION" ]; then
  echo "[OK] Todo instalado. Iniciando..."
else
  instalar
fi

echo ""
.venv/bin/python clasificar_camara.py "$@"
codigo=$?
if [ "$codigo" -ne 0 ]; then
  echo ""
  echo "[ERROR] El clasificador termino con un error (ver mensajes de arriba)."
  echo "        Consulta 'Solucion de problemas' en README.md"
  pausa
fi
exit "$codigo"
