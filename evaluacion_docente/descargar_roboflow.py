"""Alternativa a los zips manuales: descarga con la API de Roboflow a imagenes/_descargas/<clave>/
y luego indexa con preparar_test.py. La API key se lee de ROBOFLOW_API_KEY (no se guarda en el repo).

Proyectos conocidos (README de los zips):
  v4  : rupiah-detector/garbage-classification-xxgap/4
  med : visalatchi-t-bme-student/smart-medical-waste-segregation/3
  v1i : (pendiente: copiar workspace/proyecto de la URL de Universe del dataset "garbage" v1)

  $env:ROBOFLOW_API_KEY = "xxxx"
  python -I evaluacion_docente/descargar_roboflow.py --v1i <ws>/<proy>/1
"""
import argparse
import os
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import preparar_test as pt  # noqa: E402

CONOCIDOS = {"v4": "rupiah-detector/garbage-classification-xxgap/4",
             "med": "visalatchi-t-bme-student/smart-medical-waste-segregation/3"}


def main():
    ap = argparse.ArgumentParser()
    for k in pt.DATASETS:
        ap.add_argument(f"--{k}", help="workspace/proyecto/version")
    a = ap.parse_args()
    if "ROBOFLOW_API_KEY" not in os.environ:
        sys.exit("Falta ROBOFLOW_API_KEY")
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    for k in pt.DATASETS:
        ref = getattr(a, k)
        if not ref:
            continue
        ws, proy, ver = ref.strip("/").split("/")
        rf.workspace(ws).project(proy).version(int(ver)).download(
            "folder", location=str(pt.DESC / k), overwrite=False)
    pt.indexar()


if __name__ == "__main__":
    main()
