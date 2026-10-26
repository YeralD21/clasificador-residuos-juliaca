"""Toma fotos REALES con la MX Brio y las guarda en el split y objeto que TÚ eliges.

Uso:
  python scripts/capturar_fotos.py --split train
  python scripts/capturar_fotos.py --split test --camara 1

Flujo: elige la clase con su tecla y presiona ESPACIO; cada foto se guarda en
  dataset/juliaca/<split>/<clase>/objNNN/
Presiona N cuando cambies de objeto físico (otra botella, otra lata) -> objNNN+1.
Así todas las fotos de un mismo objeto quedan juntas y en UN solo split.

Teclas:
  1 papel_carton  2 metales  3 plasticos  4 vidrio  5 organico
  6 peligrosos    7 generales  0 sin_residuo
  ESPACIO = tomar foto de la clase activa   N = nuevo objeto   Q = salir
Luego puedes renombrar la carpeta objNNN a algo descriptivo (obj007_lata_pilsen).
"""
import argparse
import re
import time

import cv2

from comun import CLASES, JULIACA, SPLITS, abrir_camara, imagenes, roi_cuadrado

TECLAS = {ord("1"): "papel_carton", ord("2"): "metales", ord("3"): "plasticos", ord("4"): "vidrio",
          ord("5"): "organico", ord("6"): "peligrosos", ord("7"): "generales", ord("0"): "sin_residuo"}


def siguiente_objeto(clase):
    """Número de objeto libre mirando los 3 splits (los IDs no se repiten entre splits)."""
    usados = [0]
    for s in SPLITS:
        d = JULIACA / s / clase
        if d.exists():
            for sub in d.iterdir():
                m = re.match(r"obj(\d+)", sub.name)
                if sub.is_dir() and m:
                    usados.append(int(m.group(1)))
    return max(usados) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--camara", default="auto", help="índice de OpenCV, o auto (MX Brio si está, si no la integrada)")
    args = ap.parse_args()

    cap = abrir_camara(args.camara)
    clase = "plasticos"
    obj = siguiente_objeto(clase)
    fotos_obj = 0
    msg, hasta = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        recorte, (x1, y1, x2, y2) = roi_cuadrado(frame)
        vista = frame.copy()
        cv2.rectangle(vista, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vista, f"SPLIT={args.split}  CLASE={clase}  OBJETO=obj{obj:03d} ({fotos_obj} fotos)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vista, "ESPACIO foto | N nuevo objeto | 0-7 clase | Q salir",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if time.time() < hasta:
            cv2.putText(vista, msg, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Captura de residuos", vista)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        if k in TECLAS and TECLAS[k] != clase:
            clase = TECLAS[k]
            obj, fotos_obj = siguiente_objeto(clase), 0
        elif k in (ord("n"), ord("N")):
            obj, fotos_obj = siguiente_objeto(clase), 0
            msg, hasta = f"Nuevo objeto: obj{obj:03d}", time.time() + 1.2
        elif k == 32:  # espacio
            carpeta = JULIACA / args.split / clase / f"obj{obj:03d}"
            carpeta.mkdir(parents=True, exist_ok=True)
            f = carpeta / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            cv2.imwrite(str(f), recorte, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fotos_obj += 1
            msg, hasta = f"Guardada ({fotos_obj})", time.time() + 0.8
            print(f"[OK] {f.relative_to(JULIACA)}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nFotos por clase en", args.split)
    for c in CLASES:
        print(f"  {c:14s} {len(imagenes(JULIACA / args.split / c))}")


if __name__ == "__main__":
    main()
