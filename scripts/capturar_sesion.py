"""Sesión de fotos con la cámara de la laptop (MX Brio): guarda cada foto TAL CUAL, a resolución completa.

  python scripts/capturar_sesion.py                 # cámara auto, D:\\Proyecto-Residuos-ML\\fotos_laptop\\<fecha>\\
  python scripts/capturar_sesion.py --camara 1

Las fotos van FUERA del repo y de dataset/: <destino>/<AAAA-MM-DD>/objNNN/<hora>.jpg (JPG calidad 95, el frame
de la cámara sin recortar ni redimensionar). La clase y el split los decides TÚ después, a mano.
Cada vez que se abre empieza un objeto nuevo (el siguiente objNNN libre de ese día).

Teclas: ESPACIO o clic = foto | N = nuevo objeto | F = pantalla completa | Q = salir
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from comun import RAIZ, abrir_camara, crear_ventana, teclas_ventana

DESTINO = RAIZ.parent / "fotos_laptop"   # D:\Proyecto-Residuos-ML\fotos_laptop (fuera del repo)
VENTANA = "Captura - ESPACIO/clic foto | N nuevo objeto | Q salir"


def siguiente_objeto(dia):
    usados = [int(m.group(1)) for d in dia.glob("obj*") if d.is_dir() and (m := re.match(r"obj(\d+)", d.name))]
    return max(usados, default=0) + 1


def texto(img, t, y, escala, color=(255, 255, 255)):
    cv2.putText(img, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 5, cv2.LINE_AA)  # borde negro
    cv2.putText(img, t, (16, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", default="auto", help="índice de OpenCV, o auto (MX Brio si está, si no la integrada)")
    ap.add_argument("--destino", default=str(DESTINO))
    ap.add_argument("--ancho", type=int, default=3840, help="resolución pedida; la cámara da la mayor que soporte")
    ap.add_argument("--alto", type=int, default=2160)
    args = ap.parse_args()

    dia = Path(args.destino) / time.strftime("%Y-%m-%d")
    cap = abrir_camara(args.camara, args.ancho, args.alto)
    dia.mkdir(parents=True, exist_ok=True)  # después de abrir la cámara: si falla no deja carpetas vacías
    obj, n_obj, total = siguiente_objeto(dia), 0, 0
    clic = {"foto": False}

    def raton(evento, *_):
        if evento == cv2.EVENT_LBUTTONDOWN:
            clic["foto"] = True

    crear_ventana(VENTANA)
    cv2.setMouseCallback(VENTANA, raton)
    destello = 0.0
    print(f"Guardando en {dia}  (objeto actual obj{obj:03d})")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        k = teclas_ventana(VENTANA)
        if k == "q":
            break
        if k == "n":
            if n_obj:
                obj, n_obj = obj + 1, 0
                print(f"Nuevo objeto: obj{obj:03d}")
            else:  # un objeto sin fotos no avanza: así no quedan números saltados
                print(f"obj{obj:03d} todavía no tiene fotos: sigue siendo el objeto actual")
        if k == " " or clic["foto"]:
            clic["foto"] = False
            carpeta = dia / f"obj{obj:03d}"
            carpeta.mkdir(exist_ok=True)
            f = carpeta / f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            if cv2.imwrite(str(f), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):  # el frame tal cual, sin dibujos
                n_obj, total, destello = n_obj + 1, total + 1, time.time()
                print(f"[OK] {f}  ({w}x{h})")
            else:
                print(f"[ERROR] no se pudo guardar {f}")

        e = 1280 / w  # la vista previa se achica; lo guardado es el frame completo
        vista = cv2.resize(frame, (1280, int(h * e)), interpolation=cv2.INTER_AREA)
        if (t := time.time() - destello) < 0.25:  # destello blanco que se apaga en 0.25 s
            a = 0.8 * (1 - t / 0.25)
            vista = cv2.addWeighted(vista, 1 - a, np.full_like(vista, 255), a, 0)
        texto(vista, f"OBJETO obj{obj:03d}   fotos: {n_obj}   (total sesion: {total})", 44, 1.1, (0, 255, 255))
        texto(vista, f"{dia}   {w}x{h}", 80, 0.6)
        texto(vista, "ESPACIO o clic = foto   N = nuevo objeto   F = pantalla completa   Q = salir",
              vista.shape[0] - 20, 0.6)
        cv2.imshow(VENTANA, vista)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{total} fotos en {dia}")
    for d in sorted(dia.glob("obj*")):
        print(f"  {d.name}: {sum(1 for _ in d.glob('*.jpg'))} fotos")


if __name__ == "__main__":
    main()
