"""Etiqueta A MANO cada objeto de una sesión de capturar_sesion.py: clase y split, por objeto. No toca dataset/.

  python scripts/etiquetar_sesion.py                    # la fecha más reciente de D:\\Proyecto-Residuos-ML\\fotos_laptop
  python scripts/etiquetar_sesion.py --fecha 2026-10-26

Por cada fotos_laptop/<fecha>/objNNN/ muestra un mosaico de sus fotos. Eliges la CLASE con su tecla y luego el
SPLIT (T train, V val); al elegir el split se guarda la fila. test no se ofrece: se arma aparte, a mano.
Teclas:
  1 papel_carton  2 metales  3 plasticos  4 vidrio  5 organico  6 peligrosos  7 generales  0 sin_residuo
  T train  V val  |  S saltar (vuelve a salir la próxima vez)  D descartar  Z deshacer el último  Q salir
Resultado: fotos_laptop/<fecha>/etiquetas.csv (objeto, clase, split, n_fotos, nota). Se escribe tras cada
decisión; si ya existe, retoma: los objetos ya etiquetados o descartados no vuelven a salir. La columna `nota`
se puede llenar a mano en el CSV. Después: python scripts/importar_sesion.py
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np

from capturar_fotos import TECLAS
from comun import CLASES, EXT, RAIZ, crear_ventana, teclas_ventana

FOTOS = RAIZ.parent / "fotos_laptop"   # D:\Proyecto-Residuos-ML\fotos_laptop (fuera del repo)
CAMPOS = ["objeto", "clase", "split", "n_fotos", "nota"]
CLASE_DE = {chr(k): v for k, v in TECLAS.items()}
SPLIT_DE = {"t": "train", "v": "val"}
VENTANA = "Etiquetar sesion - clase 0-7, split T/V | S saltar D descartar Z deshacer Q salir"


def ultima_fecha(raiz):
    fechas = sorted(d.name for d in raiz.iterdir() if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name))
    if not fechas:
        raise SystemExit(f"No hay sesiones (carpetas AAAA-MM-DD) en {raiz}")
    return fechas[-1]


def fotos_de(obj):
    return sorted(p for p in obj.iterdir() if p.is_file() and p.suffix.lower() in EXT)


def leer(csv_path):
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def escribir(csv_path, filas):
    tmp = csv_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(filas)
    tmp.replace(csv_path)  # no deja el CSV a medias si se corta


def mosaico(fotos, ancho=1280, cols=4, filas=3):
    """Hasta cols*filas miniaturas repartidas entre todas las fotos del objeto."""
    elegidas = [fotos[int(i)] for i in np.linspace(0, len(fotos) - 1, min(len(fotos), cols * filas))]
    lado = ancho // cols
    celdas = []
    for p in elegidas:
        im = cv2.imread(str(p))
        celda = np.full((lado, lado, 3), 30, np.uint8)
        if im is not None:
            e = lado / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * e), int(im.shape[0] * e)), interpolation=cv2.INTER_AREA)
            y, x = (lado - im.shape[0]) // 2, (lado - im.shape[1]) // 2
            celda[y:y + im.shape[0], x:x + im.shape[1]] = im
        celdas.append(celda)
    celdas += [np.full((lado, lado, 3), 30, np.uint8)] * (-len(celdas) % cols)
    return np.vstack([np.hstack(celdas[i:i + cols]) for i in range(0, len(celdas), cols)])


def cabecera(ancho, lineas):
    img = np.full((40 + 34 * len(lineas), ancho, 3), 20, np.uint8)
    for i, (t, color, esc) in enumerate(lineas):
        cv2.putText(img, t, (14, 36 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, esc, color, 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=str(FOTOS), help="carpeta con las sesiones AAAA-MM-DD")
    ap.add_argument("--fecha", help="AAAA-MM-DD (por defecto la más reciente)")
    args = ap.parse_args()

    raiz = Path(args.raiz)
    dia = raiz / (args.fecha or ultima_fecha(raiz))
    objetos = sorted(d for d in dia.glob("obj*") if d.is_dir() and fotos_de(d))
    csv_path = dia / "etiquetas.csv"
    filas = leer(csv_path)
    hechos = {f["objeto"] for f in filas}
    pendientes = [o for o in objetos if o.name not in hechos]
    print(f"Sesión {dia}: {len(objetos)} objetos, {len(hechos)} ya en etiquetas.csv, {len(pendientes)} pendientes")
    if not pendientes:
        print("Nada que etiquetar.")
        return

    crear_ventana(VENTANA)
    i, clase, saltados = 0, None, 0
    while i < len(pendientes):
        obj = pendientes[i]
        fotos = fotos_de(obj)
        img_obj = mosaico(fotos)
        while True:
            elegido = f"clase: {clase}  -> ahora T (train) o V (val)" if clase else "elige la CLASE (0-7)"
            cab = cabecera(img_obj.shape[1], [
                (f"{obj.name}   {len(fotos)} fotos   (pendiente {i + 1} de {len(pendientes)})", (0, 255, 255), 0.9),
                (elegido, (255, 255, 255), 0.8),
                ("1 papel_carton 2 metales 3 plasticos 4 vidrio 5 organico 6 peligrosos 7 generales 0 sin_residuo",
                 (180, 180, 180), 0.55),
                ("T train  V val  |  S saltar  D descartar  Z deshacer  Q salir", (180, 180, 180), 0.55),
            ])
            cv2.imshow(VENTANA, np.vstack([cab, img_obj]))
            k = teclas_ventana(VENTANA, 30)
            if k == "q":
                cv2.destroyAllWindows()
                print(f"Guardado en {csv_path}. Para seguir, vuelve a ejecutar el script.")
                return
            if k in CLASE_DE:
                clase = CLASE_DE[k]
            elif k in SPLIT_DE and clase:
                filas.append({"objeto": obj.name, "clase": clase, "split": SPLIT_DE[k], "n_fotos": len(fotos),
                              "nota": ""})
                break
            elif k == "d":
                filas.append({"objeto": obj.name, "clase": "descartado", "split": "", "n_fotos": len(fotos),
                              "nota": ""})
                break
            elif k == "s":
                saltados += 1
                break
            elif k == "z":
                if clase:
                    clase = None  # primero deshace la clase elegida sin confirmar
                elif filas and filas[-1]["objeto"] in {o.name for o in pendientes[:i]}:
                    ultimo = filas.pop()
                    escribir(csv_path, filas)
                    print(f"Deshecho: {ultimo['objeto']} ({ultimo['clase']})")
                    i = [o.name for o in pendientes].index(ultimo["objeto"])
                    obj, fotos = pendientes[i], fotos_de(pendientes[i])
                    img_obj = mosaico(fotos)
        if k != "s":
            escribir(csv_path, filas)
            f = filas[-1]
            print(f"{f['objeto']}: {f['clase']} {f['split']}".rstrip())
        clase = None
        i += 1

    cv2.destroyAllWindows()
    print(f"\nListo: {csv_path} ({len(filas)} objetos; {saltados} saltados, salen la próxima vez)")
    print("Revisa el CSV y luego: python scripts/importar_sesion.py --dry-run")


if __name__ == "__main__":
    main()
