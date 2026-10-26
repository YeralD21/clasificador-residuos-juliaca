"""Compara detectores de inferencia_deteccion.py en las fotos de dataset/val_camara. No entrena ni modifica el dataset.

  python scripts/comparar_detectores.py                                   # yoloe-26n-seg vs yoloe-26s-seg
  python scripts/comparar_detectores.py --detectores yoloe-26n-seg.pt yoloe-26m-seg.pt

val_camara no tiene cajas anotadas (solo la clase de cada foto), así que "la caja cubre el objeto" se mide con
aproximaciones, por foto, con los filtros por defecto de inferencia_deteccion.py y la caja principal (mayor área):
  caja      : queda al menos una caja de residuo tras los filtros
  centro    : la caja principal contiene el centro de la foto (los objetos se fotografiaron centrados)
  clase_ok  : el recorte de la caja principal lo clasifica bien --modelo
  final_ok  : clase final de un solo frame (caja principal, o el frame completo si no hay caja = respaldo)
Guarda hojas de contacto (miniaturas con la caja principal) en revision/comparar_detectores/<detector>/ para
revisar a ojo, y el detalle por foto en reportes/comparar_detectores.csv. Sin persistencia (una pasada por foto).
"""
import argparse
import csv

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from comun import CLASES, RAIZ, VAL_CAMARA, imagenes
from inferencia_deteccion import cargar_detector, clasificar, crear_parser, filtrar, recorte_cuadrado

HOJAS = RAIZ / "revision" / "comparar_detectores"
INFORME = RAIZ / "reportes" / "comparar_detectores.csv"


def hoja(miniaturas, ruta, cols=6):
    filas = [np.hstack(miniaturas[i:i + cols] + [np.zeros_like(miniaturas[0])] * (cols - len(miniaturas[i:i + cols])))
             for i in range(0, len(miniaturas), cols)]
    cv2.imwrite(str(ruta), np.vstack(filas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectores", nargs="+", default=["yoloe-26n-seg.pt", "yoloe-26s-seg.pt"])
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "etapa2_ext_full.pt"))
    args = ap.parse_args()

    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    clf = YOLO(args.modelo)
    nombres = clf.names
    fotos = [(c, p) for c in CLASES for p in imagenes(VAL_CAMARA / c)]
    dargs = crear_parser().parse_args([])  # filtros por defecto
    ignorar = [s.strip() for s in dargs.ignorar.split(",") if s.strip()]
    prompts = [s.strip() for s in dargs.prompts.split(",") if s.strip() and s.strip() not in ignorar]

    filas, resumen = [], []
    for nombre in args.detectores:
        det = cargar_detector(nombre, prompts + ignorar)
        dest = HOJAS / nombre.removesuffix(".pt")
        dest.mkdir(parents=True, exist_ok=True)
        for viejo in dest.glob("*.jpg"):  # solo salidas de este script
            viejo.unlink()
        cuenta = dict.fromkeys(("caja", "centro", "clase_ok", "final_ok"), 0)
        mini, n_hoja = [], 0
        for c, p in fotos:
            frame = cv2.imread(str(p))
            h, w = frame.shape[:2]
            r = det.predict(frame, conf=dargs.conf_det, imgsz=dargs.imgsz_det, device=device,
                            quantize=16 if gpu else None, agnostic_nms=False, max_det=50, verbose=False)[0]
            cajas, _, _ = filtrar(r, frame, dargs, ignorar)
            fila = {"detector": nombre, "foto": p.relative_to(VAL_CAMARA).as_posix(), "clase_real": c,
                    "caja": int(bool(cajas)), "centro": 0, "clase_caja": "", "clase_final": ""}
            if cajas:
                x1, y1, x2, y2 = cajas[0]
                fila["centro"] = int(x1 <= w / 2 <= x2 and y1 <= h / 2 <= y2)
                pr = clasificar(clf, [recorte_cuadrado(frame, cajas[0])], dargs.imgsz, device, gpu)[0]
                fila["clase_caja"] = fila["clase_final"] = nombres[int(pr.argmax())]
            else:
                pr = clasificar(clf, [frame], dargs.imgsz, device, gpu)[0]
                fila["clase_final"] = nombres[int(pr.argmax())]
            cuenta["caja"] += fila["caja"]
            cuenta["centro"] += fila["centro"]
            cuenta["clase_ok"] += fila["clase_caja"] == c
            cuenta["final_ok"] += fila["clase_final"] == c
            filas.append(fila)

            m = cv2.resize(frame, (240, 320)) if h > w else cv2.resize(frame, (320, 240))
            m = cv2.copyMakeBorder(m, 0, 320 - m.shape[0], 0, 320 - m.shape[1], cv2.BORDER_CONSTANT)
            if cajas:
                s = m.shape[1] / w if h <= w else 240 / w
                x1, y1, x2, y2 = (int(v * s) for v in cajas[0])
                cv2.rectangle(m, (x1, y1), (x2, y2), (0, 255, 0) if fila["centro"] else (0, 0, 255), 3)
            cv2.putText(m, f"{c[:8]} -> {fila['clase_caja'][:8] or 'SIN CAJA'}", (4, 314),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            mini.append(m)
            if len(mini) == 30:
                hoja(mini, dest / f"hoja_{n_hoja:02d}.jpg")
                mini, n_hoja = [], n_hoja + 1
        if mini:
            hoja(mini, dest / f"hoja_{n_hoja:02d}.jpg")
        resumen.append((nombre, cuenta))

    with open(INFORME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    n = len(fotos)
    print(f"\n{n} fotos de val_camara | clasificador {args.modelo}")
    print(f"{'detector':22s} {'caja':>14s} {'centro':>14s} {'clase_ok':>14s} {'final_ok':>14s}")
    for nombre, k in resumen:
        print(f"{nombre:22s} " + " ".join(f"{k[x]:4d} ({k[x] / n:6.1%})" for x in ("caja", "centro", "clase_ok", "final_ok")))
    print(f"\nDetalle: {INFORME.relative_to(RAIZ)} | hojas de contacto: {HOJAS.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
