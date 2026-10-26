"""Diagnóstico: ¿inferencia_deteccion.py clasifica igual que evaluar.py? No entrena ni modifica el dataset.

  python scripts/diagnostico_inferencia.py                       # modelos/etapa2_ext_full.pt en dataset/val_camara
  python scripts/diagnostico_inferencia.py --modelo modelos/etapa1_ganador.pt --recortes 0

Pasa cada foto de val_camara por:
  1. evaluar.py: model.predict(ruta)  (referencia, debe dar lo mismo que reportes/<modelo>_val_camara.json)
  2. inferencia_deteccion.py: cv2.imread -> la imagen completa como si fuera UNA caja -> recorte_cuadrado
     -> clasificar() (la misma función que usa en vivo, con fp16 en GPU)
  3. clasificar() con el frame entero, sin recorte (aísla el efecto del recorte del de fp16 / lectura)
y compara top-1, F1 macro y cuántas fotos cambian de predicción. También comprueba que cv2.imread lea
los mismos píxeles que el lector de Ultralytics. Con --recortes N guarda N recortes REALES del detector
(con los filtros por defecto) en revision/recortes_debug/ para mirarlos.
"""
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.patches import imread as imread_ultralytics

from comun import CLASES, RAIZ, VAL_CAMARA, imagenes
from evaluar import metricas, predecir_carpeta
from inferencia_deteccion import IGNORAR, cargar_detector, clasificar, crear_parser, filtrar, recorte_cuadrado

SALIDA = RAIZ / "revision" / "recortes_debug"


def resumen(nombre, preds, reales, nombres):
    idx = {v: k for k, v in nombres.items()}
    m = np.zeros((len(nombres), len(nombres)), dtype=int)
    for p, r in zip(preds, reales):
        m[idx[p], idx[r]] += 1
    _, f1s = metricas(m, nombres)
    top1, f1 = np.trace(m) / len(preds), float(np.mean(f1s))
    print(f"{nombre:52s} top1={top1:.4f}  F1 macro={f1:.4f}")
    return top1, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "etapa2_ext_full.pt"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--recortes", type=int, default=10, help="recortes reales del detector a guardar (0 = ninguno)")
    args = ap.parse_args()

    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    clf = YOLO(args.modelo)
    nombres = clf.names
    fotos = [(c, p) for c in CLASES for p in imagenes(VAL_CAMARA / c)]
    reales = [c for c, _ in fotos]
    print(f"{len(fotos)} fotos de val_camara | {args.modelo} | {'GPU' if gpu else 'CPU'}\n")

    # 1. referencia: exactamente lo que hace evaluar.py
    _, por_foto, _, _ = predecir_carpeta(clf, VAL_CAMARA, args.imgsz, 32)
    ref = [f["prediccion"] for f in por_foto]

    # 2 y 3. lo que hace inferencia_deteccion.py
    distintos_px, completa, entera = 0, [], []
    for i in range(0, len(fotos), 32):
        frames = []
        for _, p in fotos[i:i + 32]:
            f = cv2.imread(str(p))  # así lee --fuente
            g = imread_ultralytics(str(p))  # así lee evaluar.py (vía model.predict(ruta))
            distintos_px += f.shape != g.shape or not np.array_equal(f, g)
            frames.append(f)
        recs = [recorte_cuadrado(f, (0, 0, f.shape[1], f.shape[0])) for f in frames]
        completa += [nombres[int(pr.argmax())] for pr in clasificar(clf, recs, args.imgsz, device, gpu)]
        entera += [nombres[int(pr.argmax())] for pr in clasificar(clf, frames, args.imgsz, device, gpu)]

    print(f"cv2.imread vs lector de Ultralytics: {distintos_px} fotos con píxeles distintos\n")
    resumen("1. evaluar.py (ruta)", ref, reales, nombres)
    resumen("2. deteccion: imagen completa como caja (recorte)", completa, reales, nombres)
    resumen("3. deteccion: clasificar(frame entero)", entera, reales, nombres)
    print(f"\npredicciones distintas de evaluar.py: (2) {sum(a != b for a, b in zip(ref, completa))}"
          f"  (3) {sum(a != b for a, b in zip(ref, entera))} de {len(fotos)}")

    if args.recortes:
        dargs = crear_parser().parse_args([])  # filtros por defecto de inferencia_deteccion.py
        ignorar = [s.strip() for s in IGNORAR.split(",") if s.strip()]
        prompts = [s.strip() for s in dargs.prompts.split(",") if s.strip() and s.strip() not in ignorar]
        det = cargar_detector(dargs.detector, prompts + ignorar)
        SALIDA.mkdir(parents=True, exist_ok=True)
        for viejo in SALIDA.glob("*.jpg"):  # solo salidas de este script
            viejo.unlink()
        guardados, paso = 0, max(1, len(fotos) // (3 * args.recortes))
        for c, p in fotos[::paso]:  # repartidas entre clases
            frame = cv2.imread(str(p))
            r = det.predict(frame, conf=dargs.conf_det, imgsz=dargs.imgsz_det, device=device,
                            quantize=16 if gpu else None, agnostic_nms=False, max_det=50, verbose=False)[0]
            cajas, _, _ = filtrar(r, frame, dargs, ignorar)
            if not cajas:
                continue
            rec = recorte_cuadrado(frame, cajas[0])
            pr = clasificar(clf, [rec], args.imgsz, device, gpu)[0]
            k = int(pr.argmax())
            f = SALIDA / f"{guardados:02d}_real-{c}_pred-{nombres[k]}-{pr[k]:.0%}_{p.parent.name}.jpg"
            cv2.imwrite(str(f), rec)
            guardados += 1
            if guardados >= args.recortes:
                break
        print(f"\n{guardados} recortes reales del detector guardados en {SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
