"""Detección + clasificación en dos etapas (sin recuadro fijo), pensado para la laptop RTX 3050.

  python scripts/inferencia_deteccion.py --camara 1
  python scripts/inferencia_deteccion.py --camara 1 --detector yoloe-26s-seg.pt --umbral 0.7
  python scripts/inferencia_deteccion.py --fuente video.mp4            # probar sin webcam
  python scripts/inferencia_deteccion.py --fuente foto.jpg --bench 100 # medir FPS sin ventana

Etapa 1: un detector de vocabulario abierto (YOLOE o YOLO-World) busca objetos en TODA la imagen
usando prompts de texto (--prompts). Etapa 2: cada caja se recorta (cuadrada, +10 % de margen) y la
clasifica TU modelo (--modelo). La caja se dibuja con el color del tacho (NTP 900.058); si la
confianza es menor a --umbral se dibuja gris con "NO SEGURO". Los recortes que el modelo ve como
sin_residuo se ignoran. El detector solo localiza: la clase la decide siempre el clasificador.
Teclas: Q salir, S guardar cada recorte del frame en capturas_revisar/ (para que TÚ lo clasifiques).
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.utils.downloads import attempt_download_asset

from comun import RAIZ, TACHO, abrir_camara
from inferencia_webcam import revisar_versiones

PROMPTS = ("bottle,can,plastic bag,snack wrapper,battery,phone,charger,cable,cup,glass jar,"
           "box,paper,fruit peel,food waste,tube,container")
DETECTORES = RAIZ / "modelos" / "detectores"   # pesos descargados de Ultralytics (no se suben, *.pt)
GRIS = (150, 150, 150)


def cargar_detector(nombre: str, prompts: list[str]):
    ruta = Path(nombre)
    if not ruta.exists():
        DETECTORES.mkdir(parents=True, exist_ok=True)
        ruta = Path(attempt_download_asset(DETECTORES / ruta.name))
    if "world" in ruta.name:
        det = YOLOWorld(str(ruta))
        det.set_classes(prompts)
    else:
        det = YOLOE(str(ruta))
        if "-pf" in ruta.name:
            print("[AVISO] Detector prompt-free: se ignoran --prompts (usa su vocabulario interno).")
        else:
            det.set_classes(prompts, det.get_text_pe(prompts))
    return det


def recorte_cuadrado(frame, caja, margen=0.10):
    """Cuadrado centrado en la caja, lado = lado mayor * (1 + 2*margen). Rellena si sale del frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caja
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = max(x2 - x1, y2 - y1) * (1 + 2 * margen)
    a, b = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    c, d = a + int(round(lado)), b + int(round(lado))
    rec = frame[max(b, 0):min(d, h), max(a, 0):min(c, w)]
    return cv2.copyMakeBorder(rec, max(-b, 0), max(d - h, 0), max(-a, 0), max(c - w, 0),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def etiqueta(frame, caja, texto, color):
    x1, y1, x2, y2 = map(int, caja)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y = max(y1, th + 10)
    cv2.rectangle(frame, (x1, y - th - 10), (x1 + tw + 10, y), color, -1)
    tinta = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)  # texto legible sobre blanco/amarillo
    cv2.putText(frame, texto, (x1 + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tinta, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "etapa2_avance.pt"))
    ap.add_argument("--detector", default="yoloe-26n-seg.pt",
                    help="yoloe-26{n,s,m,l,x}-seg.pt (texto), yoloe-26?-seg-pf.pt (prompt-free), yolov8s-worldv2.pt")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--fuente", help="video o imagen en lugar de la webcam (para pruebas)")
    ap.add_argument("--umbral", type=float, default=0.6)
    ap.add_argument("--conf-det", type=float, default=0.15, help="confianza mínima del detector")
    ap.add_argument("--max-objetos", type=int, default=5)
    ap.add_argument("--min-area", type=float, default=0.005, help="área mínima de la caja, fracción del frame")
    ap.add_argument("--prompts", default=PROMPTS, help="separados por coma")
    ap.add_argument("--imgsz", type=int, default=224, help="tamaño del clasificador (con el que se entrenó)")
    ap.add_argument("--imgsz-det", type=int, default=640, help="tamaño del detector (menor = más rápido)")
    ap.add_argument("--bench", type=int, default=0, help="N frames sin ventana y muestra FPS por etapa")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    ruta = Path(args.modelo)
    revisar_versiones(ruta)
    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    clf = YOLO(str(ruta))
    nombres = clf.names
    det = cargar_detector(args.detector, prompts)
    print(f"{'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'} | detector {args.detector} | "
          f"prompts: {prompts}\nclases: {list(nombres.values())}")

    imagen = None
    if args.fuente:
        imagen = cv2.imread(args.fuente) if Path(args.fuente).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else None
        cap = None if imagen is not None else cv2.VideoCapture(args.fuente)
    else:
        cap = abrir_camara(args.camara)

    t0, fps, n = time.time(), 0.0, 0
    tiempos = {"det": [], "clf": [], "total": []}
    while True:
        if imagen is not None:
            frame = imagen.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                if args.fuente:
                    break  # fin del video
                continue
        ta = time.perf_counter()
        r = det.predict(frame, conf=args.conf_det, imgsz=args.imgsz_det, device=device, quantize=16 if gpu else None,
                        agnostic_nms=True, max_det=50, verbose=False)[0]
        tb = time.perf_counter()

        area = frame.shape[0] * frame.shape[1]
        cajas = [c for c in r.boxes.xyxy.cpu().numpy()
                 if (c[2] - c[0]) * (c[3] - c[1]) >= args.min_area * area][:args.max_objetos]  # ya vienen por conf
        recortes = [recorte_cuadrado(frame, c) for c in cajas]
        dibujos = []
        if recortes:
            for caja, rec, rc in zip(cajas, recortes, clf.predict(recortes, imgsz=args.imgsz, device=device,
                                                                  quantize=16 if gpu else None, verbose=False)):
                i, conf = int(rc.probs.top1), float(rc.probs.top1conf)
                clase = nombres[i]
                if clase == "sin_residuo":
                    continue
                if conf < args.umbral:
                    dibujos.append((caja, rec, f"NO SEGURO ({conf:.0%})", GRIS, "NOSEGURO"))
                else:
                    dibujos.append((caja, rec, f"{clase} ({conf:.0%})", TACHO[clase][1], clase))
        tc = time.perf_counter()
        tiempos["det"].append(tb - ta)
        tiempos["clf"].append(tc - tb)
        tiempos["total"].append(tc - ta)
        n += 1

        if args.bench:
            if n >= args.bench:
                break
            continue

        for caja, _, texto, color, _ in dibujos:
            etiqueta(frame, caja, texto, color)
        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - t0, 1e-6)
        t0 = ahora
        cv2.putText(frame, f"{fps:.0f} FPS | {len(dibujos)} obj", (12, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Deteccion + clasificacion - NTP 900.058", frame)

        k = cv2.waitKey(0 if imagen is not None else 1) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        if k in (ord("s"), ord("S")):
            d = RAIZ / "capturas_revisar"
            d.mkdir(exist_ok=True)
            marca = time.strftime("%Y%m%d_%H%M%S")
            for j, (_, rec, _, _, clase) in enumerate(dibujos):
                f = d / f"pred-{clase}_{marca}_{j}.jpg"
                cv2.imwrite(str(f), rec)
                print(f"Guardado {f}")
            if dibujos:
                print("Clasifícalos TÚ y muévelos a dataset/juliaca/<split>/<clase>/objNNN/")

    if args.bench and n:
        w = max(1, n // 10)  # descarta el calentamiento inicial
        for k, v in tiempos.items():
            ms = 1000 * float(np.mean(v[w:]))
            print(f"{k:6s} {ms:7.1f} ms  ({1000 / ms:5.1f} FPS)")
        print(f"frames medidos: {n - w} (descartados {w} de calentamiento) | objetos último frame: {len(dibujos)}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
