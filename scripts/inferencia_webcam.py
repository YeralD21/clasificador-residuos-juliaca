"""Clasificación en vivo con la Logitech MX Brio (pensado para la laptop RTX 3050).

  python scripts/inferencia_webcam.py                                     # modelo actual: modelos/etapa2_j1028_s43.pt
  python scripts/inferencia_webcam.py --modelo modelos/etapa2_....pt --camara 1 --umbral 0.7
  python scripts/inferencia_webcam.py --recuadro 1.0   # cuadrado de alto completo

Pon el residuo dentro del recuadro: muestra el COLOR DEL TACHO al que va.
Si la confianza es menor al umbral muestra "NO SEGURO" en lugar de adivinar.
Teclas: Q salir, F pantalla completa, S guardar el recorte en capturas_revisar/ (para que TÚ lo clasifiques después).
"""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from comun import FRACCION_ROI, RAIZ, TACHO, abrir_camara, crear_ventana, roi_cuadrado, teclas_ventana


def revisar_versiones(modelo: Path):
    meta = modelo.with_suffix(".json")
    if not meta.exists():
        print("[AVISO] No hay .json de versiones junto al modelo.")
        return
    ent = json.loads(meta.read_text()).get("entorno", {})
    aqui = {"torch": torch.__version__, "ultralytics": ultralytics.__version__}
    for k, v in aqui.items():
        a, b = str(ent.get(k, "")).split("+")[0], v.split("+")[0]
        if a and a != b:
            print(f"[AVISO] {k}: entrenado con {a}, aquí {b}. Ejecuta pip install -r requirements-lock.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "etapa2_j1028_s43.pt"),
                    help="clasificador (modelo actual, ver EXPERIMENTOS.md); alternativa: modelos/etapa2_ext_full.pt")
    ap.add_argument("--camara", default="auto", help="índice de OpenCV, o auto (MX Brio si está, si no la integrada)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--umbral", type=float, default=0.6)
    ap.add_argument("--suavizado", type=int, default=8, help="frames que se promedian")
    ap.add_argument("--recuadro", type=float, default=FRACCION_ROI,
                    help="lado del recuadro como fracción del alto del frame (1.0 = alto completo)")
    args = ap.parse_args()

    ruta = Path(args.modelo)
    revisar_versiones(ruta)
    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    model = YOLO(str(ruta))
    nombres = model.names
    print(f"Modelo en {'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'} | clases: {list(nombres.values())}")

    cap = abrir_camara(args.camara)
    hist = deque(maxlen=args.suavizado)
    ventana = "Clasificador de residuos - NTP 900.058"
    crear_ventana(ventana)
    t0, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        recorte, (x1, y1, x2, y2) = roi_cuadrado(frame, args.recuadro)
        r = model.predict(recorte, imgsz=args.imgsz, device=device, half=gpu, verbose=False)[0]
        hist.append(r.probs.data.float().cpu().numpy())
        prom = np.mean(hist, axis=0)
        i = int(prom.argmax())
        clase, conf = nombres[i], float(prom[i])
        if conf < args.umbral:
            texto, color = f"NO SEGURO ({conf:.0%})", (0, 165, 255)
        else:
            texto, color = TACHO.get(clase, (clase, (0, 255, 0)))
            texto = f"{texto} ({conf:.0%})"

        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - t0, 1e-6)
        t0 = ahora
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
        cv2.putText(frame, texto, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        for k, j in enumerate(np.argsort(prom)[::-1][:3]):
            cv2.putText(frame, f"{nombres[int(j)]}: {prom[j]:.2f}", (12, 80 + 26 * k),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"{fps:.0f} FPS", (frame.shape[1] - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(ventana, frame)

        k = teclas_ventana(ventana)
        if k == "q":
            break
        if k == "s":
            d = RAIZ / "capturas_revisar"
            d.mkdir(exist_ok=True)
            f = d / f"pred-{clase}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(f), recorte)
            print(f"Guardado {f}. Clasifícalo TÚ y muévelo a dataset/juliaca/<split>/<clase>/objNNN/")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
