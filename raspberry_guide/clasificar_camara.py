#!/usr/bin/env python3
"""
Clasificador de residuos en tiempo real - Raspberry Pi 4 + camara.
Proyecto: clasificador-residuos-juliaca (YOLO26s-cls, NTP 900.058:2019).

Mismo comportamiento que scripts/inferencia_webcam.py del proyecto:
recuadro central, imgsz 224, umbral 0.6, promedio de 8 frames y color del tacho.

Uso (normalmente a traves de INICIAR.sh):
    python clasificar_camara.py                 # detecta camara y modelo solos
    python clasificar_camara.py --rapido        # convierte a NCNN (mas FPS en la Pi)
    python clasificar_camara.py --camara 1      # forzar una camara concreta
    python clasificar_camara.py --sin-ventana   # sin pantalla (resultado en terminal)

Teclas (con ventana):  q / ESC = salir    s = guardar captura
"""
import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

VERSION_ULTRALYTICS = "8.4.184"   # version con la que se entreno el modelo
IMGSZ = 224                       # tamano con el que se entreno
BASE = Path(__file__).resolve().parent

# Clase -> (texto, color del tacho en BGR). Sin tildes: OpenCV solo dibuja ASCII.
TACHOS = {
    "papel_carton": ("TACHO AZUL - Papel y carton", (200, 90, 0)),
    "metales":      ("TACHO AMARILLO - Metales",    (0, 220, 255)),
    "plasticos":    ("TACHO BLANCO - Plasticos",    (255, 255, 255)),
    "vidrio":       ("TACHO GRIS - Vidrio",         (140, 140, 140)),
    "organico":     ("TACHO MARRON - Organicos",    (30, 75, 140)),
    "peligrosos":   ("TACHO ROJO - Peligrosos",     (0, 0, 220)),
    "generales":    ("TACHO NEGRO - Generales",     (25, 25, 25)),
    "sin_residuo":  ("Sin residuo",                 (80, 80, 80)),
}
MORADO = (160, 40, 150)   # NO SEGURO (morado porque el gris es el tacho de vidrio)


def verificar_entorno():
    from importlib.metadata import version, PackageNotFoundError
    try:
        v = version("ultralytics")
    except PackageNotFoundError:
        sys.exit("[ERROR] Falta ultralytics. Ejecuta:  bash INICIAR.sh")
    if v != VERSION_ULTRALYTICS:
        print(f"[AVISO] ultralytics {v} instalado; el modelo se entreno con {VERSION_ULTRALYTICS}.")
        print("        Si hay errores al cargar el modelo, ejecuta:  bash INICIAR.sh")
    else:
        print(f"[OK] ultralytics {v}")


def buscar_modelo(ruta):
    if ruta:
        p = Path(ruta)
        if not p.exists():
            sys.exit(f"[ERROR] No existe el modelo: {p}")
        return p
    pts = sorted((BASE / "modelo").glob("*.pt"))
    if not pts:
        sys.exit("[ERROR] No hay ningun .pt en la carpeta 'modelo/'.")
    if len(pts) > 1:
        print(f"[AVISO] Hay {len(pts)} modelos en 'modelo/'; se usa {pts[0].name}")
    return pts[0]


class CamaraUSB:
    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class CamaraPi:
    """Modulo de camara oficial de Raspberry Pi (cable plano), via picamera2."""
    def __init__(self, cam):
        self.cam = cam

    def read(self):
        frame = self.cam.capture_array()   # 'RGB888' de picamera2 ya viene en orden BGR
        return True, frame

    def release(self):
        self.cam.stop()


def abrir_camara(indice, ancho, alto):
    import cv2
    linux = sys.platform.startswith("linux")
    backend = cv2.CAP_V4L2 if linux else cv2.CAP_ANY
    indices = [indice] if indice is not None else range(0, 10)

    for i in indices:
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # menos retraso
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"[OK] Camara USB en indice {i} ({w}x{h})")
            return CamaraUSB(cap)
        cap.release()

    if indice is None:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_preview_configuration(
                main={"size": (ancho, alto), "format": "RGB888"}))
            cam.start()
            time.sleep(1)
            print("[OK] Modulo de camara Raspberry Pi (picamera2)")
            return CamaraPi(cam)
        except Exception:
            pass

    sys.exit("[ERROR] No se encontro ninguna camara. Conecta una webcam USB y "
             "revisa con:  v4l2-ctl --list-devices")


def cargar_modelo(ruta_pt, rapido):
    from ultralytics import YOLO
    if rapido:
        ncnn_dir = ruta_pt.with_name(ruta_pt.stem + "_ncnn_model")
        try:
            if not ncnn_dir.exists():
                print("[...] Convirtiendo a NCNN (solo la primera vez, ~1-3 min)...")
                YOLO(str(ruta_pt)).export(format="ncnn", imgsz=IMGSZ)
            modelo = YOLO(str(ncnn_dir), task="classify")
            print(f"[OK] Modelo NCNN: {ncnn_dir.name}")
            return modelo
        except Exception as e:
            print(f"[AVISO] No se pudo usar NCNN ({e}). Se usa el .pt original.")
    modelo = YOLO(str(ruta_pt))
    print(f"[OK] Modelo: {ruta_pt.name}")
    return modelo


def recuadro_central(frame, fraccion):
    """Cuadrado centrado de lado = fraccion x alto del frame (igual que la demo)."""
    h, w = frame.shape[:2]
    lado = max(16, min(int(h * fraccion), h, w))
    x0, y0 = (w - lado) // 2, (h - lado) // 2
    return x0, y0, lado


def texto_tacho(nombre, conf, umbral):
    if nombre == "sin_residuo":
        return TACHOS["sin_residuo"][0], TACHOS["sin_residuo"][1]
    if conf < umbral:
        return f"NO SEGURO ({conf:.0%})", MORADO
    etiqueta, color = TACHOS.get(nombre, (nombre.upper(), (0, 160, 0)))
    return f"{etiqueta} ({conf:.0%})", color


def color_texto(fondo):
    b, g, r = fondo
    return (0, 0, 0) if (0.114 * b + 0.587 * g + 0.299 * r) > 140 else (255, 255, 255)


def dibujar(frame, caja, texto, color, top3, fps):
    import cv2
    x0, y0, lado = caja
    cv2.rectangle(frame, (x0, y0), (x0 + lado, y0 + lado), color, 3)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), color, -1)
    cv2.putText(frame, texto, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color_texto(color), 2)
    y = 72
    for nombre, p in top3:
        cv2.putText(frame, f"{nombre}: {p:.0%}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
    cv2.putText(frame, f"{fps:.1f} FPS", (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Clasificador de residuos con camara")
    ap.add_argument("--modelo", help="ruta al .pt (por defecto: el que este en modelo/)")
    ap.add_argument("--camara", type=int, help="indice de camara (por defecto: autodetecta)")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--umbral", type=float, default=0.6,
                    help="confianza minima; por debajo muestra NO SEGURO (defecto 0.6)")
    ap.add_argument("--suavizado", type=int, default=8,
                    help="frames que se promedian (defecto 8)")
    ap.add_argument("--recuadro", type=float, default=0.75,
                    help="lado del recuadro como fraccion del alto (defecto 0.75; 1.0 = alto completo)")
    ap.add_argument("--rapido", action="store_true",
                    help="convierte a NCNN para mas velocidad en la Raspberry")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="no abre ventana; muestra resultados en la terminal")
    args = ap.parse_args()

    verificar_entorno()
    import cv2
    import numpy as np

    ruta_pt = buscar_modelo(args.modelo)
    modelo = cargar_modelo(ruta_pt, args.rapido)
    nombres = modelo.names
    print(f"[OK] Clases: {list(nombres.values())}")

    sin_pantalla = sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    sin_ventana = args.sin_ventana or sin_pantalla
    if sin_pantalla and not args.sin_ventana:
        print("[AVISO] No hay pantalla (SSH?). Resultados solo en la terminal.")

    cam = abrir_camara(args.camara, args.ancho, args.alto)
    print("Coloca el residuo dentro del recuadro. (q o ESC = salir, s = guardar captura)"
          if not sin_ventana else "Clasificando... (Ctrl+C para salir)")

    historial = deque(maxlen=max(1, args.suavizado))
    fps, t_prev, t_print = 0.0, time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[ERROR] La camara dejo de enviar imagen.")
                break

            x0, y0, lado = recuadro_central(frame, args.recuadro)
            recorte = frame[y0:y0 + lado, x0:x0 + lado]
            r = modelo.predict(recorte, imgsz=IMGSZ, verbose=False)[0]
            historial.append(r.probs.data.cpu().numpy())
            prom = np.mean(historial, axis=0)

            orden = np.argsort(prom)[::-1]
            top1, conf = int(orden[0]), float(prom[orden[0]])
            top3 = [(nombres[int(i)], float(prom[i])) for i in orden[:3]]
            texto, color = texto_tacho(nombres[top1], conf, args.umbral)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(ahora - t_prev, 1e-6))
            t_prev = ahora

            if sin_ventana:
                if ahora - t_print >= 1.0:
                    print(f"{texto:45s} | {fps:4.1f} FPS")
                    t_print = ahora
                continue

            dibujar(frame, (x0, y0, lado), texto, color, top3, fps)
            cv2.imshow("Clasificador de residuos - Juliaca", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                nombre = BASE / f"captura_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(nombre), frame)
                print(f"[OK] Captura guardada: {nombre.name}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        if not sin_ventana:
            cv2.destroyAllWindows()
        print("Fin.")


if __name__ == "__main__":
    main()
