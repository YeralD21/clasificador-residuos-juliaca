#!/usr/bin/env python3
"""
Clasificador de residuos en tiempo real - Raspberry Pi 4 + camara.
Proyecto: clasificador-residuos-juliaca (YOLO26s-cls).

Uso:
    python clasificar_camara.py              # detecta camara y modelo solos
    python clasificar_camara.py --rapido     # convierte a NCNN (mas FPS en la Pi)
    python clasificar_camara.py --camara 1   # forzar una camara concreta
    python clasificar_camara.py --sin-ventana  # sin pantalla (resultado en terminal)

Teclas (con ventana):  q / ESC = salir    s = guardar captura
"""
import argparse
import os
import sys
import time
from pathlib import Path

VERSION_ULTRALYTICS = "8.3.209"   # version con la que se entreno el modelo
BASE = Path(__file__).resolve().parent


def verificar_entorno():
    from importlib.metadata import version, PackageNotFoundError
    try:
        v = version("ultralytics")
    except PackageNotFoundError:
        sys.exit("[ERROR] Falta ultralytics. Ejecuta primero:  bash INICIAR.sh")
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
                YOLO(str(ruta_pt)).export(format="ncnn")
            modelo = YOLO(str(ncnn_dir), task="classify")
            print(f"[OK] Modelo NCNN: {ncnn_dir.name}")
            return modelo
        except Exception as e:
            print(f"[AVISO] No se pudo usar NCNN ({e}). Se usa el .pt original.")
    modelo = YOLO(str(ruta_pt))
    print(f"[OK] Modelo: {ruta_pt.name}")
    return modelo


def dibujar(frame, texto, top3, fps, color):
    import cv2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (0, 0, 0), -1)
    cv2.putText(frame, texto, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    y = 70
    for nombre, p in top3:
        cv2.putText(frame, f"{nombre}: {p:.0%}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24
    cv2.putText(frame, f"{fps:.1f} FPS", (frame.shape[1] - 120, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="Clasificador de residuos con camara")
    ap.add_argument("--modelo", help="ruta al .pt (por defecto: el que este en modelo/)")
    ap.add_argument("--camara", type=int, help="indice de camara (por defecto: autodetecta)")
    ap.add_argument("--ancho", type=int, default=640)
    ap.add_argument("--alto", type=int, default=480)
    ap.add_argument("--umbral", type=float, default=0.50,
                    help="confianza minima para mostrar la clase (0-1)")
    ap.add_argument("--rapido", action="store_true",
                    help="convierte a NCNN para mas velocidad en la Raspberry")
    ap.add_argument("--sin-ventana", action="store_true",
                    help="no abre ventana; muestra resultados en la terminal")
    args = ap.parse_args()

    verificar_entorno()
    import cv2

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
    print("Clasificando... (q o ESC para salir, s para guardar captura)"
          if not sin_ventana else "Clasificando... (Ctrl+C para salir)")

    fps, t_prev, t_print = 0.0, time.time(), 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                print("[ERROR] La camara dejo de enviar imagen.")
                break

            r = modelo.predict(frame, verbose=False)[0]
            probs = r.probs
            top1, conf = int(probs.top1), float(probs.top1conf)
            top3 = [(nombres[int(i)], float(probs.data[int(i)])) for i in probs.top5[:3]]

            ahora = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(ahora - t_prev, 1e-6))
            t_prev = ahora

            if conf >= args.umbral:
                texto, color = f"{nombres[top1]} ({conf:.0%})", (0, 255, 0)
            else:
                texto, color = "Incierto - acerca el objeto", (0, 200, 255)

            if sin_ventana:
                if ahora - t_print >= 1.0:
                    print(f"{texto:40s} | {fps:4.1f} FPS")
                    t_print = ahora
                continue

            dibujar(frame, texto, top3, fps, color)
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
