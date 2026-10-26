"""Configuración compartida del proyecto (clases, colores de tacho, rutas, cámara)."""
import hashlib
import platform
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATASET = RAIZ / "dataset"
CRUDO = DATASET / "crudo"
PUBLICO = DATASET / "publico"
JULIACA = DATASET / "juliaca"
ETAPA2 = DATASET / "_etapa2"          # generado automáticamente, no editar
ETAPA1 = DATASET / "_etapa1"          # generado con --externo (publico + externo), no editar
EXTERNO = DATASET / "externo"         # datos externos reales: SOLO train (regla 1 de CLAUDE.md)
VAL_CAMARA = DATASET / "val_camara"   # evaluación aparte con la cámara de la laptop: <clase>/<objeto>/, nunca se entrena
SPLITS = ("train", "val", "test")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# El modelo clasifica el MATERIAL. El color es el TACHO donde va (NTP 900.058).
# Orden alfabético = orden de índices que usa Ultralytics.
CLASES = [
    "generales", "metales", "organico", "papel_carton",
    "peligrosos", "plasticos", "sin_residuo", "vidrio",
]

TACHO = {  # clase: (texto, color BGR para OpenCV)
    "papel_carton": ("AZUL - Papel y carton", (200, 90, 0)),
    "metales":      ("AMARILLO - Metales", (0, 220, 255)),
    "plasticos":    ("BLANCO - Plasticos", (255, 255, 255)),
    "vidrio":       ("GRIS - Vidrio", (150, 150, 150)),
    "organico":     ("MARRON - Organicos", (30, 75, 120)),
    "peligrosos":   ("ROJO - Peligrosos", (0, 0, 230)),
    "generales":    ("NEGRO - No aprovechables", (40, 40, 40)),
    "sin_residuo":  ("Coloca un residuo en el recuadro", (120, 120, 120)),
}

FRACCION_ROI = 0.75  # el recuadro de captura/inferencia ocupa 75% del alto (inferencia_webcam: --recuadro)


def imagenes(carpeta: Path):
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def objeto_de(ruta: Path, clase_dir: Path):
    """Nombre del objeto = subcarpeta directa dentro de la clase.
    dataset/juliaca/train/plasticos/obj012_botella_inca_kola/foto1.jpg -> 'obj012_botella_inca_kola'
    Si la foto está suelta en la carpeta de la clase devuelve None."""
    rel = ruta.relative_to(clase_dir)
    return rel.parts[0] if len(rel.parts) > 1 else None


def md5(ruta: Path) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def dhash(ruta: Path, tam: int = 8) -> int:
    """Hash perceptual: fotos casi iguales dan hashes con pocos bits distintos."""
    from PIL import Image
    with Image.open(ruta) as im:
        g = im.convert("L").resize((tam + 1, tam), Image.BILINEAR)
        px = list(g.getdata())
    bits = 0
    for fila in range(tam):
        for col in range(tam):
            i = fila * (tam + 1) + col
            bits = (bits << 1) | (px[i] > px[i + 1])
    return bits


def nombres_camaras():
    """Nombres de las cámaras conectadas según Windows (WMI), o None si no se pueden consultar. Dicen SI está la
    MX Brio, pero no su índice en OpenCV (el orden de WMI no es el de DirectShow)."""
    if platform.system() != "Windows":
        return None
    import subprocess
    cmd = ("Get-CimInstance Win32_PnPEntity | Where-Object { $_.PNPClass -in 'Camera','Image' -and "
           "$_.Status -eq 'OK' } | ForEach-Object { $_.Name }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return [n.strip() for n in r.stdout.splitlines() if n.strip()] if r.returncode == 0 else None


def probar_camara(indice):
    """(ancho, alto) máximos que acepta la cámara `indice` (pide 4K y lee lo que queda), o None si no da frames."""
    import cv2
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        ok, frame = cap.read()
        return (frame.shape[1], frame.shape[0]) if ok and frame is not None else None
    finally:
        cap.release()


def elegir_camara(camara="auto"):
    """--camara: un número se usa tal cual. 'auto' prueba los índices 0-5: si Windows ve una MX Brio, elige la
    cámara que acepta mayor resolución (la Brio llega a 4K); si no hay Brio, la primera que funciona (la integrada).
    Imprime qué eligió."""
    if str(camara).lower() != "auto":
        return int(camara)
    import cv2
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # sin avisos por índices vacíos
    except AttributeError:
        pass
    nombres = nombres_camaras()
    brio = [n for n in nombres or [] if "brio" in n.lower()]
    print(f"[camara auto] Windows ve: {nombres if nombres is not None else '(no se pudo consultar)'}", flush=True)
    halladas = {i: r for i in range(6) if (r := probar_camara(i))}
    if not halladas:
        raise SystemExit("[camara auto] No se encontró ninguna cámara en los índices 0-5. Cierra Logi Tune / OBS / "
                         "Teams o usa --camara N.")
    print("[camara auto] índices que dan imagen: " + ", ".join(f"{i} ({w}x{h} máx)" for i, (w, h) in halladas.items()))
    if brio or nombres is None:
        i = max(halladas, key=lambda k: halladas[k][0] * halladas[k][1])
        motivo = f"MX Brio conectada ({brio[0]}): la de mayor resolución" if brio else "la de mayor resolución"
    else:
        i = min(halladas)
        motivo = "sin MX Brio: la integrada (primer índice que funciona)"
    print(f"[camara auto] Elegida: índice {i}, {halladas[i][0]}x{halladas[i][1]} máx -> {motivo}. "
          "Para forzar otra: --camara N", flush=True)
    return i


def abrir_camara(indice="auto", ancho=1280, alto=720, fps=30):
    """Abre la Logitech MX Brio (o cualquier webcam). En Windows usa DirectShow + MJPG. indice: número o 'auto'
    (ver elegir_camara)."""
    import cv2
    indice = elegir_camara(indice)
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {indice}. Prueba --camara 1 o cierra Logi Tune / OBS / Teams.")
    return cap


def roi_cuadrado(frame, fraccion=FRACCION_ROI):
    """Cuadrado centrado de lado = fraccion * lado menor del frame. Con 1.0 es el cuadrado de alto completo,
    que es justo lo que ve el clasificador si le pasas el frame entero (Ultralytics hace Resize + CenterCrop)."""
    h, w = frame.shape[:2]
    lado = int(min(h, w) * min(max(fraccion, 0.05), 1.0))
    x1, y1 = (w - lado) // 2, (h - lado) // 2
    return frame[y1:y1 + lado, x1:x1 + lado].copy(), (x1, y1, x1 + lado, y1 + lado)


def crear_ventana(nombre, ancho=1280, alto=720):
    """Ventana redimensionable. cv2.imshow sin namedWindow crea una WINDOW_AUTOSIZE: al maximizar, el frame
    se queda en su tamaño nativo y el resto de la ventana sale gris. Con WINDOW_NORMAL | WINDOW_FREERATIO
    el video se escala a toda el área de la ventana (maximizada o en pantalla completa)."""
    import cv2
    cv2.namedWindow(nombre, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.resizeWindow(nombre, ancho, alto)


def teclas_ventana(nombre, espera=1):
    """waitKey + tecla F (alterna pantalla completa). Devuelve la tecla en minúscula, o 'q' si cerraron la ventana."""
    import cv2
    k = cv2.waitKey(espera) & 0xFF
    if cv2.getWindowProperty(nombre, cv2.WND_PROP_VISIBLE) < 1:
        return "q"
    c = chr(k).lower() if k != 255 else ""
    if c == "f":
        completa = cv2.getWindowProperty(nombre, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
        cv2.setWindowProperty(nombre, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_NORMAL if completa else cv2.WINDOW_FULLSCREEN)
    return c
