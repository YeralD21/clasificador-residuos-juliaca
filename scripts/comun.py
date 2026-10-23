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

FRACCION_ROI = 0.75  # el recuadro de captura/inferencia ocupa 75% del alto


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


def abrir_camara(indice=0, ancho=1280, alto=720, fps=30):
    """Abre la Logitech MX Brio (o cualquier webcam). En Windows usa DirectShow + MJPG."""
    import cv2
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(indice, backend)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {indice}. Prueba --camara 1 o cierra Logi Tune / OBS / Teams.")
    return cap


def roi_cuadrado(frame):
    h, w = frame.shape[:2]
    lado = int(min(h, w) * FRACCION_ROI)
    x1, y1 = (w - lado) // 2, (h - lado) // 2
    return frame[y1:y1 + lado, x1:x1 + lado].copy(), (x1, y1, x1 + lado, y1 + lado)
