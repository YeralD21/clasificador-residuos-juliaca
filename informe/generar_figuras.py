"""Figuras del informe final (sección 6 y resultados).

Se corre con el .venv del proyecto (ultralytics, opencv, pillow, matplotlib):
    .venv\\Scripts\\python.exe informe\\generar_figuras.py

Solo LEE dataset/, modelos/ y reportes/; escribe en informe/figuras/ e informe/datos/.
- Predicciones: modelo final etapa2_nf_s43 SOLO sobre validación (val_camara y juliaca/val).
  dataset/juliaca/test no se toca (reglas 5 y 9 de CLAUDE.md).
- Privacidad: toda foto candidata pasa por YOLOE (prompts 'human face' y 'head') y por el detector de
  rostros YuNet de OpenCV. Se prefieren fotos sin rostros; si una clase no tiene suficientes, se usan
  fotos con el rostro pixelado y desenfocado. Además, cada mosaico se revisa a ojo.
"""
from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DS = RAIZ / "dataset"
OUT = RAIZ / "informe" / "figuras"
DAT = RAIZ / "informe" / "datos"
OUT.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)
MODELO = RAIZ / "modelos" / "etapa2_nf_s43.pt"
DETECTOR = RAIZ / "modelos" / "detectores" / "yoloe-26s-seg.pt"
DATA_NEW = RAIZ.parent / "DATA-NEW"

CLASES = ["generales", "metales", "organico", "papel_carton", "peligrosos", "plasticos", "sin_residuo", "vidrio"]
TACHO = {"generales": ("Negro", (30, 30, 30)), "metales": ("Amarillo", (240, 200, 0)),
         "organico": ("Marrón", (120, 72, 36)), "papel_carton": ("Azul", (30, 90, 200)),
         "peligrosos": ("Rojo", (210, 40, 40)), "plasticos": ("Blanco", (250, 250, 250)),
         "sin_residuo": ("—", (160, 160, 160)), "vidrio": ("Gris", (128, 128, 128))}
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OK, MAL = (12, 163, 12), (208, 59, 59)
TINTA, TINTA2 = (11, 11, 11), (82, 81, 78)
FONDO = (252, 252, 251)

FUENTE = "C:/Windows/Fonts/arial.ttf"
FUENTE_B = "C:/Windows/Fonts/arialbd.ttf"


def font(n, bold=False):
    return ImageFont.truetype(FUENTE_B if bold else FUENTE, n)


def imagenes(c: Path):
    return sorted(p for p in c.rglob("*") if p.is_file() and p.suffix.lower() in EXT)


def abrir(src) -> Image.Image:
    if isinstance(src, Image.Image):
        return src
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    return im.convert("RGB")


def cuadrado(im: Image.Image, lado: int) -> Image.Image:
    """Recorte cuadrado central (lo mismo que ve el clasificador: resize lado corto + center crop)."""
    return ImageOps.fit(im, (lado, lado), Image.BILINEAR)


# ---------------------------------------------------------------- privacidad
class Privacidad:
    """Detecta rostros (YuNet de OpenCV + YOLOE con prompts 'human face' y 'head').
    Las manos o brazos solos no cuentan. Cache en informe/datos/privacidad.csv."""

    def __init__(self):
        self.cache_f = DAT / "privacidad.csv"
        self.cache = {}
        if self.cache_f.exists():
            for r in csv.DictReader(open(self.cache_f, encoding="utf-8")):
                self.cache[r["clave"]] = r
        self.det = None
        self._yunet = None

    @property
    def yunet(self):
        # YuNet (opencv_zoo, face_detection_yunet_2023mar.onnx); en OpenCV 5 ya no existe CascadeClassifier.
        # Se carga al primer uso, como YOLOE: --solo-app no lo necesita.
        if self._yunet is None:
            self._yunet = cv2.FaceDetectorYN.create(
                str(RAIZ / "modelos/detectores/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.5, 0.3, 5000)
        return self._yunet

    def _detector(self):
        if self.det is None:
            from ultralytics import YOLOE
            self.det = YOLOE(str(DETECTOR))
            pr = ["human face", "head", "person"]
            self.det.set_classes(pr, self.det.get_text_pe(pr))
        return self.det

    def caras(self, clave: str, im: Image.Image):
        if clave not in self.cache:
            arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
            cajas, n_persona = [], 0
            r = self._detector().predict(arr, conf=0.25, imgsz=640, verbose=False)[0]
            for x, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if int(c) == 2:
                    n_persona += 1          # 'person' (puede ser solo una mano): no se toma como rostro
                else:
                    cajas.append([int(v) for v in x])
            esc = min(1.0, 960 / max(arr.shape[:2]))
            g = cv2.resize(arr, None, fx=esc, fy=esc) if esc < 1 else arr
            self.yunet.setInputSize((g.shape[1], g.shape[0]))
            _, f = self.yunet.detect(g)
            for x in ([] if f is None else f):
                x0, y0, w, h = (x[:4] / esc).tolist()
                cajas.append([int(x0), int(y0), int(x0 + w), int(y0 + h)])
            self.cache[clave] = {"clave": clave, "n_rostros": len(cajas), "n_persona": n_persona,
                                 "cajas": json.dumps(cajas)}
        return json.loads(self.cache[clave]["cajas"])

    def con_persona(self, clave):
        return int(self.cache[clave]["n_persona"]) > 0

    def guardar(self):
        with open(self.cache_f, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["clave", "n_rostros", "n_persona", "cajas"])
            w.writeheader()
            w.writerows(self.cache.values())


PRIV = Privacidad()


_DET_SEG = {}


def _seg(prompts):
    k = tuple(prompts)
    if k not in _DET_SEG:
        from ultralytics import YOLOE
        d = YOLOE(str(DETECTOR))
        d.set_classes(list(prompts), d.get_text_pe(list(prompts)))
        _DET_SEG[k] = d
    return _DET_SEG[k]


def _mascara(im: Image.Image, prompts, conf):
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    r = _seg(prompts).predict(arr, conf=conf, imgsz=640, retina_masks=True, verbose=False)[0]
    m = np.zeros(arr.shape[:2], bool)
    if r.masks is not None:
        for mk in r.masks.data.cpu().numpy():
            m |= cv2.resize(mk.astype(np.uint8), (arr.shape[1], arr.shape[0])) > 0
    return m


# Fotos que, tras difuminar, aún dejaban ver parte de un rostro en la revisión a ojo: no se usan.
EXCLUIR_MANUAL = {
    "dataset/juliaca/train/organico/obj105_20261028-obj063/20261028_obj063_02.jpg",
}

PROMPTS_PERSONA = ("person", "human face", "head", "hair", "human hand", "arm")
PROMPTS_OBJETO = ("sheet of paper", "paper", "napkin", "tissue", "paper airplane", "notebook", "package",
                  "bottle", "can", "box", "bag", "food", "fruit", "jar", "cup", "wrapper", "sachet")


def difuminar(im: Image.Image, cajas, persona_entera=False):
    """Pixela y desenfoca: cada caja de rostro (ampliada 35 %) y, con persona_entera, toda la máscara de
    persona (YOLOE-seg) menos la del objeto, para que solo quede nítido el residuo."""
    from PIL import ImageFilter
    W, H = im.size
    m = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in cajas:
        mx, my = (x1 - x0) * 0.6, (y1 - y0) * 0.6
        m[max(0, int(y0 - my)):min(H, int(y1 + my)), max(0, int(x0 - mx)):min(W, int(x1 + mx))] = True
    if persona_entera:
        pm = _mascara(im, PROMPTS_PERSONA, 0.2)
        om = _mascara(im, PROMPTS_OBJETO, 0.25)
        pm = cv2.dilate(pm.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
        m |= pm & ~om
    if not m.any():
        return im
    peq = im.resize((max(1, W // 40), max(1, H // 40)), Image.BILINEAR).resize((W, H), Image.NEAREST)
    borr = peq.filter(ImageFilter.GaussianBlur(max(6, W // 60)))
    alfa = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
    return Image.composite(borr, im, alfa)


PROMPTS_RECORTE = {
    "generales": ("napkin", "tissue", "paper towel", "crumpled paper"),
    "papel_carton": ("sheet of paper", "paper", "paper airplane", "notebook", "document", "envelope"),
}
FONDO_RECORTE = (236, 236, 233)


def _objeto(im: Image.Image, prompts):
    """Caja (x0, y0, x1, y1) y máscara del objeto más confiable según YOLOE-seg con los prompts de su clase."""
    arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    r = _seg(prompts).predict(arr, conf=0.15, imgsz=640, retina_masks=True, verbose=False)[0]
    W, H = im.size
    mejor = None
    for k, (b, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())):
        area = (b[2] - b[0]) * (b[3] - b[1]) / (W * H)
        if 0.02 <= area <= 0.95 and (mejor is None or c > mejor[1]):
            mejor = (b, float(c), k)
    if mejor is None:
        return None, None
    # máscara = unión de todas las detecciones de papel (una sola a veces deja partes del objeto fuera)
    mk = np.zeros((H, W), bool)
    for k in range(len(r.boxes)):
        m1 = cv2.resize(r.masks.data[k].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        mk |= m1 > 0
    return [int(v) for v in mejor[0]], mk


_CABEZAS = {}


def _cabezas(clave, im: Image.Image):
    """Cajas de cabeza/rostro con umbral bajo (0.10) en la foto completa: capta rostros tapados en parte."""
    if clave not in _CABEZAS:
        arr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        r = _seg(("human face", "head", "eyes", "eyeglasses")).predict(arr, conf=0.10, imgsz=640, verbose=False)[0]
        _CABEZAS[clave] = [[int(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()]
    return _CABEZAS[clave]


def _pixelar(im: Image.Image, mascara: np.ndarray, bloque: int):
    from PIL import ImageFilter
    W, H = im.size
    peq = im.resize((max(1, W // bloque), max(1, H // bloque)), Image.BILINEAR).resize((W, H), Image.NEAREST)
    borr = peq.filter(ImageFilter.GaussianBlur(max(4, bloque // 2)))
    alfa = Image.fromarray((mascara * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
    return Image.composite(borr, im, alfa)


def recortar_objeto(it):
    """Recorta la foto alrededor del objeto (margen 4 %), sin ampliar a cuadrado para no meter la cara.
    Si en el recorte queda parte de un rostro, se difumina SOLO el rostro: las cajas de rostro (YuNet y YOLOE
    'human face'/'head'/'eyes'/'eyeglasses', en la foto completa y en el recorte), ampliadas 60 %, menos la
    máscara del objeto, que queda siempre nítido. El recorte se centra sobre un fondo neutro cuadrado.
    Devuelve None si no se encuentra el objeto."""
    im = cargar(it)
    caja, mobj = _objeto(im, PROMPTS_RECORTE[it["clase"]])
    if caja is None:
        return None
    W, H = im.size
    x0, y0, x1, y1 = caja
    mx, my = (x1 - x0) * 0.04, (y1 - y0) * 0.04
    caja = (max(0, int(x0 - mx)), max(0, int(y0 - my)), min(W, int(x1 + mx)), min(H, int(y1 + my)))
    rec0 = im.crop(caja)
    caras = PRIV.caras(it["clave"], im) + _cabezas(it["clave"], im)
    caras += [[a + caja[0], b + caja[1], c + caja[0], d + caja[1]]
              for a, b, c, d in PRIV.caras(it["clave"] + "#recorte", rec0)]
    mcara = np.zeros((H, W), bool)
    for fx0, fy0, fx1, fy1 in caras:
        ex, ey = (fx1 - fx0) * 0.6, (fy1 - fy0) * 0.6
        mcara[max(0, int(fy0 - ey)):min(H, int(fy1 + ey)), max(0, int(fx0 - ex)):min(W, int(fx1 + ex))] = True
    mobj = cv2.erode(mobj.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    m = mcara & ~mobj
    sub = m[caja[1]:caja[3], caja[0]:caja[2]]
    it["rostros_en_recorte"] = int(sub.any())
    if sub.any():
        im = _pixelar(im, m, 24)
        it["difuminada"] = True
    rec = im.crop(caja)
    it["recorte"] = caja
    lado = max(rec.size)
    lienzo = Image.new("RGB", (lado, lado), FONDO_RECORTE)
    lienzo.paste(rec, ((lado - rec.width) // 2, (lado - rec.height) // 2))
    return lienzo


def preparar(it, permitir_difuminar=True):
    """Carga la imagen; si tiene rostros (o el conjunto pide anonimizar a la persona entera), la difumina.
    Devuelve None si tiene rostros y no se permite difuminar."""
    if it.get("recorte_objeto"):
        return recortar_objeto(it)
    im = cargar(it)
    cajas = PRIV.caras(it["clave"], im)
    persona = PRIV.con_persona(it["clave"])
    entera = bool(it.get("anon_persona"))
    if cajas or persona or entera:
        if not permitir_difuminar and not entera:
            return None
        it["difuminada"] = True
        return difuminar(im, cajas, persona_entera=True)
    return im


def elegir(items, n, semilla=42, por=lambda it: it.get("objeto", "")):
    """Elige n items repartidos entre objetos (round-robin). Primero solo fotos sin rostros; si no alcanzan,
    completa con fotos con los rostros difuminados. Devuelve (elegidos, n_difuminadas)."""
    rnd = random.Random(semilla)
    grupos = defaultdict(list)
    for it in items:
        grupos[por(it)].append(it)
    for g in grupos.values():
        rnd.shuffle(g)
    claves = sorted(grupos)
    rnd.shuffle(claves)
    elegidos = []
    for permitir in (False, True):
        pend = {k: list(grupos[k]) for k in claves}
        usados = {id(e) for e in elegidos}
        while len(elegidos) < n and any(pend[k] for k in claves):
            for k in claves:
                if len(elegidos) >= n:
                    break
                while pend[k]:
                    it = pend[k].pop()
                    if id(it) in usados or it["clave"] in EXCLUIR_MANUAL:
                        continue
                    im = preparar(it, permitir)
                    if im is not None:
                        it["im"] = im
                        elegidos.append(it)
                        usados.add(id(it))
                        break
    return elegidos, sum(1 for e in elegidos if e.get("difuminada"))


def cargar(it):
    s = it["src"]
    if isinstance(s, tuple):
        z, nombre = s
        with zipfile.ZipFile(z) as zf:
            return abrir(io.BytesIO(zf.read(nombre)))
    return abrir(s)


# ---------------------------------------------------------------- mosaico
def mosaico(filas, salida, lado=200, cols=6, etiqueta_ancho=230, pie_alto=0, titulo=None):
    """filas: lista de (titulo_fila, subtitulo, color_muestra|None, [items]); cada item tiene 'im' y
    opcional 'texto' (lista de líneas) y 'borde' (color)."""
    pad = 6
    bloques = []
    for nombre, sub, color, items in filas:
        n_rows = max(1, -(-len(items) // cols))
        bloques.append((nombre, sub, color, items, n_rows))
    celda_h = lado + pie_alto
    alto = sum(b[4] * (celda_h + pad) + pad * 2 for b in bloques) + (50 if titulo else 0)
    ancho = etiqueta_ancho + cols * (lado + pad) + pad
    lienzo = Image.new("RGB", (ancho, alto), FONDO)
    d = ImageDraw.Draw(lienzo)
    y = 0
    if titulo:
        d.text((pad * 2, 12), titulo, fill=TINTA, font=font(26, True))
        y = 50
    for nombre, sub, color, items, n_rows in bloques:
        alto_b = n_rows * (celda_h + pad) + pad
        d.line([(0, y), (ancho, y)], fill=(225, 225, 222), width=2)
        ty = y + 14
        if color is not None:
            d.rectangle([pad * 2, ty + 2, pad * 2 + 22, ty + 24], fill=color, outline=(90, 90, 90))
            tx = pad * 2 + 32
        else:
            tx = pad * 2
        d.text((tx, ty), nombre, fill=TINTA, font=font(22, True))
        for k, linea in enumerate(sub if isinstance(sub, list) else [sub]):
            d.text((pad * 2, ty + 34 + k * 24), linea, fill=TINTA2, font=font(18))
        for i, it in enumerate(items):
            r, c = divmod(i, cols)
            x0 = etiqueta_ancho + c * (lado + pad)
            y0 = y + pad + r * (celda_h + pad)
            th = cuadrado(it["im"], lado)
            if it.get("borde"):
                b = 7
                th = ImageOps.expand(cuadrado(it["im"], lado - 2 * b), border=b, fill=it["borde"])
            lienzo.paste(th, (x0, y0))
            for k, linea in enumerate(it.get("texto", [])):
                d.text((x0 + 2, y0 + lado + 3 + k * 20), linea, fill=TINTA if k == 0 else TINTA2,
                       font=font(17, k == 0))
        y += alto_b + pad
    lienzo.save(salida, quality=90)
    print("  ->", salida.relative_to(RAIZ), lienzo.size)
    return salida


# ---------------------------------------------------------------- 6.1 datos propios
def items_juliaca(filtro_carpeta, splits=("train", "val")):
    out = defaultdict(list)
    for sp in splits:
        for c in CLASES:
            for o in sorted((DS / "juliaca" / sp / c).glob("*")):
                if o.is_dir() and filtro_carpeta(o.name):
                    for p in imagenes(o):
                        out[c].append({"src": p, "clave": p.relative_to(RAIZ).as_posix(), "objeto": o.name,
                                       "split": sp, "clase": c})
    return out


def items_valcamara():
    out = defaultdict(list)
    for c in CLASES:
        for o in sorted((DS / "val_camara" / c).glob("*")):
            for p in imagenes(o):
                out[c].append({"src": p, "clave": p.relative_to(RAIZ).as_posix(), "objeto": o.name, "clase": c})
    return out


def figura_propia(por_clase, nombre, n=12, grupos=4):
    """Una o dos figuras (máx 'grupos' clases por figura). Devuelve metadatos para el pie."""
    clases = [c for c in CLASES if por_clase.get(c)]
    meta = {"clases": {}, "archivos": []}
    partes = [clases[i:i + grupos] for i in range(0, len(clases), grupos)]
    for k, parte in enumerate(partes):
        filas = []
        for c in parte:
            items = por_clase[c]
            objs = sorted({it["objeto"] for it in items})
            el, desc = elegir(items, n)
            for it in el:
                corto = it["objeto"].split("_20261")[0]
                corto = corto if len(corto) <= 22 else corto[:21] + "…"
                it["texto"] = [corto + (f" ({it['split']})" if "split" in it else "")]
            meta["clases"][c] = {"mostradas": len(el), "fotos": len(items), "objetos": len(objs),
                                 "objetos_mostrados": len({it["objeto"] for it in el}), "difuminadas": desc,
                                 "recortadas": sum(1 for it in el if "recorte" in it),
                                 "rostros_en_recorte": sum(1 for it in el if it.get("rostros_en_recorte")),
                                 "fotos_recorte": [{"foto": it["clave"], "caja": it["recorte"],
                                                    "rostros": it["rostros_en_recorte"]} for it in el if "recorte" in it]}
            filas.append((c, [f"tacho {TACHO[c][0]}", f"{len(el)} de {len(items)} fotos", f"{len(objs)} objetos"],
                          TACHO[c][1], el))
        suf = "" if len(partes) == 1 else "abcd"[k]
        f = mosaico(filas, OUT / f"{nombre}{suf}.jpg", pie_alto=24)
        meta["archivos"].append(f.name)
    return meta


# ---------------------------------------------------------------- 6.2 datos de internet
def items_prefijo(base: Path, prefijo, clases=CLASES):
    out = defaultdict(list)
    for c in clases:
        for p in sorted((base / c).glob(f"{prefijo}*")):
            if p.suffix.lower() in EXT:
                # "objeto" = imagen original (para no repetir recortes de la misma foto)
                partes = p.stem.split("_")
                out[c].append({"src": p, "clave": p.relative_to(RAIZ).as_posix(), "clase": c,
                               "objeto": "_".join(partes[:3])})
    return out


def figura_fuente(por_clase, nombre, n_por_clase, cols=6, titulo=None):
    filas, meta = [], {}
    for c, n in n_por_clase.items():
        items = por_clase.get(c, [])
        if not items:
            continue
        el, desc = elegir(items, n)
        filas.append((c, [f"tacho {TACHO[c][0]}", f"{len(el)} de {len(items)}"], TACHO[c][1], el))
        meta[c] = {"mostradas": len(el), "total": len(items), "difuminadas": desc}
    mosaico(filas, OUT / f"{nombre}.jpg", cols=cols, titulo=titulo)
    return meta


def figura_no_finales():
    ext = DS / "externo" / "train"
    fuentes = [("realwaste", "RealWaste (UCI)"), ("tacox", "TACO extra"), ("trashnet", "TrashNet"),
               ("frg", "Freiburg Groceries"), ("gro", "Grocery Store"), ("bdw", "BDWaste"),
               ("msk", "MaskNet"), ("ewa", "E-Waste v9"), ("rcb", "RecyBat24 v1"),
               ("rfv4", "Garbage Class. v4"), ("rfmed", "Smart medical v3")]
    filas, meta = [], {}
    for pref, etiqueta in fuentes:
        items = []
        for c in CLASES:
            for p in sorted((ext / c).glob(f"{pref}_*")):
                items.append({"src": p, "clave": p.relative_to(RAIZ).as_posix(), "clase": c, "objeto": c})
        el, desc = elegir(items, 6)  # round-robin por clase
        for it in el:
            it["texto"] = [it["clase"]]
        cnt = defaultdict(int)
        for it in items:
            cnt[it["clase"]] += 1
        filas.append((etiqueta, [f"{len(items)} imágenes", f"prefijo {pref}_"], None, el))
        meta[pref] = {"total": len(items), "por_clase": dict(cnt), "difuminadas": desc}
    mosaico(filas, OUT / "fig_internet_no_final.jpg", pie_alto=22, lado=180)
    return meta


# ---------------------------------------------------------------- 6.3 descartados
def figura_duplicado():
    a = RAIZ / "evaluacion_docente/imagenes/_descargas/med/train/Infectious/IMG_6894_JPG.rf.6b32234109a417fd7e84b8f7ccf11c9d.jpg"
    b = (RAIZ / "evaluacion_docente/imagenes/test_docente/Smart medical waste segregation.v3i.folder/Infectious/"
         "IMG_6894_JPG_jpg.rf.febd15d75de1185cee8454e4e2b9e7ba.jpg")
    ia, ib = abrir(a), abrir(b)
    for p, im in ((a, ia), (b, ib)):
        if PRIV.caras(p.relative_to(RAIZ).as_posix(), im):
            print("  [AVISO] el par duplicado tiene un posible rostro:", p.name)

    def dhash(im, s=8):
        g = np.asarray(im.convert("L").resize((s + 1, s), Image.BILINEAR), dtype=int)
        return (g[:, 1:] > g[:, :-1]).flatten()
    dist = int((dhash(ia) != dhash(ib)).sum())
    lado, pad, alto_t = 420, 16, 96
    L = Image.new("RGB", (lado * 2 + pad * 3, lado + alto_t + pad * 2), FONDO)
    d = ImageDraw.Draw(L)
    for k, (im, t1, t2) in enumerate(((ia, "Candidata externa (descartada)", "Smart medical v3 · split train"),
                                      (ib, "Imagen del benchmark (test docente)", "Smart medical v3 · Infectious"))):
        x = pad + k * (lado + pad)
        L.paste(ImageOps.expand(cuadrado(im, lado - 8), border=4, fill=MAL if k == 0 else (90, 90, 90)), (x, pad))
        d.text((x, lado + pad + 8), t1, fill=TINTA, font=font(20, True))
        d.text((x, lado + pad + 34), t2, fill=TINTA2, font=font(17))
        d.text((x, lado + pad + 58), "IMG_6894 (mismo original)", fill=TINTA2, font=font(17))
    L.save(OUT / "fig_duplicado_benchmark.jpg", quality=92)
    print("  -> fig_duplicado_benchmark.jpg dHash", dist)
    return {"dhash": dist, "a": a.name, "b": b.name}


def figura_rechazadas():
    filas, meta = [], {}
    # 1. Household (Kaggle alistairking), parte real_world
    hh = DS / "externo/_fuentes/household/extraido/images/images"
    items = [{"src": p, "clave": p.relative_to(RAIZ).as_posix(), "objeto": p.parts[-3]}
             for p in sorted(hh.glob("*/real_world/*")) if p.suffix.lower() in EXT]
    el, desc = elegir(items, 6, semilla=7)
    for it in el:
        it["texto"] = [it["objeto"][:20]]
    filas.append(("Household Waste", ["Kaggle alistairking", "Motivo: fotos de stock,", "gráficos, duplicados"], None, el))
    meta["household"] = {"total": len(items), "difuminadas": desc}
    # 2. Surgical Waste Dataset (zip)
    z = DATA_NEW / "Surgical Waste Dataset.zip"
    with zipfile.ZipFile(z) as zf:
        nombres = [n for n in zf.namelist() if n.lower().endswith(".jpg")]
    items = [{"src": (z, n), "clave": "zip:surgical/" + n, "objeto": n.split("/")[-1].split("_")[0]} for n in nombres]
    el, desc = elegir(items, 6, semilla=3)
    for it in el:
        it["texto"] = [it["src"][1].split("/")[-1][:20]]
    filas.append(("Surgical Waste", ["Mendeley rnhz3fvbj2", "Motivo: stock con marca", "de agua, cuadros de video"], None, el))
    meta["surgical"] = {"total": len(items), "difuminadas": desc}
    # 3. Freiburg CHOCOLATE / JUICE (envase mezclado)
    fr = DATA_NEW / "freiburg_groceries_dataset" / "images"
    items = [{"src": p, "clave": "frg:" + p.name, "objeto": p.parent.name}
             for cat in ("CHOCOLATE", "JUICE", "COFFEE") for p in sorted((fr / cat).glob("*.png"))]
    el, desc = elegir(items, 6, semilla=5)
    for it in el:
        it["texto"] = [it["objeto"]]
    filas.append(("Freiburg (clases)", ["CHOCOLATE, JUICE,", "COFFEE. Motivo: mezclan", "papel, vidrio y plástico"], None, el))
    meta["freiburg"] = {"total": len(items), "difuminadas": desc}
    # 4. TACO, recortes descartados en la revisión manual
    td = DS / "externo/_descartados"
    items = [{"src": p, "clave": p.relative_to(RAIZ).as_posix(), "objeto": p.parent.name}
             for p in sorted(td.rglob("*")) if p.suffix.lower() in EXT]
    el, desc = elegir(items, 6, semilla=9)
    for it in el:
        it["texto"] = [it["objeto"].replace("_", " ")[:20]]
    filas.append(("TACO (recortes)", ["revisión manual", "Motivo: material dudoso,", "recorte ilegible"], None, el))
    meta["taco_descartados"] = {"total": len(items), "difuminadas": desc}
    mosaico(filas, OUT / "fig_rechazadas.jpg", pie_alto=22, lado=180, etiqueta_ancho=270)
    return meta


# ---------------------------------------------------------------- predicciones (solo validación)
def predicciones():
    f = DAT / "predicciones_validacion.csv"
    if f.exists():
        return list(csv.DictReader(open(f, encoding="utf-8")))
    from ultralytics import YOLO
    m = YOLO(str(MODELO))
    fotos = []
    for c in CLASES:
        for p in imagenes(DS / "val_camara" / c):
            fotos.append(("val_camara", c, p.parent.name, p))
        for p in imagenes(DS / "juliaca" / "val" / c):
            fotos.append(("juliaca/val", c, p.parent.name, p))
    filas = []
    for i in range(0, len(fotos), 32):
        lote = fotos[i:i + 32]
        for (conj, c, o, p), r in zip(lote, m.predict([str(x[3]) for x in lote], imgsz=224, verbose=False)):
            pr = r.probs.data.cpu().numpy()
            k = int(pr.argmax())
            filas.append({"conjunto": conj, "clase_real": c, "objeto": o, "foto": p.relative_to(RAIZ).as_posix(),
                          "prediccion": m.names[k], "confianza": round(float(pr[k]), 4),
                          "prob_real": round(float(pr[CLASES.index(c)]), 4)})
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    return filas


def verificar_predicciones(filas):
    res = {}
    for conj, js in (("val_camara", "etapa2_e2_nf_s43_20261110_2132_val_camara.json"),
                     ("juliaca/val", "etapa2_e2_nf_s43_20261110_2132_juliaca_val.json")):
        sub = [r for r in filas if r["conjunto"] == conj]
        ok = sum(r["clase_real"] == r["prediccion"] for r in sub)
        ref = json.load(open(RAIZ / "reportes" / js, encoding="utf-8"))
        res[conj] = {"n": len(sub), "aciertos": ok, "top1": round(ok / len(sub), 4), "top1_informe": ref["accuracy_top1"]}
    print("  verificación:", res)
    return res


def figura_predicciones(filas):
    filas_fig, meta = [], {}
    for c in CLASES:
        conj = "juliaca/val" if c == "sin_residuo" else "val_camara"
        sub = [r for r in filas if r["clase_real"] == c and r["conjunto"] == conj]
        ok = [dict(r, src=RAIZ / r["foto"], clave=r["foto"]) for r in sub if r["prediccion"] == c]
        mal = [dict(r, src=RAIZ / r["foto"], clave=r["foto"]) for r in sub if r["prediccion"] != c]
        e_ok, _ = elegir(ok, 3, semilla=11)
        # errores: los de mayor confianza, de objetos distintos si se puede
        mal.sort(key=lambda r: -float(r["confianza"]))
        e_mal, vistos = [], set()
        for permitir in (False, True):          # primero fotos sin personas; luego difuminadas
            for ronda in (0, 1):                # primero objetos distintos
                for r in mal:
                    if len(e_mal) >= 3:
                        break
                    if r in e_mal or r["clave"] in EXCLUIR_MANUAL or (ronda == 0 and r["objeto"] in vistos):
                        continue
                    im = preparar(r, permitir)
                    if im is None:
                        continue
                    r["im"] = im
                    e_mal.append(r)
                    vistos.add(r["objeto"])
        for r in e_ok:
            r["borde"] = OK
            r["texto"] = [f"ACIERTO  {float(r['confianza']):.2f}", f"pred: {r['prediccion']}"]
        for r in e_mal:
            r["borde"] = MAL
            r["texto"] = [f"ERROR  {float(r['confianza']):.2f}", f"pred: {r['prediccion']}"]
        n_ok = len(ok)
        filas_fig.append((c, [f"real: {c}", f"{conj}", f"aciertos {n_ok}/{len(sub)}"], TACHO[c][1], e_ok + e_mal))
        meta[c] = {"conjunto": conj, "n": len(sub), "aciertos": n_ok}
    mosaico(filas_fig[:4], OUT / "fig_predicciones_a.jpg", pie_alto=44, etiqueta_ancho=230)
    mosaico(filas_fig[4:], OUT / "fig_predicciones_b.jpg", pie_alto=44, etiqueta_ancho=230)
    return meta


# ---------------------------------------------------------------- gráficas de resultados
AZUL, NARANJA = "#2a78d6", "#eb6834"


def matriz(js, salida, titulo):
    d = json.load(open(RAIZ / "reportes" / js, encoding="utf-8"))
    m = np.array(d["matriz_real_x_predicho"])
    filas = [i for i in range(8) if m[i].sum() > 0]
    m2 = m[filas]
    norm = m2 / m2.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(filas) + 1.6), dpi=200)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
    ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(filas)):
        for j in range(8):
            v = m2[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b", fontweight="bold" if filas[i] == j else None)
    ax.set_xticks(range(8), CLASES, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(filas)), [f"{CLASES[i]} (n={m[i].sum()})" for i in filas], fontsize=8.5)
    ax.set_xlabel("clase predicha", fontsize=9, color="#52514e")
    ax.set_ylabel("clase real", fontsize=9, color="#52514e")
    ax.set_title(titulo, fontsize=10, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / salida)
    plt.close(fig)
    return {"top1": d["accuracy_top1"], "f1": d["f1_macro"], "n": int(m.sum()), "aciertos": int(np.trace(m))}


def barras_f1(salida):
    a = json.load(open(RAIZ / "reportes/etapa2_e2_nf_s43_20261110_2132_val_camara.json", encoding="utf-8"))
    b = json.load(open(RAIZ / "reportes/etapa2_nf_s43_juliaca_test.json", encoding="utf-8"))
    fa = {r["clase"]: (r["f1"], r["n"]) for r in a["por_clase"]}
    fb = {r["clase"]: (r["f1"], r["n"]) for r in b["por_clase"]}
    x = np.arange(8)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.4, 3.3), dpi=200)
    va = [fa[c][0] if fa[c][1] else np.nan for c in CLASES]
    vb = [fb[c][0] for c in CLASES]
    ax.bar(x - w / 2 - 0.01, va, w, color=AZUL, label="val_camara (293 fotos, elección del modelo)")
    ax.bar(x + w / 2 + 0.01, vb, w, color=NARANJA, label="juliaca/test (98 fotos, una sola evaluación)")
    for i, c in enumerate(CLASES):
        if fb[c][1] and vb[i] == 0:
            ax.text(x[i] + w / 2, 0.015, "0.00", ha="center", fontsize=7, color="#52514e")
        if not fa[c][1]:
            ax.text(x[i] - w / 2, 0.02, "sin\nfotos", ha="center", fontsize=6.5, color="#52514e")
    ax.set_xticks(x, CLASES, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 por clase", fontsize=9, color="#52514e")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.14), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / salida)
    plt.close(fig)


def curvas(salida):
    def leer(run):
        rows = list(csv.DictReader(open(RAIZ / "runs" / run / "results.csv", encoding="utf-8")))
        return {k.strip(): np.array([float(r[k]) for r in rows]) for k in rows[0]}
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=200)
    for ax, run, tit in ((axs[0], "etapa1_e1_nf", "Etapa 1 (e1_nf): val = publico/val"),
                         (axs[1], "etapa2_e2_nf_s43", "Etapa 2 (e2_nf_s43): val = juliaca/val")):
        df = leer(run)
        ax.plot(df["epoch"], df["train/loss"], color=AZUL, lw=2, label="loss train")
        ax.plot(df["epoch"], df["val/loss"], color=NARANJA, lw=2, label="loss val")
        ax.set_title(tit, fontsize=8.5, loc="left")
        ax.set_xlabel("época", fontsize=8, color="#52514e")
        ax.grid(color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, length=0)
        ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / salida)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6), dpi=200)
    for ax, run in ((axs[0], "etapa1_e1_nf"), (axs[1], "etapa2_e2_nf_s43")):
        df = leer(run)
        ax.plot(df["epoch"], df["metrics/accuracy_top1"], color=AZUL, lw=2, label="top-1 val")
        ax.set_ylim(0, 1)
        ax.set_xlabel("época", fontsize=8, color="#52514e")
        ax.grid(color="#e6e6e3", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, length=0)
        ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / salida.replace(".png", "_top1.png"))
    plt.close(fig)


def comparacion_recetas(salida):
    # medias de F1 macro en val_camara (3 semillas), copiadas de reportes/*_val_camara.json
    grupos = {"j1028": "etapa2_e2_ext_full_j1028{}_*_val_camara.json", "rfA": "etapa2_e2_rfA_s4{}_*_val_camara.json",
              "rfB": "etapa2_e2_rfB_s4{}_*_val_camara.json", "nf": "etapa2_e2_nf_s4{}_*_val_camara.json",
              "tienda": "etapa2_e2_tienda_s4{}_*_val_camara.json", "gro": "etapa2_e2_gro_s4{}_*_val_camara.json",
              "nf_m": "etapa2_e2_nf_m_s4{}_*_val_camara.json", "pel A": "etapa2_e2_pel_s4{}_*_val_camara.json",
              "pel B": "etapa2_e2c_pel_s4{}_*_val_camara.json"}
    res = {}
    for g, pat in grupos.items():
        vals = []
        for s in ("2", "3", "4"):
            if g == "j1028":
                p = list((RAIZ / "reportes").glob(pat.format("" if s == "2" else f"_s4{s}")))
                p = [x for x in p if (s != "2") or ("_s4" not in x.name)]
            else:
                p = list((RAIZ / "reportes").glob(pat.format(s)))
            if p:
                vals.append(json.load(open(sorted(p)[0], encoding="utf-8"))["f1_macro"])
        res[g] = vals
    orden = sorted(res, key=lambda g: np.mean(res[g]))
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    for i, g in enumerate(orden):
        v = res[g]
        col = AZUL if g == "nf" else "#9a9a96"
        ax.plot([min(v), max(v)], [i, i], color=col, lw=2, alpha=0.6)
        ax.scatter(v, [i] * len(v), s=22, color=col, zorder=3)
        ax.scatter([np.mean(v)], [i], s=70, color=col, marker="D", zorder=4, edgecolor="white", linewidth=1.5)
        ax.text(max(v) + 0.004, i, f"media {np.mean(v):.4f}", va="center", fontsize=7.5, color="#0b0b0b")
    ax.set_yticks(range(len(orden)), orden, fontsize=8.5)
    ax.set_xlabel("F1 macro en val_camara (puntos = semillas 42/43/44; rombo = media)", fontsize=8, color="#52514e")
    ax.grid(axis="x", color="#e6e6e3", lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlim(0.46, 0.62)
    fig.tight_layout()
    fig.savefig(OUT / salida)
    plt.close(fig)
    return res


# ---------------------------------------------------------------- main
def main():
    meta = {}
    print("[1] predicciones de validación")
    filas = predicciones()
    meta["verificacion_predicciones"] = verificar_predicciones(filas)

    print("[2] datos propios")
    it27 = items_juliaca(lambda n: "_20261027-" in n)
    for lista in it27.values():
        for it in lista:
            # webcam frente al usuario: se recorta alrededor del objeto (y se difumina solo el rostro si queda)
            it["recorte_objeto"] = True
    meta["laptop_1027"] = figura_propia(it27, "fig_propio_laptop_1027")
    meta["plazavea_1028"] = figura_propia(items_juliaca(lambda n: "_20261028-" in n), "fig_propio_plazavea_1028")
    meta["val_camara"] = figura_propia(items_valcamara(), "fig_propio_val_camara")
    PRIV.guardar()

    print("[3] datos de internet (modelo final)")
    pub = DS / "publico" / "train"
    meta["publico"] = figura_fuente(items_prefijo(pub, "companero_"), "fig_internet_publico",
                                    {c: 6 for c in CLASES if c != "sin_residuo"})
    ext = DS / "externo" / "train"
    meta["taco"] = figura_fuente(items_prefijo(ext, "taco_"), "fig_internet_taco", {"generales": 6, "metales": 6})
    meta["fwd"] = figura_fuente(items_prefijo(ext, "fwd_"), "fig_internet_fwd", {"organico": 12})
    meta["nx"] = figura_fuente(items_prefijo(ext, "nx_"), "fig_internet_nx", {"organico": 6, "peligrosos": 6})
    meta["no_final"] = figura_no_finales()
    PRIV.guardar()

    print("[4] descartados")
    meta["duplicado"] = figura_duplicado()
    meta["rechazadas"] = figura_rechazadas()
    PRIV.guardar()

    print("[5] predicciones")
    meta["predicciones"] = figura_predicciones(filas)
    PRIV.guardar()

    print("[6] gráficas")
    meta["cm_val_camara"] = matriz("etapa2_e2_nf_s43_20261110_2132_val_camara.json", "fig_cm_val_camara.png",
                                   "etapa2_nf_s43 en val_camara (filas normalizadas; número = fotos)")
    meta["cm_test"] = matriz("etapa2_nf_s43_juliaca_test.json", "fig_cm_test.png",
                             "etapa2_nf_s43 en juliaca/test (evaluación única del 2026-11-11)")
    barras_f1("fig_f1_por_clase.png")
    curvas("fig_curvas.png")
    meta["recetas"] = comparacion_recetas("fig_recetas.png")
    if all((APP_CAPTURAS / a).exists() for a, _ in APP_PESTANAS):
        print("[7] app TachoYaa")
        meta["app"] = figura_app()
    priv = list(PRIV.cache.values())
    meta["privacidad"] = {"revisadas": len(priv), "con_rostro_o_persona": sum(str(r["n_rostros"]) != "0" or str(r["n_persona"]) != "0" for r in priv)}
    json.dump(meta, open(DAT / "figuras_meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("listo:", meta["privacidad"])


APP_CAPTURAS = RAIZ.parent / "entregables" / "capturas"
APP_PESTANAS = [("1_escanear.png", "Escanear"), ("2_historial.png", "Historial"),
                ("3_guia_ntp.png", "Guía NTP"), ("4_acerca_de.png", "Acerca de")]
# Barra de estado del emulador que se quita en la figura (1_escanear.png ya viene recortada del celular).
APP_RECORTE_ARRIBA = {"2_historial.png": 120, "3_guia_ntp.png": 120, "4_acerca_de.png": 120}


def figura_app(salida="fig_app_pestanas.png", alto=900, sep=28, pie=56):
    """Las 4 pestañas de la app TachoYaa lado a lado (capturas de ../entregables/capturas/).

    Solo lee esas capturas (revisadas a mano: sin rostros ni el cuarto del autor) y escribe la figura.
    """
    ims = []
    for archivo, _ in APP_PESTANAS:
        im = Image.open(APP_CAPTURAS / archivo).convert("RGB")
        im = im.crop((0, APP_RECORTE_ARRIBA.get(archivo, 0), im.width, im.height))
        ims.append(im.resize((round(im.width * alto / im.height), alto), Image.LANCZOS))
    ancho = sum(im.width for im in ims) + sep * (len(ims) + 1)
    lienzo = Image.new("RGB", (ancho, alto + 2 * sep + pie), FONDO)
    dib = ImageDraw.Draw(lienzo)
    x = sep
    for im, (_, nombre) in zip(ims, APP_PESTANAS):
        lienzo.paste(im, (x, sep))
        dib.rectangle([x - 1, sep - 1, x + im.width, sep + alto], outline=(200, 200, 198), width=2)
        f = font(30, bold=True)
        w = dib.textlength(nombre, font=f)
        dib.text((x + (im.width - w) / 2, sep + alto + 14), nombre, fill=TINTA, font=f)
        x += im.width + sep
    lienzo.save(OUT / salida)
    print("figura:", OUT / salida, lienzo.size)
    return {"capturas": [a for a, _ in APP_PESTANAS], "tamano": lienzo.size}


if __name__ == "__main__":
    import sys

    if "--solo-app" in sys.argv:  # solo la figura de la app (no rehace las demás)
        figura_app()
    else:
        main()
