"""Fuentes externas candidatas del 2026-11-10 (peligrosos y organico), regla 1 de CLAUDE.md.

  python -I scripts/importar_candidatas.py                    # DRY-RUN (por defecto): dedup, conteos, mosaicos
  python -I scripts/importar_candidatas.py --importar --topes fwd=300 nx=100 bmw=300 --categorias ...
                                                              # SOLO tras el OK del usuario

Los zips se leen en memoria desde ~/Downloads (no se extraen ni se ejecuta nada de ellos).
Fuentes evaluadas (licencias verificadas el 2026-11-10):
  fwd : Food Waste Detection v1 (Roboflow kkh/food-waste-detection-jghxg-ziq5h), CC BY 4.0, DETECCIÓN -> recortes.
        Una sola copia por original (Roboflow trae 3 versiones aumentadas). Solo los grupos de fotos que no vienen
        estiradas (bandeja_T: tazones redondos a 1:1; NxN: el nombre dice 1280x1280). Se descartan 'hash' (imágenes
        web de proporción desconocida, con ilustraciones) y 'cocina'/'still' (cámara de cocina 4:3 con copias giradas
        90 grados: no se sabe en qué eje corregir; además dan pocos recortes >= 64 px). Drink y Other-waste se descartan.
  nx  : NexTech14_Waste_Dataset (Mendeley p6hgv27jh4, Salam et al.), CC BY-NC 4.0, clasificación. Una imagen por
        original (<subclase>-<n>; se prefiere la no aumentada). organic -> organico; hazardous: battery -> peligrosos
        solo si el usuario lo aprueba tras el mosaico, e-waste dudoso; recyclable descartado.
  bmw : Biomedical Waste Classification Images (Kaggle mario78), MIT. Solo Dataset/Raw/ (Baackground_Removed son las
        mismas fotos con fondo recortado).
  pbw : Pharmaceutical&Biomedical Waste v1 (Roboflow medical-waste-disposal), Public Domain, DETECCIÓN -> recortes.
  mwn : Medical Waste v2 (Roboflow ninja-9t5r9), CC BY 4.0, DETECCIÓN -> recortes.
        pbw y mwn se evalúan a pedido del usuario, pero son de la MISMA FAMILIA que 'Smart medical waste segregation'
        del benchmark (pbw: nombres LINE_ALBUM_*/IMG_*_JPG = los del test médico, 30 originales por nombre; mwn:
        timestamps 1671*, 33 originales en el test del benchmark y 99 en rfmed_). Vienen estiradas a 640x640 sin
        forma de recuperar la proporción: se recortan sin corregir. Las cajas de ambas se ven mal puestas en muestras.
        DECISIÓN DEL USUARIO (2026-11-10): pbw y mwn DESCARTADOS (familia Smart medical + etiquetas incorrectas). Ya no
        son candidatas; solo se usan como referencia del filtro de familia.
Descartada sin procesar: bio medical waste classification v3 (sin etiquetas, nc 0).

Filtro de FAMILIA SMART MEDICAL (pedido del usuario; bmw contiene 110 originales de pbw): se elimina toda candidata
que coincida por nombre original, dHash <= 4 (8 giros/espejos de cada imagen de la familia, todas sus copias) u ORB
(>= 25 con las 5 vecinas por embedding, una copia por original) con CUALQUIER imagen del zip 'Smart medical waste
segregation' completo (train, valid y test), pbw y mwn completos (todos los splits) y rfmed_ de externo/train.
Nombre y dHash se aplican a todas las fuentes; ORB de familia a bmw.

Filtros, en orden (se anota el primero que elimina): sobre la IMAGEN COMPLETA
  1. nombre_benchmark  2. dhash_benchmark (8 giros/espejos, 1.455 + 238 gris)  3. dhash_propios (publico, juliaca,
  val_camara, externo/train y externo/_fuentes)  4. orb_benchmark (ORB+RANSAC >= 25 con las 5 vecinas por embedding)
  y luego sobre lo que se importaría (imagen o recorte):
  5. recorte_chico (lado menor < 64 px)  6. dhash_propios_recorte (solo recortes)  7. dup_interno (dHash <= 4 con
  8 giros/espejos contra lo ya aceptado)  8. fuera_del_tope.
Informes: reportes/externo_candidatas_{dedup,conteos}.csv y reportes/mosaicos_candidatas/<fuente>_<categoria>.jpg
"""
import argparse
import csv
import io
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES, EXT, EXTERNO, RAIZ, imagenes  # noqa: E402
from importar_roboflow import (UMBRAL, banco_benchmark, banco_propios, dhash_8, dhash_pil, dist,  # noqa: E402
                               filtro_orb, original)

DESCARGAS = Path.home() / "Downloads"
INF = RAIZ / "reportes"
MOSAICOS = INF / "mosaicos_candidatas"
REGISTRO = RAIZ / "datos_externos.csv"
MIN_LADO = 64
PAD = 0.10  # margen alrededor de cada caja (fracción del lado)

FUENTES = {
    "fwd": dict(zip="Food Waste Detection.v1i.yolov8.zip", licencia="CC BY 4.0",
                nombre="Food Waste Detection v1 (Roboflow Universe, kkh), recortes de cajas",
                url="https://universe.roboflow.com/kkh/food-waste-detection-jghxg-ziq5h"),
    "nx": dict(zip="NexTech14_Waste_Dataset A Benchmark Image Collecti.zip", licencia="CC BY-NC 4.0",
               nombre="NexTech14_Waste_Dataset (Mendeley Data p6hgv27jh4, Salam et al.)",
               url="https://data.mendeley.com/datasets/p6hgv27jh4"),
    "pbw": dict(zip="Pharmaceutical-Biomedical Waste.v1i.yolov8.zip", licencia="Public Domain",
                nombre="Pharmaceutical&Biomedical Waste v1 (Roboflow Universe, medical-waste-disposal), recortes",
                url="https://universe.roboflow.com/medical-waste-disposal/pharmaceutical-biomedical-waste"),
    "mwn": dict(zip="Medical Waste.v2i.yolov8.zip", licencia="CC BY 4.0",
                nombre="Medical Waste v2 (Roboflow Universe, ninja-9t5r9), recortes",
                url="https://universe.roboflow.com/ninja-9t5r9/medical-waste-zltxs"),
    "bmw": dict(zip="archive (2).zip", licencia="MIT",
                nombre="Biomedical Waste Classification Images Dataset (Kaggle mario78), carpeta Raw",
                url="https://www.kaggle.com/datasets/mario78/biomedical-waste-classification-images-dataset"),
}
# categoría -> (clase, estado). estado: 'propuesta' (mapeo claro), 'dudosa' (mosaico para que decida el usuario).
MAPEO_BMW = {
    "used_masks": ("peligrosos", "propuesta"), "used_medical_gloves": ("peligrosos", "propuesta"),
    "used_syringes": ("peligrosos", "propuesta"), "blood_soaked_bandages": ("peligrosos", "propuesta"),
    "expired_tablets": ("peligrosos", "propuesta"),
    "vaccine_or_medicine_vials": ("peligrosos", "dudosa"), "ampoules_full": ("peligrosos", "dudosa"),
    "ampuoles_broken": ("peligrosos", "dudosa"), "syrup_bottles": ("peligrosos", "dudosa"),
    "general_organic_waste": ("organico", "dudosa"),
    # descartadas: instrumental metálico, frascos sin medicina, órganos, papel (viene de Pharmaceutical-Biomedical),
    # botellas de agua y la carpeta 'uncategorized_or_overlapping'
    **{k: (None, "descartada") for k in (
        "disinfectant_bottles", "iv_bottles", "waterbottles", "forceps", "hemostats", "scalpels", "tweezers",
        "episiotomy_scissors", "mayo_scissors", "stitch_removal_scissors", "human_organs", "used_medical_paper",
        "uncategorized_or_overlapping")},
}
MAPEO_NX = {"food-waste": ("organico", "propuesta"), "other-organic": ("organico", "propuesta"),
            "garden-waste": ("organico", "propuesta"), "battery": ("peligrosos", "dudosa"),
            "e-waste": ("peligrosos", "dudosa")}
FWD_DESCARTE = {"Drink", "Other-waste"}
# Detección: clase de la caja -> (clase, estado); lo que no está se descarta. pbw y mwn se evalúan a pedido del
# usuario (2026-11-10) aunque son de la familia de 'Smart medical waste segregation' (ver docstring); sus imágenes
# vienen estiradas a 640x640 y la proporción original no se puede recuperar (ni nombres ni README la dan; el
# benchmark las trae a 224x224): se recortan sin corregir y se marca en el informe.
MAPEO_DET_TODAS = ("fwd", "pbw", "mwn")
MAPEO_DET = {
    "pbw": {"Gauze": ("peligrosos", "propuesta"), "Gloves": ("peligrosos", "propuesta"),
            "Mask": ("peligrosos", "propuesta"), "Syringe": ("peligrosos", "propuesta"),
            "-SN- Syringe needles": ("peligrosos", "propuesta")},
    "mwn": {"gauze": ("peligrosos", "propuesta"), "glove_pair_latex": ("peligrosos", "propuesta"),
            "glove_pair_surgery": ("peligrosos", "propuesta"), "medical_cap": ("peligrosos", "dudosa")},
}


def grupo_fwd(o):
    if o.startswith("image_"):
        return "cocina"
    if re.fullmatch(r"\d{8}T\d{6}_jpg", o):
        return "bandeja_T"
    if "Still" in o:
        return "still"
    if re.search(r"\d+x\d+", o):
        return "NxN"
    return "hash"


def cands_det(fu, cfg, desc):
    """Detección (fwd, pbw, mwn): una copia por original (valid primero), splits train y valid (test excluido)."""
    z = zipfile.ZipFile(DESCARGAS / cfg["zip"])
    names = yaml.safe_load(z.read("data.yaml"))["names"]
    ims = sorted(x for x in z.namelist() if "/images/" in x and x.split("/")[0] in ("train", "valid")
                 and Path(x).suffix.lower() in EXT and not Path(x).name.startswith("."))
    ims.sort(key=lambda x: x.split("/")[0] != "valid")  # valid (sin augmentation) primero
    vistos, res = set(), []
    for x in ims:
        o = original(x.split("/")[-1])
        if o in vistos:
            desc[(fu, "copia_del_mismo_original")] += 1
            continue
        vistos.add(o)
        g = grupo_fwd(o) if fu == "fwd" else "-"
        if fu == "fwd" and g not in ("bandeja_T", "NxN"):
            desc[(fu, f"grupo_{g}")] += 1
            continue
        cajas = []
        lab = x.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        for r in (z.read(lab).decode().splitlines() if lab in z.NameToInfo else []):
            if not r.strip():
                continue
            c, xc, yc, w, h = r.split()[:5]
            cat = names[int(c)]
            if fu == "fwd":
                destino = None if cat in FWD_DESCARTE else ("organico", "propuesta")
            else:
                destino = MAPEO_DET[fu].get(cat)
            if destino is None:
                desc[(fu, f"caja_{cat}")] += 1
                continue
            cajas.append((cat, *destino, float(xc), float(yc), float(w), float(h)))
        res.append(dict(fuente=fu, zip=z, ruta_zip=x, original=o, split=x.split("/")[0], grupo=g, cajas=cajas))
    return res


def cands_nx(cfg, desc):
    zf = zipfile.ZipFile(DESCARGAS / cfg["zip"])
    z = zipfile.ZipFile(io.BytesIO(zf.read(zf.namelist()[0])))
    porig = {}
    for x in sorted(z.namelist()):
        if not x.endswith(".jpg"):
            continue
        _, split, macro, archivo = x.split("/")
        if macro == "recyclable":
            desc[("nx", "recyclable")] += 1
            continue
        m = re.match(r"([a-z\-]+?)-(\d+)", archivo)
        sub, o = m.group(1), f"{m.group(1)}-{m.group(2)}"
        k = (sub, o)
        if k not in porig or ("_aug" in porig[k]["archivo"] and "_aug" not in archivo):
            porig[k] = dict(fuente="nx", zip=z, ruta_zip=x, original=o, split=split, categoria=sub, archivo=archivo)
    n_tot = sum(1 for x in z.namelist() if x.endswith(".jpg") and "/recyclable/" not in x)
    desc[("nx", "copia_aumentada_mismo_original")] += n_tot - len(porig)
    return list(porig.values())


def cands_bmw(cfg, desc):
    z = zipfile.ZipFile(DESCARGAS / cfg["zip"])
    res = []
    for x in sorted(z.namelist()):
        p = x.split("/")
        if len(p) != 4 or Path(x).suffix.lower() not in EXT or ".." in p:
            continue
        if p[1] != "Raw":
            desc[("bmw", "Baackground_Removed")] += 1
            continue
        if p[2] not in MAPEO_BMW:
            sys.exit(f"ERROR: bmw tiene una clase sin decidir: {p[2]}")
        if MAPEO_BMW[p[2]][0] is None:
            desc[("bmw", p[2])] += 1
            continue
        res.append(dict(fuente="bmw", zip=z, ruta_zip=x, original=original(p[3]), split="-", categoria=p[2],
                        archivo=p[3]))
    return res


ZIP_SMARTMED = "Smart medical waste segregation.v3i.folder.zip"


def banco_familia():
    """Familia Smart medical: zip completo (todos los splits), pbw y mwn completos, rfmed_ de externo/train.
    Devuelve nombres originales, dHash (8 giros/espejos por imagen), quién, e imágenes 224 (una por original) para ORB."""
    fuentes = []
    z = zipfile.ZipFile(DESCARGAS / ZIP_SMARTMED)
    fuentes += [("smartmed", z, x) for x in z.namelist() if Path(x).suffix.lower() in EXT]
    for fu in ("pbw", "mwn"):
        z = zipfile.ZipFile(DESCARGAS / FUENTES[fu]["zip"])
        fuentes += [(fu, z, x) for x in z.namelist() if "/images/" in x and Path(x).suffix.lower() in EXT
                    and not Path(x).name.startswith(".")]
    rfmed = [p for c in CLASES for p in imagenes(EXTERNO / "train" / c) if p.name.startswith("rfmed_")]
    nombres, hs, quien, ims_orb, quien_orb, vistos = {}, [], [], [], [], set()
    for fu, z, x in fuentes + [("rfmed", None, p) for p in rfmed]:
        if z is None:
            nom = x.name.split("_", 2)[2]
            im = Image.open(x).convert("RGB")
            ruta = f"rfmed/{x.name}"
        else:
            nom = Path(x).name
            im = Image.open(io.BytesIO(z.read(x))).convert("RGB")
            ruta = f"{fu}/{x}"
        o = original(nom)
        nombres.setdefault(o, ruta)
        for h in dhash_8(im):
            hs.append(h)
            quien.append(ruta)
        if (fu, o) not in vistos:
            vistos.add((fu, o))
            ims_orb.append(im.resize((224, 224)))
            quien_orb.append(ruta)
    print(f"Familia Smart medical: {len(nombres)} nombres originales, {len(quien) // 8} imágenes, "
          f"{len(ims_orb)} para ORB ({Counter(q.split('/')[0] for q in quien_orb)})", flush=True)
    return nombres, np.array(hs, dtype=np.uint64), quien, ims_orb, quien_orb


def filtro_orb_ref(oks, bims, bquien, motivo, k=5, umbral=25):
    """Igual que importar_roboflow.filtro_orb, pero contra una lista de imágenes de referencia cualquiera."""
    import cv2
    from ultralytics import YOLO
    modelo = YOLO(str(RAIZ / "yolo26s-cls.pt"))
    orb, bf = cv2.ORB_create(1000), cv2.BFMatcher(cv2.NORM_HAMMING)

    def emb(ims):
        out = []
        for i in range(0, len(ims), 128):
            e = modelo.embed([np.asarray(x)[:, :, ::-1] for x in ims[i:i + 128]], imgsz=224, verbose=False)
            out += [t.flatten().float().cpu().numpy() for t in e]
        m = np.stack(out)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def feats(im):
        g = cv2.equalizeHist(np.asarray(im.convert("L").resize((320, 320))))
        return orb.detectAndCompute(g, None)

    def inliers(fa, fb):
        (ka, da), (kb, db) = fa, fb
        if da is None or db is None or len(ka) < 10 or len(kb) < 10:
            return 0
        ms = [p[0] for p in bf.knnMatch(da, db, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(ms) < 8:
            return len(ms)
        pa = np.float32([ka[x.queryIdx].pt for x in ms])
        pb = np.float32([kb[x.trainIdx].pt for x in ms])
        _, mask = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=5)
        return int(mask.sum()) if mask is not None else 0

    BE, vi = [], []
    for i0 in range(0, len(bims), 512):  # 8 giros/espejos por referencia, por lotes
        lote, idx = [], []
        for i, im in enumerate(bims[i0:i0 + 512], start=i0):
            for kk in range(4):
                r = im.rotate(90 * kk, expand=True)
                lote += [r, r.transpose(Image.FLIP_LEFT_RIGHT)]
                idx += [i, i]
        BE.append(emb(lote).astype(np.float16))
        vi += idx
    BE, vi = np.concatenate(BE), np.array(vi)
    print(f"ORB {motivo}: {len(bims)} referencias, {len(oks)} candidatas", flush=True)
    cache = {}
    for i0 in range(0, len(oks), 256):
        lote = oks[i0:i0 + 256]
        ims = [abrir(c).resize((224, 224)) for c in lote]
        S = emb(ims).astype(np.float16) @ BE.T
        for c, im, s_ in zip(lote, ims, S):
            vec = []
            for j in np.argsort(-s_.astype(np.float32)):
                if vi[j] not in vec:
                    vec.append(int(vi[j]))
                if len(vec) == k:
                    break
            fc = feats(im)
            mejor = (0, None)
            for b in vec:
                if b not in cache:
                    cache[b] = feats(bims[b])
                mejor = max(mejor, (inliers(fc, cache[b]), b), key=lambda x: x[0])
            c["orb_fam"] = mejor[0]
            if mejor[0] >= umbral:
                c["motivo"], c["detalle"], c["dist"] = motivo, bquien[mejor[1]], mejor[0]
        print(f"  ORB {motivo} {i0 + len(lote)}/{len(oks)}", flush=True)


def abrir(c):
    return Image.open(io.BytesIO(c["zip"].read(c["ruta_zip"]))).convert("RGB")


def piezas(c, im):
    """Lo que se importaría de una candidata: la imagen, o un recorte por caja (fwd)."""
    if c["fuente"] in ("nx", "bmw"):
        cl, est = (MAPEO_BMW if c["fuente"] == "bmw" else MAPEO_NX)[c["categoria"]]
        return [dict(categoria=c["categoria"], clase=cl, estado=est, img=im, lado=min(im.size),
                     archivo=f"{c['fuente']}_{c['split']}_{Path(c['archivo']).stem}.jpg"
                     if c["fuente"] == "nx" else f"bmw_{c['categoria']}_{Path(c['archivo']).stem}.jpg")]
    W, H = im.size  # 640x640 (fwd: grupos sin estirar; pbw/mwn: estiradas, proporción no recuperable)
    out = []
    for i, (cat, cl, est, xc, yc, w, h) in enumerate(c["cajas"]):
        bw, bh = w * W, h * H
        x0, y0 = max(0, (xc - w / 2) * W - PAD * bw), max(0, (yc - h / 2) * H - PAD * bh)
        x1, y1 = min(W, (xc + w / 2) * W + PAD * bw), min(H, (yc + h / 2) * H + PAD * bh)
        rec = im.crop((int(x0), int(y0), int(x1), int(y1)))
        cat_a = re.sub(r"[^A-Za-z0-9]+", "-", cat).strip("-")
        out.append(dict(categoria=cat, clase=cl, estado=est, img=rec, lado=int(min(bw, bh)),
                        archivo=f"{c['fuente']}_{c['split']}_{c['original']}_{i:02d}_{cat_a}.jpg"))
    return out


def mosaico(items, ruta, titulo, n=48, lado=150, semilla=0):
    sel = random.Random(semilla).sample(items, min(n, len(items)))
    cols = 8
    filas = (len(sel) + cols - 1) // cols
    M = Image.new("RGB", (cols * lado, filas * lado + 24), "white")
    d = ImageDraw.Draw(M)
    d.text((4, 4), titulo, fill="black")
    for i, p in enumerate(sel):
        t = p["img"].copy()
        t.thumbnail((lado - 4, lado - 4))
        M.paste(t, ((i % cols) * lado + 2, (i // cols) * lado + 26))
    M.save(ruta, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importar", action="store_true", help="copia a externo/train (solo tras el OK del usuario)")
    ap.add_argument("--topes", nargs="+", default=[], metavar="FUENTE[:CLASE]=N",
                    help="máx. por fuente y clase, p. ej. fwd=300 bmw:peligrosos=400")
    ap.add_argument("--aprobar", nargs="+", default=[], metavar="FUENTE:CATEGORIA",
                    help="categorías 'dudosas' que el usuario aprueba (las demás dudosas no se importan)")
    ap.add_argument("--excluir", nargs="+", default=[], metavar="FUENTE:CATEGORIA",
                    help="categorías 'propuestas' que el usuario rechaza")
    ap.add_argument("--fuentes", nargs="+", choices=["fwd", "nx", "bmw"], default=["fwd", "nx", "bmw"],
                    help="fuentes candidatas a procesar (las no listadas no se leen ni se importan)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--sin-orb", action="store_true")
    a = ap.parse_args()

    desc = Counter()
    cands = ((cands_det("fwd", FUENTES["fwd"], desc) if "fwd" in a.fuentes else [])
             + (cands_nx(FUENTES["nx"], desc) if "nx" in a.fuentes else [])
             + (cands_bmw(FUENTES["bmw"], desc) if "bmw" in a.fuentes else []))
    print(f"Candidatas (imagen completa): {Counter(c['fuente'] for c in cands)}")
    print(f"Descartadas antes del dedup: {dict(desc)}")

    nombres_b, banco_b, quien_b = banco_benchmark()
    nombres_f, banco_f, quien_f, ims_f, quien_orb_f = banco_familia()
    banco_p, quien_p = banco_propios()
    hs_f, q_f = [], []
    for p in imagenes(EXTERNO / "_fuentes"):
        try:
            with Image.open(p) as im:
                hs_f.append(dhash_pil(im))
            q_f.append(f"_fuentes/{p.relative_to(EXTERNO / '_fuentes').as_posix()}")
        except OSError:
            pass
    banco_p = np.concatenate([banco_p, np.array(hs_f, dtype=np.uint64)])
    quien_p = quien_p + q_f
    print(f"Banco propio + _fuentes: {len(quien_p)} imágenes")

    # 1-3: imagen completa contra benchmark y propios
    for i, c in enumerate(cands):
        im = abrir(c)
        c["tam"] = "x".join(map(str, im.size))
        h = dhash_pil(im)
        c["motivo"], c["detalle"], c["dist"], c["orb"], c["orb_fam"] = "ok", "", "", "", ""
        if c["original"] in nombres_b:
            c["motivo"], c["detalle"] = "nombre_benchmark", nombres_b[c["original"]]
        elif (d := dist(banco_b, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_benchmark", quien_b[int(d.argmin())], int(d.min())
        elif c["original"] in nombres_f:
            c["motivo"], c["detalle"] = "nombre_familia_smartmed", nombres_f[c["original"]]
        elif (d := dist(banco_f, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_familia_smartmed", quien_f[int(d.argmin())], int(d.min())
        elif (d := dist(banco_p, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_propios", quien_p[int(d.argmin())], int(d.min())
        if i % 1000 == 0:
            print(f"  dHash {i}/{len(cands)}", flush=True)
    # 4: ORB contra el benchmark
    if not a.sin_orb:
        filtro_orb([c for c in cands if c["motivo"] == "ok"])
        filtro_orb_ref([c for c in cands if c["motivo"] == "ok" and c["fuente"] == "bmw"], ims_f, quien_orb_f,
                       "orb_familia_smartmed")
    del ims_f

    # 5-7: piezas (imagen o recortes)
    filas, aceptadas_h = [], []
    for c in cands:
        if c["motivo"] != "ok":
            cat = c.get("categoria") or "(imagen fwd)"
            filas.append(dict(fuente=c["fuente"], split=c["split"], categoria=cat, clase="", estado="",
                              origen=c["ruta_zip"], archivo="", tam=c["tam"], lado="", motivo=c["motivo"],
                              dist=c["dist"], orb=c["orb"], orb_fam=c["orb_fam"], detalle=c["detalle"], img=None))
            continue
        im = abrir(c)
        for i_p, p in enumerate(piezas(c, im)):
            f = dict(fuente=c["fuente"], split=c["split"], categoria=p["categoria"], clase=p["clase"],
                     estado=p["estado"], origen=c["ruta_zip"], archivo=p["archivo"], tam=c["tam"], lado=p["lado"],
                     motivo="ok", dist="", orb=c["orb"], orb_fam=c["orb_fam"], detalle="", img=None, _c=c, _i=i_p)
            if p["lado"] < MIN_LADO:
                f["motivo"] = "recorte_chico"
            else:
                hs = dhash_8(p["img"])
                if c["fuente"] in MAPEO_DET_TODAS and (d := dist(banco_p, hs[0])).min() <= UMBRAL:
                    f["motivo"], f["detalle"], f["dist"] = "dhash_propios_recorte", quien_p[int(d.argmin())], int(d.min())
                elif aceptadas_h and (dm := min(dist(np.array(aceptadas_h, dtype=np.uint64), x).min() for x in hs)) <= UMBRAL:
                    f["motivo"], f["dist"] = "dup_interno", int(dm)
                else:
                    aceptadas_h.append(hs[0])
                    f["img"] = p["img"].copy()
                    f["img"].thumbnail((150, 150))  # solo miniatura para el mosaico; al importar se relee
            filas.append(f)

    # aprobaciones y topes (solo cuentan para --importar; en dry-run se informa todo lo limpio)
    aprob = set(a.aprobar)
    excl = set(a.excluir)
    topes = {}
    for t in a.topes:
        k, v = t.split("=")
        topes[k] = int(v)
    rng = random.Random(a.semilla)
    grupos = defaultdict(list)
    for f in filas:
        if f["motivo"] != "ok":
            continue
        clave = f"{f['fuente']}:{f['categoria']}"
        if a.importar and ((f["estado"] == "dudosa" and clave not in aprob) or clave in excl):
            f["motivo"] = "no_aprobada"
            continue
        grupos[(f["fuente"], f["clase"])].append(f)
    for (fu, cl), g in grupos.items():
        tope = topes.get(f"{fu}:{cl}", topes.get(fu))
        if tope is not None and len(g) > tope:
            for f in rng.sample(g, len(g) - tope):
                f["motivo"] = "fuera_del_tope"

    INF.mkdir(exist_ok=True)
    campos = ["fuente", "split", "categoria", "clase", "estado", "origen", "archivo", "tam", "lado", "motivo",
              "dist", "orb", "orb_fam", "detalle"]
    with open(INF / "externo_candidatas_dedup.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    motivos = ["nombre_benchmark", "dhash_benchmark", "nombre_familia_smartmed", "dhash_familia_smartmed",
               "dhash_propios", "orb_benchmark", "orb_familia_smartmed", "recorte_chico",
               "dhash_propios_recorte", "dup_interno", "no_aprobada", "fuera_del_tope", "ok"]
    cont = defaultdict(Counter)
    for f in filas:
        k = (f["fuente"], f["categoria"], f["clase"] or "-", f["estado"] or "-")
        cont[k][f["motivo"]] += 1
        cont[k]["total"] += 1
    with open(INF / "externo_candidatas_conteos.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "categoria", "clase", "estado", "total"] + motivos)
        for k, v in sorted(cont.items()):
            w.writerow([*k, v["total"]] + [v[m] for m in motivos])
        for (fu, motivo), n in sorted(desc.items()):
            w.writerow([fu, motivo, "descartada", "-", n] + [""] * len(motivos))
    print(f"\n{'fuente/categoría -> clase (estado)':52s}{'total':>6s}" + "".join(f"{m[:10]:>11s}" for m in motivos))
    for k, v in sorted(cont.items()):
        print(f"{k[0] + '/' + k[1] + ' -> ' + k[2] + ' (' + k[3] + ')':52s}{v['total']:>6d}"
              + "".join(f"{v[m]:>11d}" for m in motivos))
    # de qué banco propio salen los dhash_propios (indicio de fork/derivado)
    orig_p = Counter((f["fuente"], re.sub(r"_[^/]*$", "", f["detalle"].split("/")[-1])[:12] if "externo" in f["detalle"]
                      else f["detalle"].split("/")[0]) for f in filas if f["motivo"].startswith("dhash_propios"))
    print("\nOrigen de los duplicados con datos propios:", dict(orig_p))
    limpias = Counter((f["fuente"], f["clase"], f["estado"]) for f in filas if f["motivo"] == "ok")
    print("Limpias por fuente/clase/estado:", dict(limpias))

    MOSAICOS.mkdir(exist_ok=True)
    porcat = defaultdict(list)
    for f in filas:
        if f["motivo"] == "ok":
            porcat[(f["fuente"], f["categoria"] if f["fuente"] != "fwd" else f["clase"])].append(f)
    for (fu, cat), its in porcat.items():
        mosaico(its, MOSAICOS / f"{fu}_{cat}.jpg", f"{fu} / {cat}: {len(its)} limpias (muestra de {min(48, len(its))})")
    print(f"Mosaicos en {MOSAICOS}")

    if not a.importar:
        print("\nDRY-RUN: no se tocó dataset/externo/train ni datos_externos.csv.")
        return

    # --- importación real (solo con el OK del usuario) ---
    prefijos = tuple(f + "_" for f in FUENTES)
    for cl in CLASES:
        for p in imagenes(EXTERNO / "train" / cl):
            if p.name.startswith(prefijos):
                p.unlink()  # solo archivos propios de este script (regenerables)
    agg = defaultdict(list)
    for f in filas:
        if f["motivo"] != "ok":
            continue
        dst = EXTERNO / "train" / f["clase"] / f["archivo"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        piezas(f["_c"], abrir(f["_c"]))[f["_i"]]["img"].save(dst, quality=95)
        agg[(f["fuente"], f["categoria"], f["clase"])].append(f["archivo"])
    hoy = date.today().isoformat()
    nuevas = []
    for (fu, cat, cl), arch in sorted(agg.items()):
        cfg = FUENTES[fu]
        nuevas.append({"fuente": f"{fu}: {cfg['nombre']}, deduplicado contra el benchmark NTP 900.058 (nombre, dHash<=4 "
                                 f"con 8 giros/espejos y ORB) y contra todos los datos propios", "url": cfg["url"],
                       "licencia": cfg["licencia"], "fecha": hoy, "categoria_origen": cat, "clase_destino": cl,
                       "n_imagenes": len(arch), "archivos": ";".join(sorted(arch))})
    viejas = list(csv.DictReader(open(REGISTRO, encoding="utf-8")))
    campos_r = list(viejas[0].keys())
    viejas = [r for r in viejas if not r["fuente"].startswith(tuple(f + ":" for f in FUENTES))]
    with open(REGISTRO, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos_r)
        w.writeheader()
        w.writerows(viejas + nuevas)
    print(f"Importadas {sum(len(v) for v in agg.values())} imágenes; {len(nuevas)} filas en datos_externos.csv")


if __name__ == "__main__":
    main()
