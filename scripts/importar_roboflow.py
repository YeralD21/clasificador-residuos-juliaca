"""Importa como datos EXTERNOS (regla 1 de CLAUDE.md) los splits train/valid de 3 datasets de Roboflow Universe que
el usuario aprobó (2026-11-08) para el experimento del benchmark NTP 900.058 del docente.

  python -I scripts/importar_roboflow.py                 # DRY-RUN (por defecto): deduplica, informa, no copia nada
  python -I scripts/importar_roboflow.py --importar      # copia a dataset/externo/train y registra en datos_externos.csv

Fuentes (zips "Folder Structure" en ~/Downloads; se leen en memoria, no se extraen ni se ejecuta nada de ellos):
  rfgc  : Garbage Collection v1 (amcsalazar/garbage-collection-celih), CC BY 4.0. NO es el garbage.v1i del benchmark.
  rfv4  : Garbage Classification v4 (rupiah-detector/garbage-classification-xxgap), CC BY 4.0. Todo en gris.
  rfmed : Smart medical waste segregation v3 (visalatchi-t-bme-student/smart-medical-waste-segregation), CC BY 4.0.
El split test de los 3 zips se excluye POR COMPLETO (no se lee). Las clases sin mapeo inequívoco se descartan.

Filtros, en orden (se anota el primero que elimina a cada candidata):
  (Si una versión de un original cae por 1-3, caen todas las versiones de ese original.)
  1. nombre_benchmark : el nombre original (lo que va antes de _jpg.rf.) coincide con el de alguna de las 1.455
                        imágenes del benchmark (pred_01_best_jhoel.csv).
  2. dhash_benchmark  : dHash <= 4 contra alguna de las 1.455 (a color, garbage.v1i reconstruido) o de las 238 en
                        gris del docente, probando las 8 rotaciones/espejos de cada imagen del benchmark
                        (Garbage Collection y el médico traen copias giradas/espejadas por augmentation de Roboflow).
  3. dhash_propios    : dHash <= 4 contra TODOS los splits de publico y juliaca, val_camara y lo que ya hay en
                        externo/train.
  4. copia_roboflow   : otra versión (augmentation de Roboflow) del mismo original ya aceptada: se queda una por
                        original (la primera por nombre; valid antes que train no importa, todo va a train).
  5. dup_interno      : dHash <= 4 contra una candidata nueva ya aceptada (dentro de la fuente o entre fuentes).
  5b. orb_benchmark   : (extra, más allá de nombre y dHash) la misma foto del benchmark girada, recortada o
                        deformada por la augmentation de Roboflow: >= 25 puntos ORB consistentes (RANSAC) con alguna
                        de sus 5 vecinas del benchmark por embedding (ver filtro_orb).
  6. fuera_del_tope   : pasa los filtros pero queda fuera del tope por clase y fuente (--tope; muestra aleatoria
                        con --semilla).
Informes: reportes/externo_roboflow_dedup.csv (cada candidata), reportes/externo_roboflow_conteos.csv y
reportes/externo_roboflow_benchmark.csv (eliminadas por benchmark, por dataset y clase).
Con --importar copia los elegidos a dataset/externo/train/<clase>/<prefijo>_<split>_<archivo> (píxeles sin cambios)
y reemplaza SOLO sus propias filas en datos_externos.csv. Nunca toca val, test ni dataset/juliaca.
"""
import argparse
import csv
import io
import random
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES, EXT, EXTERNO, JULIACA, PUBLICO, RAIZ, VAL_CAMARA, imagenes  # noqa: E402

DESCARGAS = Path.home() / "Downloads"
BENCH = RAIZ / "evaluacion_docente"
UMBRAL = 4
UMBRAL_ORB = 25
SPLITS_OK = ("train", "valid")
FUENTES = {
    "rfgc": dict(zip="Garbage Collection.v1i.folder.zip", nombre="Garbage Collection v1 (Roboflow Universe, amcsalazar)",
                 url="https://universe.roboflow.com/amcsalazar/garbage-collection-celih", licencia="CC BY 4.0",
                 mapeo={"cardboard": "papel_carton", "paper": "papel_carton", "glass": "vidrio", "metal": "metales",
                        "plastic": "plasticos"}),
    "rfv4": dict(zip="Garbage Classification.v4i.folder.zip",
                 nombre="Garbage Classification v4 (Roboflow Universe, rupiah-detector)",
                 url="https://universe.roboflow.com/rupiah-detector/garbage-classification-xxgap", licencia="CC BY 4.0",
                 mapeo={"Organik": "organico", "Anorganik": None, "Residu": None}),
    "rfmed": dict(zip="Smart medical waste segregation.v3i.folder.zip",
                  nombre="Smart medical waste segregation v3 (Roboflow Universe, visalatchi-t-bme-student)",
                  url="https://universe.roboflow.com/visalatchi-t-bme-student/smart-medical-waste-segregation",
                  licencia="CC BY 4.0", mapeo={"Infectious": "peligrosos", "General": None}),
}
INF = RAIZ / "reportes"
REGISTRO = RAIZ / "datos_externos.csv"


def original(nombre):
    """cardboard101_jpg.rf.<hash>.jpg -> cardboard101 ; sin .rf. -> nombre sin extensión."""
    base = nombre.split(".rf.")[0] if ".rf." in nombre else nombre.rsplit(".", 1)[0]
    for suf in ("_jpg", "_jpeg", "_png", "_JPG", "_JPEG", "_PNG", "_webp", "_bmp"):
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def dhash_pil(im, tam=8):
    g = np.asarray(im.convert("L").resize((tam + 1, tam), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash_8(im):
    """dHash de las 8 rotaciones/espejos (sobre la imagen en gris)."""
    g = im.convert("L")
    out = []
    for k in range(4):
        r = g.rotate(90 * k, expand=True)
        out += [dhash_pil(r), dhash_pil(r.transpose(Image.FLIP_LEFT_RIGHT))]
    return out


def dist(banco, h):
    if not len(banco):
        return np.array([99])
    return np.bitwise_count(banco ^ np.uint64(h))


def banco_benchmark():
    ref = list(csv.DictReader(open(BENCH / "datos_docente" / "pred_01_best_jhoel.csv", encoding="utf-8")))
    nombres = {}
    for f in ref:
        n = f["path"].replace("\\", "/").split("/")[-1]
        nombres[original(n)] = f["dataset"]
    hs, quien = [], []
    for carpeta in (BENCH / "imagenes" / "test_docente", BENCH / "imagenes" / "gris_docente_garbage_v1i"):
        fotos = [p for p in carpeta.rglob("*") if p.suffix.lower() in EXT]
        for p in fotos:
            with Image.open(p) as im:
                for h in dhash_8(im):
                    hs.append(h)
                    quien.append(p.name)
    n_img = len({q for q in quien})
    if len(ref) != 1455 or n_img < 1455:
        sys.exit(f"ERROR: el benchmark no está completo ({len(ref)} filas, {n_img} imágenes). Corre correr_todo.py.")
    return nombres, np.array(hs, dtype=np.uint64), quien


def banco_propios():
    hs, quien = [], []
    for base, etiqueta in ((PUBLICO, "publico"), (JULIACA, "juliaca"), (VAL_CAMARA, "val_camara"),
                           (EXTERNO / "train", "externo/train")):
        for p in imagenes(base):
            if p.name.startswith("._"):
                continue
            try:
                with Image.open(p) as im:
                    hs.append(dhash_pil(im))
                quien.append(f"{etiqueta}/{p.relative_to(base).as_posix()}")
            except OSError:
                print(f"  aviso: no se pudo leer {p}")
    print(f"Banco propio: {Counter(q.split('/')[0] for q in quien)}")
    return np.array(hs, dtype=np.uint64), quien


def filtro_orb(oks, k=5, umbral=UMBRAL_ORB):
    """Para cada candidata, busca sus k vecinas más parecidas del benchmark (embedding de yolo26s-cls preentrenado,
    con las 8 rotaciones/espejos del benchmark) y cuenta puntos ORB consistentes (RANSAC, transformación de
    similitud). >= umbral -> misma foto girada/recortada -> 'orb_benchmark'. Calibrado a ojo el 2026-11-08: los 16
    pares con >= 40 eran la misma foto; entre 15 y 40 hay misma foto, misma sesión y falsos positivos en gris."""
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

    rutas = [p for p in (BENCH / "imagenes" / "test_docente").rglob("*") if p.suffix.lower() in EXT
             and not p.parts[-3].startswith("garbage.v1i")]
    rutas += [p for p in (BENCH / "imagenes" / "_descargas" / "v1i" / "test").rglob("*") if p.suffix.lower() in EXT]
    bims = [Image.open(p).convert("RGB").resize((224, 224)) for p in rutas]
    vims, vi = [], []
    for i, im in enumerate(bims):
        for kk in range(4):
            r = im.rotate(90 * kk, expand=True)
            vims += [r, r.transpose(Image.FLIP_LEFT_RIGHT)]
            vi += [i, i]
    print(f"ORB: {len(rutas)} imágenes del benchmark, {len(oks)} candidatas", flush=True)
    BE, vi = emb(vims), np.array(vi)
    cache = {}
    for i0 in range(0, len(oks), 256):
        lote = oks[i0:i0 + 256]
        ims = [Image.open(io.BytesIO(c["zip"].read(c["ruta_zip"]))).convert("RGB").resize((224, 224)) for c in lote]
        S = emb(ims) @ BE.T
        for c, im, s in zip(lote, ims, S):
            vec = []
            for j in np.argsort(-s):
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
            c["orb"] = mejor[0]
            if mejor[0] >= umbral:
                c["motivo"], c["detalle"], c["dist"] = "orb_benchmark", rutas[mejor[1]].name, mejor[0]


def candidatas(fuente, cfg):
    zp = DESCARGAS / cfg["zip"]
    if not zp.is_file():
        sys.exit(f"ERROR: falta {zp}")
    z = zipfile.ZipFile(zp)
    res, descartadas = [], Counter()
    for n in sorted(z.namelist()):
        partes = n.split("/")
        if len(partes) != 3 or Path(n).suffix.lower() not in EXT or ".." in partes:
            continue
        split, cat, archivo = partes
        if split not in SPLITS_OK:          # test: excluido por completo
            continue
        if cat not in cfg["mapeo"]:
            sys.exit(f"ERROR: {fuente} tiene una clase sin decidir en el mapeo: {cat}")
        clase = cfg["mapeo"][cat]
        if clase is None:
            descartadas[(cat, split)] += 1
            continue
        res.append(dict(fuente=fuente, split=split, categoria=cat, archivo=archivo, clase=clase, zip=z, ruta_zip=n,
                        original=original(archivo)))
    return res, descartadas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importar", action="store_true", help="copia a externo/train (solo tras el OK del usuario)")
    ap.add_argument("--tope", type=int, default=None, help="máx. imágenes por clase y fuente (None = sin tope)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--fuentes", nargs="+", choices=sorted(FUENTES), default=sorted(FUENTES))
    ap.add_argument("--sin-orb", action="store_true", help="omite el filtro ORB (solo para pruebas)")
    a = ap.parse_args()

    nombres_b, banco_b, quien_b = banco_benchmark()
    banco_p, quien_p = banco_propios()
    cands, desc = [], {}
    for f, cfg in FUENTES.items():
        if f not in a.fuentes:
            continue
        c, d = candidatas(f, cfg)
        cands += c
        desc[f] = d
        print(f"{f}: {len(c)} candidatas mapeadas; descartadas por clase sin mapeo: {dict(d)}")

    # pasada 1: filtros contra el benchmark y contra lo propio, imagen por imagen
    for i, c in enumerate(cands):
        with Image.open(io.BytesIO(c["zip"].read(c["ruta_zip"]))) as im:
            c["tam"] = "x".join(map(str, im.size))
            c["h"] = h = dhash_pil(im)
        c["motivo"], c["detalle"], c["dist"] = "ok", "", ""
        if c["original"] in nombres_b:
            c["motivo"], c["detalle"] = "nombre_benchmark", nombres_b[c["original"]]
        elif (d := dist(banco_b, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_benchmark", quien_b[int(d.argmin())], int(d.min())
        elif (d := dist(banco_p, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_propios", quien_p[int(d.argmin())], int(d.min())
        if i % 2000 == 0:
            print(f"  {i}/{len(cands)}", flush=True)
    # pasada 2: si una versión de un original cae por el benchmark (o por lo propio), caen todas sus versiones
    malo = {}
    for c in cands:
        if c["motivo"] != "ok":
            k = (c["fuente"], c["original"])
            if malo.get(k) != "dhash_benchmark" and malo.get(k) != "nombre_benchmark":
                malo[k] = c["motivo"]
    for c in cands:
        k = (c["fuente"], c["original"])
        if c["motivo"] == "ok" and k in malo:
            c["motivo"] = malo[k]
            c["detalle"] = "otra versión del mismo original"
    # pasada 3: una versión por original y duplicados internos
    aceptadas_h, vistos_orig = [], set()
    for c in cands:
        if c["motivo"] != "ok":
            continue
        h = c["h"]
        clave = (c["fuente"], c["original"])
        if clave in vistos_orig:
            c["motivo"] = "copia_roboflow"
            continue
        d = dist(np.array(aceptadas_h, dtype=np.uint64), h) if aceptadas_h else np.array([99])
        if d.min() <= UMBRAL:
            c["motivo"], c["dist"] = "dup_interno", int(d.min())
            continue
        vistos_orig.add(clave)
        aceptadas_h.append(h)

    # pasada 4: duplicados girados/recortados del benchmark que el dHash no ve (ORB + RANSAC)
    if not a.sin_orb:
        filtro_orb([c for c in cands if c["motivo"] == "ok"])

    # tope por clase y fuente
    rng = random.Random(a.semilla)
    grupos = defaultdict(list)
    for c in cands:
        if c["motivo"] == "ok":
            grupos[(c["fuente"], c["clase"])].append(c)
    for k, g in grupos.items():
        if a.tope and len(g) > a.tope:
            fuera = rng.sample(g, len(g) - a.tope)
            for c in fuera:
                c["motivo"] = "fuera_del_tope"

    INF.mkdir(exist_ok=True)
    campos = ["fuente", "split", "categoria", "clase", "archivo", "original", "tam", "motivo", "dist", "orb", "detalle"]
    with open(INF / "externo_roboflow_dedup.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    motivos = ["nombre_benchmark", "dhash_benchmark", "dhash_propios", "copia_roboflow", "dup_interno",
               "orb_benchmark", "fuera_del_tope", "ok"]
    cont = defaultdict(Counter)
    for c in cands:
        cont[(c["fuente"], c["categoria"], c["clase"])][c["motivo"]] += 1
        cont[(c["fuente"], c["categoria"], c["clase"])]["candidatas"] += 1
    with open(INF / "externo_roboflow_conteos.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "categoria", "clase", "candidatas"] + motivos)
        for (f, cat, cl), k in sorted(cont.items()):
            w.writerow([f, cat, cl, k["candidatas"]] + [k[m] for m in motivos])
        for f, d in desc.items():
            for (cat, split), n in sorted(d.items()):
                w.writerow([f, f"{cat} ({split})", "descartada", n] + [""] * len(motivos))
    bench = Counter((c["fuente"], c["categoria"], c["split"], c["motivo"]) for c in cands
                    if c["motivo"] in ("nombre_benchmark", "dhash_benchmark", "orb_benchmark"))
    with open(INF / "externo_roboflow_benchmark.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "categoria", "split", "motivo", "n"])
        for k, n in sorted(bench.items()):
            w.writerow([*k, n])

    print(f"\n{'fuente/categoría -> clase':38s}{'cand':>6s}" + "".join(f"{m[:12]:>14s}" for m in motivos))
    for (f, cat, cl), k in sorted(cont.items()):
        print(f"{f + '/' + cat + ' -> ' + cl:38s}{k['candidatas']:>6d}" + "".join(f"{k[m]:>14d}" for m in motivos))
    print("\nEliminadas por el benchmark (fuente, categoría, split, motivo):")
    for k, n in sorted(bench.items()):
        print(f"  {k}: {n}")
    final = Counter(c["clase"] for c in cands if c["motivo"] == "ok")
    print("\nNuevas por clase:", dict(final))

    if not a.importar:
        print("\nDRY-RUN: no se tocó dataset/externo/train ni datos_externos.csv.")
        return

    # --- importación real (solo con el OK del usuario) ---
    prefijos = tuple(f + "_" for f in FUENTES)
    for cl in CLASES:
        for p in imagenes(EXTERNO / "train" / cl):
            if p.name.startswith(prefijos):
                p.unlink()          # solo archivos propios de este script (regenerables)
    filas = []
    for c in cands:
        if c["motivo"] != "ok":
            continue
        dst = EXTERNO / "train" / c["clase"] / f"{c['fuente']}_{c['split']}_{c['archivo']}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(c["zip"].read(c["ruta_zip"]))
    agg = Counter((c["fuente"], c["categoria"], c["clase"]) for c in cands if c["motivo"] == "ok")
    nombres = defaultdict(list)  # auditar.py exige cada archivo en la columna 'archivos'
    for c in cands:
        if c["motivo"] == "ok":
            nombres[(c["fuente"], c["categoria"], c["clase"])].append(f"{c['fuente']}_{c['split']}_{c['archivo']}")
    hoy = date.today().isoformat()
    for (f, cat, cl), n in sorted(agg.items()):
        cfg = FUENTES[f]
        filas.append({"fuente": f"{f}: {cfg['nombre']}, splits train+valid (test excluido), deduplicado contra el "
                                f"benchmark NTP 900.058 (nombre, dHash<=4 con 8 giros/espejos y ORB), tope 500", "url": cfg["url"],
                      "licencia": cfg["licencia"], "fecha": hoy, "categoria_origen": cat, "clase_destino": cl,
                      "n_imagenes": n, "archivos": ";".join(sorted(nombres[(f, cat, cl)]))})
    viejas = list(csv.DictReader(open(REGISTRO, encoding="utf-8")))
    campos_r = list(viejas[0].keys())
    viejas = [r for r in viejas if not r["fuente"].startswith(tuple(f + ":" for f in FUENTES))]
    with open(REGISTRO, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos_r)
        w.writeheader()
        w.writerows(viejas + filas)
    print(f"Importadas {sum(agg.values())} imágenes a dataset/externo/train; {len(filas)} filas en datos_externos.csv")


if __name__ == "__main__":
    main()
