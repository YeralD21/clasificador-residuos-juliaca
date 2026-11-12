"""Fase A (solo informe) de la ronda del 2026-11-11 para reforzar peligrosos: BDWaste, MaskNet, E-Waste y RecyBat24.

  python -I scripts/candidatas_peligrosos.py --bdwaste <dir> --recybat <dir>    # dedup, conteos, tope y mosaicos

No hay modo de importación: no escribe en dataset/ ni en datos_externos.csv. No ejecuta nada de los archivos leídos.

Fuentes (licencias verificadas el 2026-11-11):
  bdw : BDWaste (Mendeley 96g5pgfnfw v1, Akter et al. 2023), CC BY 4.0, clasificación. Fotos de móvil en Dhaka.
        Las subcarpetas se extraen antes (zips anidados) a <dir>/<Categoria>/. Glaves, Mask y Empty_medicine_packet
        -> peligrosos; cáscaras y restos de comida -> organico (Fish_ash: dudosa); Paper, Coffee_cup, Cane, Wire y el
        resto de 'Indigestive' se descartan (no pedidas o ambiguas). El autor dice que hay aumentos: los casi
        duplicados internos (dHash con giros y embedding) se quedan con una foto por grupo.
  msk : MaskNet (Peyvandi et al. 2022; Kaggle tenebris97/masknet = github.com/Tenebris97/MaskNet, licencia MIT del
        repositorio; Kaggle: ODbL). Mascarillas tiradas en la calle, 540x720, cajas en labels.csv -> recortes.
  ewa : E-Waste Classifications v9 (Roboflow student-esos5), CC BY 4.0, detección YOLO, estiradas a 640x640 sin forma
        de recuperar la proporción. Battery y Cable -> peligrosos (no hay clase de cargadores); Bulb y PCB se
        descartan (ambiguas). Solo splits train y valid, como en importar_roboflow.py.
  rcb : RecyBat24 (Zenodo 13126689, versión 1, CC BY 4.0), COCO. Esta versión SOLO trae imágenes aumentadas
        (rotated_<ang>_bright_<b>); la versión sin aumentar está en versiones posteriores con CC BY-NC-ND 4.0.
        Se usa la única variante equivalente al original: rotated_90_bright_1.0 (giro exacto de 90 grados, sin
        interpolación ni esquinas negras, brillo original). Existe para pouch y cylindric; prismatic no la tiene.
        <dir> = carpeta con coco/{train,val}/ ya extraída solo con esas variantes. Tope 100 (fondo oscuro uniforme).
  Surgical Waste Dataset (Mendeley rnhz3fvbj2): descartada sin procesar (ver EXPERIMENTOS.md).

Filtros, en orden (se anota el primero que elimina). Sobre la IMAGEN COMPLETA:
  nombre_benchmark (solo nombres distintivos: no '12.jpg') | dhash_benchmark (8 giros/espejos, 1.455 + 238 gris) |
  nombre_familia | dhash_familia (Smart medical completo, pbw, mwn, rfmed_ y Medical-Waste-4.0 A y B) |
  dhash_propios (publico, juliaca, val_camara, externo/train, externo/_fuentes) | orb_benchmark | orb_familia
  (solo candidatas de peligrosos). Sobre lo que se importaría (imagen o recorte con margen del 10 %):
  recorte_chico (lado menor de la caja < 64 px) | dhash_propios_recorte | dup_interno (dHash <= 4 con 8 giros contra
  lo aceptado de la misma fuente) | consecutiva (embedding yolo26s-cls, coseno >= 0.95 con una aceptada de la misma
  fuente y categoría) | fuera_del_tope (150 por fuente y clase; rcb 100; reparto por categoría).
Informes: reportes/externo_candidatas4_{dedup,conteos}.csv y reportes/mosaicos_candidatas4/.
"""
import argparse
import csv
import io
import json
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import EXT, EXTERNO, RAIZ, imagenes  # noqa: E402
from importar_candidatas import banco_familia, filtro_orb_ref, mosaico  # noqa: E402
from importar_roboflow import UMBRAL, banco_benchmark, banco_propios, dhash_8, dhash_pil, dist, filtro_orb, original  # noqa: E402

DESCARGAS = Path.home() / "Downloads"
DATA = Path(r"D:\Proyecto-Residuos-ML\DATA-NEW")
INF = RAIZ / "reportes"
MOSAICOS = INF / "mosaicos_candidatas4"
MIN_LADO = 64
PAD = 0.10
COS_CONSEC = 0.95
TOPE = {"bdw": 150, "msk": 150, "ewa": 150, "rcb": 100}
ZIP_MW4 = DESCARGAS / "Medical-Waste-4.0-Dataset-Dataset.zip"
ZIP_MSK = DATA / "archive (3).zip"
ZIP_EWA = DATA / "E-Waste Classifications.v9i.yolov8.zip"

ORG = ("organico", "propuesta")
MAPEO = {
    "bdw": {"Glaves": ("peligrosos", "propuesta"), "Mask": ("peligrosos", "propuesta"),
            "Empty_medicine_packet": ("peligrosos", "propuesta"),
            **{k: ORG for k in ("Sugarcane_husk", "Potato_Peel", "Rice", "Mango_Peel", "Shell_of_Malta",
                                "Lemon_Peel", "Banana_peel", "Egg_shell")},
            "Fish_ash": ("organico", "dudosa")},
    "msk": {"Mask": ("peligrosos", "propuesta")},
    "ewa": {"Battery": ("peligrosos", "propuesta"), "Cable": ("peligrosos", "propuesta")},
    "rcb": {"pouch": ("peligrosos", "propuesta"), "cylindric": ("peligrosos", "propuesta")},
}


class Disco:
    def read(self, ruta):
        return Path(ruta).read_bytes()


DISCO = Disco()


def abrir(c):
    im = Image.open(io.BytesIO(c["zip"].read(c["ruta_zip"])))
    im = ImageOps.exif_transpose(im) if c["fuente"] == "bdw" else im
    im = im.convert("RGB")
    if max(im.size) > 1600:  # fotos de móvil de BDWaste (hasta 4000 px): se reducen para el análisis
        im.thumbnail((1600, 1600))
    return im


def cands_bdw(base, desc):
    res = []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        fotos = sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)
        if d.name not in MAPEO["bdw"]:
            desc[("bdw", f"{d.name} (no pedida o ambigua)")] += len(fotos)
            continue
        res += [dict(fuente="bdw", zip=DISCO, ruta_zip=str(p), original=p.stem, split="-", categoria=d.name,
                     archivo=p.name, cajas=None) for p in fotos]
    return res


def cands_msk(desc):
    z = zipfile.ZipFile(ZIP_MSK)
    cajas = defaultdict(list)
    for f in csv.DictReader(io.StringIO(z.read("labels.csv").decode("utf-8"))):
        cajas[f["filename"]].append((f["class"], *(float(f[k]) for k in ("xmin", "ymin", "xmax", "ymax"))))
    res = []
    for x in sorted(z.namelist()):
        if not x.lower().endswith(".jpg"):
            continue
        n = x.split("/")[-1]
        cs = [c for c in cajas.get(n, []) if c[0] in MAPEO["msk"]]
        if not cs:
            desc[("msk", "sin caja de Mask")] += 1
            continue
        res.append(dict(fuente="msk", zip=z, ruta_zip=x, original=Path(n).stem, split="-", categoria="Mask",
                        archivo=n, cajas=cs, norm=False))
    return res


def cands_ewa(desc):
    z = zipfile.ZipFile(ZIP_EWA)
    nombres = ["Battery", "Bulb", "Cable", "PCB"]
    res = []
    for x in sorted(z.namelist()):
        p = x.split("/")
        if len(p) != 3 or p[1] != "images" or Path(x).suffix.lower() not in EXT:
            continue
        if p[0] not in ("train", "valid"):
            desc[("ewa", f"split {p[0]}")] += 1
            continue
        lab = f"{p[0]}/labels/{Path(p[2]).stem}.txt"
        cs = []
        for linea in z.read(lab).decode().splitlines():
            v = linea.split()
            if len(v) != 5:
                continue
            cat = nombres[int(v[0])]
            if cat not in MAPEO["ewa"]:
                desc[("ewa", f"caja {cat} (ambigua)")] += 1
                continue
            xc, yc, w, h = map(float, v[1:])
            cs.append((cat, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
        if not cs:
            desc[("ewa", "imagen sin cajas Battery/Cable")] += 1
            continue
        # categoría de la imagen = la de su caja más grande (para el reparto del tope se usa la del recorte)
        res.append(dict(fuente="ewa", zip=z, ruta_zip=x, original=original(p[2]), split=p[0],
                        categoria=Counter(c[0] for c in cs).most_common(1)[0][0], archivo=p[2], cajas=cs, norm=True))
    return res


def cands_rcb(base, desc):
    res = []
    for s in ("train", "val"):
        d = json.load(open(base / "coco" / s / "annotations.json", encoding="utf-8"))
        cat = {c["id"]: c["name"] for c in d["categories"]}
        anns = defaultdict(list)
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            anns[a["image_id"]].append((cat[a["category_id"]], x, y, x + w, y + h))
        for im in d["images"]:
            if "_rotated_90_bright_1.0." not in im["file_name"]:
                continue
            ruta = base / "coco" / s / im["file_name"]
            cs = [c for c in anns.get(im["id"], []) if c[0] in MAPEO["rcb"]]
            if not ruta.exists() or not cs:
                desc[("rcb", "sin archivo o sin caja")] += 1
                continue
            res.append(dict(fuente="rcb", zip=DISCO, ruta_zip=str(ruta), original=im["file_name"].split("_rotated")[0],
                            split=s, categoria=cs[0][0], archivo=im["file_name"], cajas=cs, norm=False))
    return res


def piezas(c, im):
    if c["cajas"] is None:
        cl, est = MAPEO[c["fuente"]][c["categoria"]]
        return [dict(categoria=c["categoria"], clase=cl, estado=est, img=im, lado=min(im.size),
                     archivo=f"{c['fuente']}_{c['categoria']}_{Path(c['archivo']).stem}.jpg")]
    W, H = im.size
    out = []
    for i, (cat, x0, y0, x1, y1) in enumerate(c["cajas"]):
        if c.get("norm"):
            x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
        bw, bh = x1 - x0, y1 - y0
        rec = im.crop((int(max(0, x0 - PAD * bw)), int(max(0, y0 - PAD * bh)),
                       int(min(W, x1 + PAD * bw)), int(min(H, y1 + PAD * bh))))
        cl, est = MAPEO[c["fuente"]][cat]
        out.append(dict(categoria=cat, clase=cl, estado=est, img=rec, lado=int(min(bw, bh)),
                        archivo=f"{c['fuente']}_{c['split']}_{re.sub(r'[^A-Za-z0-9]+', '-', c['original'])}"
                                f"_{i:02d}_{cat}.jpg".replace("_-_", "_")))
    return out


def banco_mw4():
    z = zipfile.ZipFile(ZIP_MW4)
    nombres, hs, quien, ims = {}, [], [], []
    for x in z.namelist():
        if Path(x).suffix.lower() not in (".jpg", ".jpeg"):
            continue
        im = Image.open(io.BytesIO(z.read(x))).convert("RGB")
        im.thumbnail((640, 640))
        nombres.setdefault(original(Path(x).name), f"mw4/{x}")
        for h in dhash_8(im):
            hs.append(h)
            quien.append(f"mw4/{x}")
        ims.append(im.resize((224, 224)))
    print(f"Familia MW4: {len(ims)} imágenes RGB", flush=True)
    return nombres, hs, quien, ims


def distintivo(nombre):
    return bool(re.search(r"[A-Za-z]", nombre)) and len(nombre) >= 6


def embeddings(ims):
    from ultralytics import YOLO
    modelo = YOLO(str(RAIZ / "yolo26s-cls.pt"))
    out = []
    for i in range(0, len(ims), 128):
        e = modelo.embed([np.asarray(x)[:, :, ::-1] for x in ims[i:i + 128]], imgsz=224, verbose=False)
        out += [t.flatten().float().cpu().numpy() for t in e]
    m = np.stack(out)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdwaste", type=Path, required=True)
    ap.add_argument("--recybat", type=Path, required=True)
    a = ap.parse_args()

    desc = Counter()
    cands = cands_bdw(a.bdwaste, desc) + cands_msk(desc) + cands_ewa(desc) + cands_rcb(a.recybat, desc)
    print(f"Candidatas (imágenes): {Counter((c['fuente'], c['categoria']) for c in cands)}")
    print(f"Descartadas antes del dedup: {dict(desc)}", flush=True)

    nombres_b, banco_b, quien_b = banco_benchmark()
    nombres_f, banco_f, quien_f, ims_f, quien_orb_f = banco_familia()
    n4, h4, q4, i4 = banco_mw4()
    nombres_f = {**n4, **nombres_f}
    banco_f = np.concatenate([banco_f, np.array(h4, dtype=np.uint64)])
    quien_f = quien_f + q4
    ims_f, quien_orb_f = ims_f + i4, quien_orb_f + [q for q in q4[::8]]
    banco_p, quien_p = banco_propios()
    hs_x, q_x = [], []
    for p in imagenes(EXTERNO / "_fuentes"):
        try:
            with Image.open(p) as im:
                hs_x.append(dhash_pil(im))
            q_x.append(f"_fuentes/{p.relative_to(EXTERNO / '_fuentes').as_posix()}")
        except OSError:
            pass
    banco_p = np.concatenate([banco_p, np.array(hs_x, dtype=np.uint64)])
    quien_p = quien_p + q_x

    for i, c in enumerate(cands):
        im = abrir(c)
        c["tam"] = "x".join(map(str, im.size))
        h = dhash_pil(im)
        o = c["original"]
        c.update(motivo="ok", detalle="", dist="", orb="", orb_fam="")
        if distintivo(o) and o in nombres_b:
            c["motivo"], c["detalle"] = "nombre_benchmark", nombres_b[o]
        elif (d := dist(banco_b, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_benchmark", quien_b[int(d.argmin())], int(d.min())
        elif distintivo(o) and o in nombres_f:
            c["motivo"], c["detalle"] = "nombre_familia", nombres_f[o]
        elif (d := dist(banco_f, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_familia", quien_f[int(d.argmin())], int(d.min())
        elif (d := dist(banco_p, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_propios", quien_p[int(d.argmin())], int(d.min())
        if i % 500 == 0:
            print(f"  dHash {i}/{len(cands)}", flush=True)

    # filtro_orb lee los bytes con c['zip'].read: para bdw (fotos grandes) se le pasa una copia reducida
    reducidas = {}

    class Reducida:
        def read(self, ruta):
            return reducidas[ruta]

    for c in cands:
        if c["motivo"] == "ok" and c["fuente"] == "bdw":
            b = io.BytesIO()
            abrir(c).save(b, "JPEG", quality=92)
            reducidas[c["ruta_zip"]] = b.getvalue()
            c["zip_orig"], c["zip"] = c["zip"], Reducida()
    filtro_orb([c for c in cands if c["motivo"] == "ok"])
    filtro_orb_ref([c for c in cands if c["motivo"] == "ok" and any(
        MAPEO[c["fuente"]].get(k[0] if isinstance(k, tuple) else k, ("",))[0] == "peligrosos"
        for k in (c["cajas"] or [c["categoria"]]))], ims_f, quien_orb_f, "orb_familia")
    del ims_f
    for c in cands:
        if "zip_orig" in c:
            c["zip"] = c.pop("zip_orig")
    reducidas.clear()

    # piezas (imagen o recortes) de las imágenes que pasaron
    pzs = []
    for c in cands:
        if c["motivo"] != "ok":
            for p in piezas(c, Image.new("RGB", (1, 1))) if c["cajas"] is None else [
                    dict(categoria=k[0], clase=MAPEO[c["fuente"]][k[0]][0], estado=MAPEO[c["fuente"]][k[0]][1],
                         lado="", archivo="") for k in c["cajas"]]:
                p.pop("img", None)
                pzs.append({**{k: c[k] for k in ("fuente", "split", "ruta_zip", "tam", "motivo", "dist", "orb",
                                                 "orb_fam", "detalle")}, **p})
            continue
        im = abrir(c)
        for p in piezas(c, im):
            p.update({k: c[k] for k in ("fuente", "split", "ruta_zip", "tam", "orb", "orb_fam")},
                     motivo="ok", dist="", detalle="")
            if p["lado"] < MIN_LADO:
                p["motivo"] = "recorte_chico"
            elif c["cajas"] is not None and (d := dist(banco_p, dhash_pil(p["img"]))).min() <= UMBRAL:
                p["motivo"], p["detalle"], p["dist"] = "dhash_propios_recorte", quien_p[int(d.argmin())], int(d.min())
            if p["motivo"] == "ok":
                p["img"] = p["img"].copy()
                p["img"].thumbnail((448, 448))
            else:
                p.pop("img")
            pzs.append(p)
    print(f"Piezas: {Counter((p['fuente'], p['motivo']) for p in pzs)}", flush=True)

    vivos = [p for p in pzs if p["motivo"] == "ok"]
    E = embeddings([p["img"].resize((224, 224)) for p in vivos])
    acept_h = defaultdict(list)  # fuente -> [dHash]
    acept_e = defaultdict(list)  # (fuente, categoria) -> [emb]
    for p, e in zip(vivos, E):
        hs = dhash_8(p["img"])
        banco = np.array(acept_h[p["fuente"]], dtype=np.uint64)
        dmin = min((int(dist(banco, x).min()) for x in hs), default=99) if len(banco) else 99
        k = (p["fuente"], p["categoria"])
        cmax = max((float(x @ e) for x in acept_e[k]), default=0.0)
        if dmin <= UMBRAL:
            p["motivo"], p["dist"] = "dup_interno", dmin
        elif cmax >= COS_CONSEC:
            p["motivo"], p["dist"] = "consecutiva", round(cmax, 3)
        else:
            acept_h[p["fuente"]].append(hs[0])
            acept_e[k].append(e)

    rng = random.Random(42)
    grupos = defaultdict(lambda: defaultdict(list))
    for p in pzs:
        if p["motivo"] == "ok":
            grupos[(p["fuente"], p["clase"])][p["categoria"]].append(p)
    for (fu, cl), porcat in grupos.items():
        for v in porcat.values():
            rng.shuffle(v)
        colas = [v for _, v in sorted(porcat.items())]
        elegidas = []
        while len(elegidas) < TOPE[fu] and any(colas):
            for v in colas:
                if v and len(elegidas) < TOPE[fu]:
                    elegidas.append(v.pop())
        for v in colas:
            for p in v:
                p["motivo"] = "fuera_del_tope"

    INF.mkdir(exist_ok=True)
    campos = ["fuente", "split", "categoria", "clase", "estado", "ruta_zip", "archivo", "tam", "lado", "motivo",
              "dist", "orb", "orb_fam", "detalle"]
    with open(INF / "externo_candidatas4_dedup.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(pzs)
    motivos = ["nombre_benchmark", "dhash_benchmark", "nombre_familia", "dhash_familia", "dhash_propios",
               "orb_benchmark", "orb_familia", "recorte_chico", "dhash_propios_recorte", "dup_interno", "consecutiva",
               "fuera_del_tope", "ok"]
    cont = defaultdict(Counter)
    for p in pzs:
        k = (p["fuente"], p["categoria"], p["clase"], p["estado"])
        cont[k][p["motivo"]] += 1
        cont[k]["total"] += 1
    with open(INF / "externo_candidatas4_conteos.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "categoria", "clase", "estado", "total"] + motivos)
        for k, v in sorted(cont.items()):
            w.writerow([*k, v["total"]] + [v[m] for m in motivos])
        for (fu, motivo), n in sorted(desc.items()):
            w.writerow([fu, motivo, "descartada", "-", n] + [""] * len(motivos))
    print(f"\n{'fuente/categoría -> clase':40s}{'total':>6s}" + "".join(f"{m[:10]:>11s}" for m in motivos))
    for k, v in sorted(cont.items()):
        print(f"{k[0] + '/' + k[1] + ' -> ' + k[2]:40s}{v['total']:>6d}" + "".join(f"{v[m]:>11d}" for m in motivos))
    print("\nPropuestas (tras tope):", dict(Counter((p["fuente"], p["clase"]) for p in pzs if p["motivo"] == "ok")))
    print("Limpias antes del tope:", dict(Counter((p["fuente"], p["clase"]) for p in pzs
                                              if p["motivo"] in ("ok", "fuera_del_tope"))))

    MOSAICOS.mkdir(exist_ok=True)
    porcat = defaultdict(list)
    for p in pzs:
        if p["motivo"] in ("ok", "fuera_del_tope"):
            porcat[(p["fuente"], p["categoria"], p["motivo"] == "ok")].append(p)
    for (fu, cat, prop), its in sorted(porcat.items()):
        nom = "propuestas" if prop else "fuera_del_tope"
        mosaico(its, MOSAICOS / f"{fu}_{cat}_{nom}.jpg",
                f"{fu} / {cat} -> {its[0]['clase']}: {len(its)} {nom} (muestra de {min(48, len(its))})")
    print(f"Mosaicos en {MOSAICOS}\nSOLO INFORME: no se tocó dataset/ ni datos_externos.csv.")


if __name__ == "__main__":
    main()
