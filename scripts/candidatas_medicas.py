"""Fase A (solo informe) de la ronda de candidatas del 2026-11-11: Medical-Waste-4.0 y Gloves_cls (regla 1 de CLAUDE.md).

  python -I scripts/candidatas_medicas.py            # dedup, conteos y mosaicos; NO importa nada

No hay modo de importación: este script no escribe en dataset/ ni en datos_externos.csv. Los zips se leen en memoria
desde ~/Downloads (no se extraen ni se ejecuta nada de ellos). MARIDA se descarta sin procesar (parches Sentinel-2
.tif de 256x256 con máscaras de desechos marinos, no fotos de objetos).

Fuentes:
  mw4 : Medical-Waste-4.0-Dataset (davidemoronicnr, GitHub). Dataset A = Zenodo 10.5281/zenodo.7643417, CC BY 4.0.
        Dataset B (adquirido otro día) solo está en GitHub, repositorio sin archivo de licencia -> licencia no clara.
        Cada muestra = RGB 1920x1080 (.jpeg) + par estéreo en gris (_l/_r .png): solo se usa el RGB.
        Es el ORIGEN de la parte médica del benchmark: los nombres son timestamps (1671...) y aparecen en el test del
        docente, en 'Smart medical waste segregation', en mwn y en rfmed_ (ver informe).
  glv : Gloves_cls v1 (Roboflow Universe, workshop-l16o9/gloves_cls), CC BY 4.0, clasificación. Recortes de un solo
        video (gloves_bench_1, frames f70-f1730), estirados a 640x640.

Filtros sobre la imagen (se anota el primero que elimina):
  nombre_benchmark (nombre original o timestamp de 13 dígitos)  dhash_benchmark (8 giros/espejos, 1.455 + 238 gris)
  nombre_familia_smartmed  dhash_familia_smartmed (zip Smart medical completo, pbw, mwn, rfmed_)  dhash_propios
  (publico, juliaca, val_camara, externo/train, externo/_fuentes)  orb_benchmark  orb_familia_smartmed
  resolucion_baja (glv: resolución efectiva < 64 px)  dup_interno (dHash <= 4, 8 giros/espejos)
  consecutiva (embedding yolo26s-cls, coseno >= COS_CONSEC con una ya aceptada de la misma fuente y categoría: fotos
  consecutivas del mismo objeto; se queda la primera por orden de nombre)
Informes: reportes/externo_candidatas2_{dedup,conteos}.csv y reportes/mosaicos_candidatas2/.
"""
import csv
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import EXT, EXTERNO, RAIZ, imagenes  # noqa: E402
from importar_candidatas import banco_familia, filtro_orb_ref, mosaico  # noqa: E402
from importar_roboflow import UMBRAL, banco_benchmark, banco_propios, dhash_8, dhash_pil, dist, filtro_orb, original  # noqa: E402

DESCARGAS = Path.home() / "Downloads"
INF = RAIZ / "reportes"
MOSAICOS = INF / "mosaicos_candidatas2"
COS_CONSEC = 0.95
MIN_LADO = 64
TS = re.compile(r"(1\d{12})")

FUENTES = {
    "mw4": dict(zip="Medical-Waste-4.0-Dataset-Dataset.zip"),
    "glv": dict(zip="Gloves_cls.v1i.folder.zip"),
}
# categoría -> (clase, estado); lo que no está se descarta como ambiguo o fuera de la lista del usuario
MAPEO = {
    "mw4": {"gauze": ("peligrosos", "propuesta"), "gauzes": ("peligrosos", "propuesta"),
            **{f"glove_{a}_{b}": ("peligrosos", "propuesta") for a in ("pair", "single")
               for b in ("latex", "nitrile", "surgery")}},
    "glv": {"Blue_glove": ("peligrosos", "dudosa")},
}


def cands_mw4(desc):
    z = zipfile.ZipFile(DESCARGAS / FUENTES["mw4"]["zip"])
    res = []
    for x in sorted(z.namelist()):
        p = Path(x)
        if p.suffix.lower() not in EXT:
            continue
        ds, cat = p.parent.parent.name, p.parent.name
        if p.suffix.lower() == ".png":
            desc[("mw4", "estereo_gris_png")] += 1
            continue
        if cat not in MAPEO["mw4"]:
            desc[("mw4", f"{ds[-1]}/{cat} (ambigua o fuera de la lista)")] += 1
            continue
        res.append(dict(fuente="mw4", zip=z, ruta_zip=x, original=p.stem, split=f"dataset_{ds[-1]}",
                        categoria=cat, archivo=p.name))
    return res


def cands_glv(desc):
    z = zipfile.ZipFile(DESCARGAS / FUENTES["glv"]["zip"])
    res = []
    for x in sorted(z.namelist()):
        p = x.split("/")
        if len(p) != 3 or Path(x).suffix.lower() not in EXT:
            continue
        split, cat, arch = p
        if split == "test":
            desc[("glv", f"split test ({cat})")] += 1
            continue
        if cat not in MAPEO["glv"]:
            desc[("glv", f"{cat} (manos o fondo)")] += 1
            continue
        res.append(dict(fuente="glv", zip=z, ruta_zip=x, original=original(arch), split=split, categoria=cat,
                        archivo=arch))
    return res


def abrir(c):
    return Image.open(io.BytesIO(c["zip"].read(c["ruta_zip"]))).convert("RGB")


def resolucion_efectiva(im, umbral_mse=25.0):
    """Lado más chico s tal que bajar a s x s y volver a subir casi no cambia la imagen (imágenes ampliadas)."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    for s in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256):
        b = np.asarray(im.convert("L").resize((s, s), Image.BOX).resize(im.size, Image.BILINEAR), dtype=np.float32)
        if ((a - b) ** 2).mean() < umbral_mse:
            return s
    return min(im.size)


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
    desc = Counter()
    cands = cands_mw4(desc) + cands_glv(desc)
    print(f"Candidatas: {Counter((c['fuente'], c['split']) for c in cands)}")
    print(f"Descartadas antes del dedup: {dict(desc)}")

    nombres_b, banco_b, quien_b = banco_benchmark()
    nombres_f, banco_f, quien_f, ims_f, quien_orb_f = banco_familia()
    ts_b = {m.group(1): v for k, v in nombres_b.items() if (m := TS.search(k))}
    ts_f = {m.group(1): v for k, v in nombres_f.items() if (m := TS.search(k))}
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

    for i, c in enumerate(cands):
        im = abrir(c)
        c["tam"] = "x".join(map(str, im.size))
        c["res_ef"] = resolucion_efectiva(im) if c["fuente"] == "glv" else min(im.size)
        h = dhash_pil(im)
        ts = TS.search(c["original"])
        ts = ts.group(1) if ts else None
        c.update(motivo="ok", detalle="", dist="", orb="", orb_fam="")
        if c["original"] in nombres_b or (ts and ts in ts_b):
            c["motivo"], c["detalle"] = "nombre_benchmark", nombres_b.get(c["original"]) or ts_b[ts]
        elif (d := dist(banco_b, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_benchmark", quien_b[int(d.argmin())], int(d.min())
        elif c["original"] in nombres_f or (ts and ts in ts_f):
            c["motivo"], c["detalle"] = "nombre_familia_smartmed", nombres_f.get(c["original"]) or ts_f[ts]
        elif (d := dist(banco_f, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_familia_smartmed", quien_f[int(d.argmin())], int(d.min())
        elif (d := dist(banco_p, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_propios", quien_p[int(d.argmin())], int(d.min())
        if i % 500 == 0:
            print(f"  dHash {i}/{len(cands)}", flush=True)

    filtro_orb([c for c in cands if c["motivo"] == "ok"])
    filtro_orb_ref([c for c in cands if c["motivo"] == "ok"], ims_f, quien_orb_f, "orb_familia_smartmed")
    del ims_f

    for c in cands:
        if c["motivo"] == "ok" and c["res_ef"] < MIN_LADO:
            c["motivo"], c["detalle"] = "resolucion_baja", f"efectiva ~{c['res_ef']} px"

    # casi duplicados internos: dHash (8 giros) y fotos consecutivas (embedding), por fuente y categoría
    vivos = [c for c in cands if c["motivo"] == "ok"]
    E = embeddings([abrir(c).resize((224, 224)) for c in vivos]) if vivos else np.zeros((0, 1))
    acept = defaultdict(list)  # (fuente, split, categoria) -> [(hashes, emb)]
    for c, e in zip(vivos, E):
        k = (c["fuente"], c["split"], c["categoria"])
        hs = dhash_8(abrir(c))
        dmin = min((int(dist(np.array([a[0][0]], dtype=np.uint64), x).min()) for a in acept[k] for x in hs),
                   default=99)
        cmax = max((float(a[1] @ e) for a in acept[k]), default=0.0)
        if dmin <= UMBRAL:
            c["motivo"], c["dist"] = "dup_interno", dmin
        elif cmax >= COS_CONSEC:
            c["motivo"], c["dist"] = "consecutiva", round(cmax, 3)
        else:
            acept[k].append((hs, e))

    INF.mkdir(exist_ok=True)
    campos = ["fuente", "split", "categoria", "clase", "estado", "origen", "tam", "res_ef", "motivo", "dist", "orb",
              "orb_fam", "detalle"]
    for c in cands:
        c["clase"], c["estado"] = MAPEO[c["fuente"]][c["categoria"]]
        c["origen"] = c["ruta_zip"]
    with open(INF / "externo_candidatas2_dedup.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    motivos = ["nombre_benchmark", "dhash_benchmark", "nombre_familia_smartmed", "dhash_familia_smartmed",
               "dhash_propios", "orb_benchmark", "orb_familia_smartmed", "resolucion_baja", "dup_interno",
               "consecutiva", "ok"]
    cont = defaultdict(Counter)
    for c in cands:
        k = (c["fuente"], c["split"], c["categoria"], c["clase"], c["estado"])
        cont[k][c["motivo"]] += 1
        cont[k]["total"] += 1
    with open(INF / "externo_candidatas2_conteos.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "split", "categoria", "clase", "estado", "total"] + motivos)
        for k, v in sorted(cont.items()):
            w.writerow([*k, v["total"]] + [v[m] for m in motivos])
        for (fu, motivo), n in sorted(desc.items()):
            w.writerow([fu, "-", motivo, "descartada", "-", n] + [""] * len(motivos))
    print(f"\n{'fuente/split/categoría':42s}{'total':>6s}" + "".join(f"{m[:11]:>12s}" for m in motivos))
    for k, v in sorted(cont.items()):
        print(f"{k[0] + '/' + k[1] + '/' + k[2]:42s}{v['total']:>6d}" + "".join(f"{v[m]:>12d}" for m in motivos))
    print("\nResolución efectiva glv:", Counter(c["res_ef"] for c in cands if c["fuente"] == "glv"))

    MOSAICOS.mkdir(exist_ok=True)
    porcat = defaultdict(list)
    for c in cands:
        if c["motivo"] == "ok":
            im = abrir(c)
            im.thumbnail((150, 150))
            porcat[(c["fuente"], c["split"], c["categoria"])].append(dict(img=im))
    for (fu, sp, cat), its in porcat.items():
        mosaico(its, MOSAICOS / f"{fu}_{sp}_{cat}_limpias.jpg",
                f"{fu} / {sp} / {cat}: {len(its)} limpias (muestra de {min(48, len(its))})")
    print(f"Mosaicos en {MOSAICOS}\nSOLO INFORME: no se tocó dataset/ ni datos_externos.csv.")


if __name__ == "__main__":
    main()
