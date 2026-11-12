"""Fase A (solo informe) de la ronda del 2026-11-11 con productos de tienda: Freiburg Groceries y Grocery Store Dataset.

  python -I scripts/candidatas_tiendas.py            # mapeo por foto, dedup, conteos, tope propuesto y mosaicos

No hay modo de importación: no escribe en dataset/ ni en datos_externos.csv. Lee las imágenes de
D:\\Proyecto-Residuos-ML\\DATA-NEW (no ejecuta nada de esos repositorios).

Fuentes:
  frg : Freiburg Groceries Dataset (Jund et al. 2016, github.com/PhilJd/freiburg_groceries_dataset). Sin licencia
        formal (ni en el repo ni en la página del dataset). 25 clases de productos, PNG 256x256 con relleno gris
        (128,128,128) que se recorta. Las clases mezclan envases (BEANS: latas, frascos y bolsas): el envase se decide
        POR FOTO con CLIP ViT-B/32 zero-shot y solo se acepta si coincide con lo que el usuario pidió para esa clase.
        Las clases donde CLIP no separa bien los envases (ver MAPEO_FRG) se descartan tras revisar los mosaicos.
  gsd : Grocery Store Dataset (Klasson et al. 2019, github.com/marcusklasson/GroceryStoreDataset), licencia MIT del
        repositorio. Solo imágenes naturales (train, val, test del autor); las icónicas se excluyen. Fruit y
        Vegetables -> organico; Packages (todos cartones de leche, jugo, yogur: revisados en mosaicos) -> papel_carton.

Filtros (se anota el primero que elimina): envase_no_coincide (frg) | nombre_benchmark | dhash_benchmark (8 giros,
1.455 + 238 gris) | dhash_propios (publico, juliaca, val_camara, externo/train, externo/_fuentes) | orb_benchmark |
recorte_chico (lado menor < 64 tras quitar relleno) | dup_interno (dHash <= 4, 8 giros) | consecutiva (embedding
yolo26s-cls, coseno >= 0.95 con una aceptada de la misma fuente y categoría) | fuera_del_tope (150 por fuente y clase,
reparto por categoría; gsd/organico reserva 20 para Potato, Onion y Garlic).
Informes: reportes/externo_candidatas3_{dedup,conteos}.csv y reportes/mosaicos_candidatas3/.
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import EXTERNO, RAIZ, imagenes  # noqa: E402
from importar_candidatas import mosaico  # noqa: E402
from importar_roboflow import UMBRAL, banco_benchmark, banco_propios, dhash_8, dhash_pil, dist, filtro_orb, original  # noqa: E402

DATA = Path(r"D:\Proyecto-Residuos-ML\DATA-NEW")
FRG = DATA / "freiburg_groceries_dataset" / "images"
GSD = DATA / "GroceryStoreDataset" / "dataset"
INF = RAIZ / "reportes"
MOSAICOS = INF / "mosaicos_candidatas3"
MIN_LADO = 64
COS_CONSEC = 0.95
TOPE = 150
PRIORIDAD_GSD = {"Potato": 20, "Onion": 20, "Garlic": 20}
P_CLIP = 0.5  # probabilidad mínima del envase pedido (softmax sobre todos los envases)

ENVASES = {
    "caja": "a photo of cardboard boxes of food on a supermarket shelf",
    "carton_bebida": "a photo of beverage cartons, tetra pak milk or juice cartons",
    "lata": "a photo of metal tin cans",
    "bolsa_plastico": "a photo of plastic bags of snacks or sweets",
    "bolsa_papel": "a photo of paper bags",
    "botella_pet": "a photo of plastic PET bottles",
    "botella_vidrio": "a photo of glass bottles",
    "frasco_vidrio": "a photo of glass jars",
    "envase_plastico": "a photo of plastic tubs or plastic squeeze bottles",
    "envoltura": "a photo of chocolate bars in wrappers",
    "tubo_carton": "a photo of cylindrical snack tubes like Pringles",
}
# clase Freiburg -> [(envase aceptado, clase destino, estado)]; las clases que no están se descartan
LATA = ("lata", "metales", "propuesta")
MAPEO_FRG = {
    "TEA": [("caja", "papel_carton", "propuesta")],
    "MILK": [("carton_bebida", "papel_carton", "propuesta")],
    "CEREAL": [("caja", "papel_carton", "dudosa")],  # cajas, pero CLIP deja pasar bolsas de muesli (plástico)
    **{k: [LATA] for k in ("BEANS", "CORN", "FISH", "TOMATO_SAUCE")},
    "CHIPS": [("bolsa_plastico", "generales", "propuesta")],
    "CANDY": [("bolsa_plastico", "generales", "propuesta")],
    "NUTS": [("bolsa_plastico", "generales", "dudosa")],
    "WATER": [("botella_pet", "plasticos", "propuesta")],
    "SODA": [("botella_pet", "plasticos", "propuesta"), LATA],
    "JAM": [("frasco_vidrio", "vidrio", "propuesta")],
    "HONEY": [("frasco_vidrio", "vidrio", "propuesta")],
}
# Revisión visual de los mosaicos (2026-11-11): con CLIP, CHOCOLATE, COFFEE, PASTA, FLOUR y JUICE -> papel_carton
# mezclan envolturas, frascos, bolsas plásticas, botellas PET y de vidrio (~la mitad mal): se descartan enteras.
# COFFEE -> metales trae frascos de vidrio y NUTS -> metales un tarro plástico: también se descartan.
# Para papel_carton, caja y cartón de bebida son el mismo material (CLIP llama "carton de bebida" a casi todas las
# cajas de té, 263 de 283): se acepta cualquiera de los dos.
EQUIV = {"caja": ("caja", "carton_bebida"), "carton_bebida": ("caja", "carton_bebida")}


class Disco:
    """Imita zipfile para reutilizar filtro_orb de importar_roboflow con archivos en disco."""

    def read(self, ruta):
        return Path(ruta).read_bytes()


def quitar_relleno(im, gris=128, tol=2):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    contenido = (np.abs(a - gris) > tol).any(axis=2)
    filas, cols = np.where(contenido.any(axis=1))[0], np.where(contenido.any(axis=0))[0]
    if len(filas) == 0:
        return im
    return im.crop((cols[0], filas[0], cols[-1] + 1, filas[-1] + 1))


def cargar(c):
    im = Image.open(c["ruta_zip"]).convert("RGB")
    return quitar_relleno(im) if c["fuente"] == "frg" else im


def candidatas(desc):
    res = []
    for d in sorted(p for p in FRG.iterdir() if p.is_dir()):
        fotos = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg"))
        if d.name not in MAPEO_FRG:
            desc[("frg", f"{d.name} (no pedida)")] += len(fotos)
            continue
        res += [dict(fuente="frg", ruta_zip=str(p), original=p.stem, split="-", categoria=d.name, archivo=p.name)
                for p in fotos]
    for s in ("train", "val", "test"):
        for p in sorted((GSD / s).rglob("*.jpg")):
            rel = p.relative_to(GSD / s).parts  # Fruit/Apple/Golden-Delicious/x.jpg | Vegetables/Garlic/x.jpg
            grupo, coarse = rel[0], rel[1]
            clase = "papel_carton" if grupo == "Packages" else "organico"
            res.append(dict(fuente="gsd", ruta_zip=str(p), original=p.stem, split=s, categoria=coarse,
                            fino=rel[2] if len(rel) == 4 else coarse, archivo=p.name, clase=clase,
                            estado="propuesta", envase="carton_bebida" if grupo == "Packages" else "fruta_verdura"))
    desc[("gsd", "iconicas (excluidas)")] += sum(1 for _ in (GSD / "iconic-images-and-descriptions").rglob("*.jpg"))
    disco = Disco()
    for c in res:
        c["zip"] = disco
    return res


def clip_envases(cands):
    import clip
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, prep = clip.load("ViT-B/32", device=dev)
    nombres = list(ENVASES)
    with torch.no_grad():
        t = modelo.encode_text(clip.tokenize([ENVASES[k] for k in nombres]).to(dev)).float()
        t /= t.norm(dim=-1, keepdim=True)
        for i in range(0, len(cands), 128):
            lote = cands[i:i + 128]
            x = torch.stack([prep(cargar(c)) for c in lote]).to(dev)
            f = modelo.encode_image(x).float()
            f /= f.norm(dim=-1, keepdim=True)
            pr = (100 * f @ t.T).softmax(dim=-1).cpu().numpy()
            for c, p in zip(lote, pr):
                j = int(p.argmax())
                c["envase"], c["p_envase"] = nombres[j], round(float(p[j]), 3)


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
    cands = candidatas(desc)
    print(f"Candidatas: {Counter(c['fuente'] for c in cands)}; descartadas antes: {dict(desc)}", flush=True)

    frg = [c for c in cands if c["fuente"] == "frg"]
    clip_envases(frg)
    for c in cands:
        c.update(motivo="ok", detalle="", dist="", orb="")
    for c in frg:
        ok = [m for m in MAPEO_FRG[c["categoria"]] if c["envase"] in EQUIV.get(m[0], (m[0],))]
        if ok and c["p_envase"] >= P_CLIP:
            c["clase"], c["estado"] = ok[0][1], ok[0][2]
        else:
            c["clase"], c["estado"] = "", ""
            c["motivo"], c["detalle"] = "envase_no_coincide", f"{c['envase']} {c['p_envase']}"

    nombres_b, banco_b, quien_b = banco_benchmark()
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
        im = cargar(c)
        c["tam"] = "x".join(map(str, im.size))
        c["lado"] = min(im.size)
        if c["motivo"] != "ok":
            continue
        h = dhash_pil(im)
        if original(c["archivo"]) in nombres_b:
            c["motivo"], c["detalle"] = "nombre_benchmark", nombres_b[original(c["archivo"])]
        elif (d := dist(banco_b, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_benchmark", quien_b[int(d.argmin())], int(d.min())
        elif (d := dist(banco_p, h)).min() <= UMBRAL:
            c["motivo"], c["detalle"], c["dist"] = "dhash_propios", quien_p[int(d.argmin())], int(d.min())
        if i % 1000 == 0:
            print(f"  dHash {i}/{len(cands)}", flush=True)

    filtro_orb([c for c in cands if c["motivo"] == "ok"])

    for c in cands:
        if c["motivo"] == "ok" and c["lado"] < MIN_LADO:
            c["motivo"] = "recorte_chico"

    vivos = [c for c in cands if c["motivo"] == "ok"]
    E = embeddings([cargar(c).resize((224, 224)) for c in vivos])
    acept = defaultdict(list)
    for c, e in zip(vivos, E):
        k = (c["fuente"], c["categoria"], c.get("fino", ""))
        hs = dhash_8(cargar(c))
        dmin = min((int(dist(np.array([a[0]], dtype=np.uint64), x).min()) for a in acept[k] for x in hs), default=99)
        cmax = max((float(a[1] @ e) for a in acept[k]), default=0.0)
        if dmin <= UMBRAL:
            c["motivo"], c["dist"] = "dup_interno", dmin
        elif cmax >= COS_CONSEC:
            c["motivo"], c["dist"] = "consecutiva", round(cmax, 3)
        else:
            acept[k].append((hs[0], e))

    # tope por fuente y clase: reparto por categoría (round-robin), con reservas en gsd/organico
    rng = random.Random(42)
    grupos = defaultdict(lambda: defaultdict(list))
    for c in cands:
        if c["motivo"] == "ok":
            grupos[(c["fuente"], c["clase"])][c["categoria"]].append(c)
    for (fu, cl), porcat in grupos.items():
        for v in porcat.values():
            rng.shuffle(v)
        elegidas = []
        if fu == "gsd" and cl == "organico":
            for cat, n in PRIORIDAD_GSD.items():
                elegidas += porcat.get(cat, [])[:n]
                porcat[cat] = porcat.get(cat, [])[n:]
        colas = [v for _, v in sorted(porcat.items())]
        while len(elegidas) < TOPE and any(colas):
            for v in colas:
                if v and len(elegidas) < TOPE:
                    elegidas.append(v.pop())
        for v in colas:
            for c in v:
                c["motivo"] = "fuera_del_tope"
        for c in elegidas:
            c["propuesta"] = True

    INF.mkdir(exist_ok=True)
    campos = ["fuente", "split", "categoria", "fino", "envase", "p_envase", "clase", "estado", "ruta_zip", "tam",
              "lado", "motivo", "dist", "orb", "detalle"]
    with open(INF / "externo_candidatas3_dedup.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(cands)
    motivos = ["envase_no_coincide", "nombre_benchmark", "dhash_benchmark", "dhash_propios", "orb_benchmark",
               "recorte_chico", "dup_interno", "consecutiva", "fuera_del_tope", "ok"]
    cont = defaultdict(Counter)
    for c in cands:
        k = (c["fuente"], c["categoria"], c["clase"] or "-")
        cont[k][c["motivo"]] += 1
        cont[k]["total"] += 1
    with open(INF / "externo_candidatas3_conteos.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fuente", "categoria", "clase", "total"] + motivos)
        for k, v in sorted(cont.items()):
            w.writerow([*k, v["total"]] + [v[m] for m in motivos])
        for (fu, motivo), n in sorted(desc.items()):
            w.writerow([fu, motivo, "descartada", n] + [""] * len(motivos))
    print(f"\n{'fuente/categoría -> clase':40s}{'total':>6s}" + "".join(f"{m[:10]:>11s}" for m in motivos))
    for k, v in sorted(cont.items()):
        print(f"{k[0] + '/' + k[1] + ' -> ' + k[2]:40s}{v['total']:>6d}" + "".join(f"{v[m]:>11d}" for m in motivos))
    print("\nPropuestas (tras tope):", dict(Counter((c["fuente"], c["clase"]) for c in cands if c["motivo"] == "ok")))
    print("Limpias antes del tope:", dict(Counter((c["fuente"], c["clase"]) for c in cands
                                              if c["motivo"] in ("ok", "fuera_del_tope"))))
    print("Envase CLIP por clase frg:")
    for cat in sorted(MAPEO_FRG):
        print(f"  {cat:13s}", dict(Counter(c["envase"] for c in frg if c["categoria"] == cat).most_common()))

    MOSAICOS.mkdir(exist_ok=True)
    porcl = defaultdict(list)
    rech = defaultdict(list)
    for c in cands:
        if c["motivo"] in ("ok", "fuera_del_tope") or (c["fuente"] == "frg" and c["motivo"] == "envase_no_coincide"):
            im = cargar(c)
            im.thumbnail((150, 150))
            if c["motivo"] == "envase_no_coincide":
                rech[c["categoria"]].append(dict(img=im))
            else:
                cl = c["clase"] if c["fuente"] == "gsd" else f"{c['categoria']}_{c['clase']}"
                porcl[(c["fuente"], cl, c["motivo"] == "ok")].append(dict(img=im))
    for (fu, cl, prop), its in porcl.items():
        nom = "propuestas" if prop else "fuera_del_tope"
        mosaico(its, MOSAICOS / f"{fu}_{cl}_{nom}.jpg", f"{fu} -> {cl}: {len(its)} {nom} (muestra de {min(64, len(its))})",
                n=64)
    for cat, its in rech.items():
        mosaico(its, MOSAICOS / f"frg_{cat}_rechazadas_envase.jpg",
                f"frg / {cat}: {len(its)} rechazadas por envase (muestra de {min(48, len(its))})")
    print(f"Mosaicos en {MOSAICOS}\nSOLO INFORME: no se tocó dataset/ ni datos_externos.csv.")


if __name__ == "__main__":
    main()
