"""Importa recortes de TACO (Trash Annotations in Context) como datos EXTERNOS (regla 1 de CLAUDE.md).

  1. Descarga data/annotations.json oficial de TACO (commit fijo) y SOLO las imágenes que contienen
     las categorías elegidas y cuya licencia está permitida (CC BY 4.0 por defecto u ODbL).
     Las imágenes con licencia "CC" sin especificar se excluyen.
  2. Recorta cada objeto por su bbox: cuadrado, lado = 1.1 * max(w, h), centrado; descarta < 96 px.
  3. Anti-leakage: dHash de cada recorte contra TODOS los splits de dataset/publico y dataset/juliaca;
     elimina los de distancia <= 4. Informe: reportes/externo_dedup.csv.
  4. Registra cada imagen de origen (con su licencia) en datos_externos.csv.
  5. Mosaicos numerados (40 por página) en revision/externo/<categoria>/ para revisión manual.
  6. Si existe reportes/externo_taco_revision.csv (decisión manual del usuario, por archivo), mueve los
     aceptados a su clase final y los descartados a dataset/externo/_descartados/<categoria>/.

Uso:  python scripts/importar_taco.py
Atribución: Proença, P. F., & Simões, P. (2020). TACO: Trash Annotations in Context for Litter Detection.
arXiv:2003.06975. Imágenes ODbL: (c) OpenLitterMap & Contributors.
"""
import csv
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import DATASET, JULIACA, PUBLICO, RAIZ, SPLITS, dhash, imagenes

COMMIT = "70479ad9adc3da971a8ae75ac97eb1026910eec7"
URL_ANOT = f"https://raw.githubusercontent.com/pedropro/TACO/{COMMIT}/data/annotations.json"
EXTERNO = DATASET / "externo"
FUENTES = EXTERNO / "_fuentes" / "taco"
REGISTRO = RAIZ / "datos_externos.csv"
INFORME = RAIZ / "reportes" / "externo_dedup.csv"
DETALLE = RAIZ / "reportes" / "externo_taco_recortes.csv"
MOSAICOS = RAIZ / "revision" / "externo"
REVISION = RAIZ / "reportes" / "externo_taco_revision.csv"
DESCARTADOS = EXTERNO / "_descartados"

# categoría TACO (nombre exacto) -> (carpeta inicial, clase destino antes de revisar, clase si se acepta)
CATEGORIAS = {
    "Crisp packet": (EXTERNO / "train" / "generales", "generales", "generales"),
    "Other plastic wrapper": (EXTERNO / "_revision" / "other_plastic_wrapper", "pendiente_revision", "generales"),
    "Aluminium foil": (EXTERNO / "train" / "metales", "metales", "metales"),
}
LICENCIAS = {  # campo "license" de TACO -> licencia efectiva (None = CC BY 4.0 por defecto, ver tacodataset.org)
    None: "CC BY 4.0",
    "ODBL (c) OpenLitterMap & Contributors": "ODbL 1.0 (c) OpenLitterMap & Contributors",
}
MARGEN = 0.10
LADO_MIN = 96
UMBRAL_DHASH = 4
POR_PAGINA, COLS, TESELA = 40, 8, 200


def descargar(url, destino, intentos=3):
    if destino.exists() and destino.stat().st_size > 0:
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    for i in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "clasificador-residuos-juliaca"})
            with urllib.request.urlopen(req, timeout=60) as r:
                datos = r.read()
            tmp = destino.with_suffix(".part")
            tmp.write_bytes(datos)
            tmp.replace(destino)
            return
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"  reintento {url}: {e}")
            time.sleep(2)


def abrir_orientada(ruta, ancho, alto):
    """TACO anota sobre la imagen ya rotada según EXIF. Devuelve la imagen cuyo tamaño coincide con la anotación."""
    im = Image.open(ruta)
    rot = ImageOps.exif_transpose(im)
    if rot.size == (ancho, alto):
        return rot.convert("RGB")
    if im.size == (ancho, alto):
        return im.convert("RGB")
    return None


def recorte_cuadrado(bbox, ancho, alto):
    x, y, w, h = bbox
    lado = min(max(w, h) * (1 + MARGEN), ancho, alto)
    cx, cy = x + w / 2, y + h / 2
    x0 = min(max(cx - lado / 2, 0), ancho - lado)
    y0 = min(max(cy - lado / 2, 0), alto - lado)
    return round(x0), round(y0), round(x0 + lado), round(y0 + lado), lado


def mosaicos(archivos, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("mosaico_*.jpg"):
        viejo.unlink()
    with open(carpeta / "indice.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["numero", "mosaico", "archivo"])
        for n, p in enumerate(archivos, 1):
            w.writerow([n, f"mosaico_{(n - 1) // POR_PAGINA + 1:02d}.jpg", p.name])
    for pag in range(0, len(archivos), POR_PAGINA):
        lote = archivos[pag:pag + POR_PAGINA]
        filas = (len(lote) + COLS - 1) // COLS
        lienzo = Image.new("RGB", (COLS * TESELA, filas * TESELA), (255, 255, 255))
        d = ImageDraw.Draw(lienzo)
        for i, p in enumerate(lote):
            with Image.open(p) as im:
                t = im.convert("RGB").resize((TESELA - 4, TESELA - 4))
            x, y = (i % COLS) * TESELA + 2, (i // COLS) * TESELA + 2
            lienzo.paste(t, (x, y))
            d.rectangle([x, y, x + 44, y + 20], fill=(0, 0, 0))
            d.text((x + 4, y + 4), str(pag + i + 1), fill=(255, 255, 0))
        lienzo.save(carpeta / f"mosaico_{pag // POR_PAGINA + 1:02d}.jpg", quality=90)


def carpeta_cat(cat):
    return cat.lower().replace(" ", "_")


def aplicar_revision(detalle):
    """Mueve (no borra) cada recorte según la decisión manual registrada en REVISION."""
    with open(REVISION, newline="", encoding="utf-8") as f:
        decision = {(r["categoria_taco"], r["archivo"]): r["decision"] for r in csv.DictReader(f)}
    for fila in detalle:
        if fila["estado"] != "recortado":
            continue
        cat = fila["categoria_taco"]
        origen = RAIZ / fila["recorte"]
        dec = decision.get((cat, origen.name))
        if dec is None:
            sys.exit(f"{origen.name} ({cat}) no tiene decisión en {REVISION.name}")
        if dec == "aceptado":
            fila["clase_final"] = CATEGORIAS[cat][2]
            destino = EXTERNO / "train" / fila["clase_final"] / origen.name
            fila["estado"] = "aceptado_revision"
        else:
            fila["clase_final"] = "descartado"
            destino = DESCARTADOS / carpeta_cat(cat) / origen.name
            fila["estado"] = "descartado_revision"
        destino.parent.mkdir(parents=True, exist_ok=True)
        if origen != destino:
            origen.replace(destino)
        fila["recorte"] = str(destino.relative_to(RAIZ)).replace("\\", "/")


def main():
    hoy = date.today().isoformat()
    ruta_anot = FUENTES / "annotations.json"
    descargar(URL_ANOT, ruta_anot)
    d = json.loads(ruta_anot.read_text(encoding="utf-8"))
    ims = {i["id"]: i for i in d["images"]}
    cat_id = {c["name"]: c["id"] for c in d["categories"]}
    faltan = set(CATEGORIAS) - set(cat_id)
    if faltan:
        sys.exit(f"Categorías no encontradas en TACO: {faltan}")
    nombre_cat = {cat_id[n]: n for n in CATEGORIAS}

    anots = [a for a in d["annotations"] if a["category_id"] in nombre_cat]
    excluidas_lic = [a for a in anots if ims[a["image_id"]]["license"] not in LICENCIAS]
    anots = [a for a in anots if ims[a["image_id"]]["license"] in LICENCIAS]
    print(f"Objetos: {len(anots)} con licencia permitida, {len(excluidas_lic)} excluidos por licencia 'CC' sin especificar")

    # --- 1-2. Descarga y recorte ---
    carpetas = {c for c, _, _ in CATEGORIAS.values()} | {EXTERNO / "train" / c for _, _, c in CATEGORIAS.values()}
    carpetas |= {DESCARTADOS / carpeta_cat(cat) for cat in CATEGORIAS}
    for carpeta in carpetas:
        carpeta.mkdir(parents=True, exist_ok=True)
        for viejo in carpeta.glob("taco_*.jpg"):  # regenerable: se rehace todo desde cero
            viejo.unlink()
    detalle = []
    por_imagen = {}
    for a in sorted(anots, key=lambda a: (a["image_id"], a["id"])):
        im_info = ims[a["image_id"]]
        cat = nombre_cat[a["category_id"]]
        fila = {"recorte": "", "categoria_taco": cat, "image_id": a["image_id"], "ann_id": a["id"],
                "url": im_info["flickr_url"], "licencia": LICENCIAS[im_info["license"]],
                "bbox": json.dumps([round(v, 1) for v in a["bbox"]]), "lado_px": "", "estado": "",
                "clase_final": CATEGORIAS[cat][1]}
        detalle.append(fila)
        if a["image_id"] not in por_imagen:
            ext = Path(im_info["flickr_url"]).suffix.lower() or ".jpg"
            orig = FUENTES / "img" / f"{a['image_id']:04d}{ext}"
            print(f"  descargando imagen {a['image_id']}")
            descargar(im_info["flickr_url"], orig)
            por_imagen[a["image_id"]] = abrir_orientada(orig, im_info["width"], im_info["height"])
        im = por_imagen[a["image_id"]]
        if im is None:
            fila["estado"] = "descartado_tamano_no_coincide"
            continue
        x0, y0, x1, y1, lado = recorte_cuadrado(a["bbox"], *im.size)
        fila["lado_px"] = round(lado)
        if lado < LADO_MIN:
            fila["estado"] = "descartado_menor_96px"
            continue
        destino = CATEGORIAS[cat][0] / f"taco_{a['image_id']}_{a['id']}.jpg"
        im.crop((x0, y0, x1, y1)).save(destino, quality=95)
        fila["recorte"] = str(destino.relative_to(RAIZ)).replace("\\", "/")
        fila["estado"] = "recortado"
    por_imagen.clear()

    # --- 3. Anti-leakage: dHash contra todos los splits de publico y juliaca ---
    refs = [p for base in (PUBLICO, JULIACA) for s in SPLITS for p in imagenes(base / s)]
    print(f"Calculando dHash de {len(refs)} imágenes de referencia (publico + juliaca, todos los splits)...")
    ref_h = np.array([dhash(p) for p in refs], dtype=np.uint64)
    filas_dedup = []
    for fila in detalle:
        if fila["estado"] != "recortado":
            continue
        ruta = RAIZ / fila["recorte"]
        h = np.uint64(dhash(ruta))
        dist = np.array([bin(int(x)).count("1") for x in (ref_h ^ h)])
        j = int(dist.argmin())
        eliminado = int(dist[j]) <= UMBRAL_DHASH
        filas_dedup.append({"recorte": fila["recorte"], "categoria_taco": fila["categoria_taco"],
                            "dist_min": int(dist[j]), "mas_cercano": str(refs[j].relative_to(RAIZ)).replace("\\", "/"),
                            "n_refs_dist_le_4": int((dist <= UMBRAL_DHASH).sum()),
                            "eliminado": "SI" if eliminado else "NO", "_fila": fila})
        if eliminado:
            ruta.unlink()
            fila["estado"] = "eliminado_dhash"
            fila["recorte"] = ""
    for fila in detalle:
        if fila["estado"] not in ("recortado", ""):
            fila["clase_final"] = ""
    for a in excluidas_lic:
        detalle.append({"recorte": "", "categoria_taco": nombre_cat[a["category_id"]], "image_id": a["image_id"],
                        "ann_id": a["id"], "url": ims[a["image_id"]]["flickr_url"], "licencia": "CC (sin especificar)",
                        "bbox": json.dumps([round(v, 1) for v in a["bbox"]]), "lado_px": "", "estado": "excluido_licencia",
                        "clase_final": ""})

    # --- 5. Mosaicos (numeración previa a la revisión; la revisión manual se hizo sobre ellos) ---
    for cat, (carpeta, _, _) in CATEGORIAS.items():
        mosaicos(sorted(carpeta.glob("taco_*.jpg")), MOSAICOS / carpeta_cat(cat))

    # --- 6. Revisión manual ---
    if REVISION.exists():
        aplicar_revision(detalle)
    for fd in filas_dedup:
        fd["ubicacion_final"] = fd.pop("_fila")["recorte"] or "eliminado"
    INFORME.parent.mkdir(exist_ok=True)
    with open(INFORME, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_dedup[0]))
        w.writeheader()
        w.writerows(filas_dedup)
    with open(DETALLE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detalle[0]))
        w.writeheader()
        w.writerows(detalle)

    # --- 4. Registro datos_externos.csv (una fila por imagen de origen, categoría y clase final) ---
    campos = ["fuente", "url", "licencia", "fecha", "categoria_origen", "clase_destino", "n_imagenes", "archivos"]
    otras = []
    if REGISTRO.exists():
        with open(REGISTRO, newline="", encoding="utf-8") as f:
            otras = [r for r in csv.DictReader(f) if not r["fuente"].startswith("TACO")]
    grupos = {}
    for fila in detalle:
        if fila["recorte"]:
            grupos.setdefault((fila["image_id"], fila["categoria_taco"], fila["clase_final"]), []).append(fila)
    nuevas = [{"fuente": f"TACO (Proença & Simões, 2020) image_id={iid}", "url": g[0]["url"],
               "licencia": g[0]["licencia"], "fecha": hoy, "categoria_origen": cat,
               "clase_destino": clase, "n_imagenes": len(g),
               "archivos": ";".join(Path(x["recorte"]).name for x in g)}
              for (iid, cat, clase), g in sorted(grupos.items())]
    with open(REGISTRO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(otras + nuevas)

    # --- Resumen ---
    print("\nResumen por categoría:")
    for cat in CATEGORIAS:
        filas = [f for f in detalle if f["categoria_taco"] == cat]
        cuenta = {}
        for f in filas:
            cuenta[f["estado"]] = cuenta.get(f["estado"], 0) + 1
        print(f"  {cat}: {cuenta}")
    print("Conteo final en dataset/externo/train:")
    for clase in sorted({c for _, _, c in CATEGORIAS.values()}):
        print(f"  {clase}: {len(list((EXTERNO / 'train' / clase).glob('taco_*.jpg')))}")
    print(f"Informe dedup: {INFORME.relative_to(RAIZ)} | detalle: {DETALLE.relative_to(RAIZ)} | registro: {REGISTRO.name}")


if __name__ == "__main__":
    main()
