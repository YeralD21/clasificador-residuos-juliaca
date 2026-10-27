"""Importa datasets públicos APROBADOS por el usuario como datos EXTERNOS (regla 1 de CLAUDE.md).

  python scripts/importar_publicos.py --dry-run     # descarga/prepara, deduplica y muestra el plan; no toca externo/train
  python scripts/importar_publicos.py               # copia a dataset/externo/train y registra en datos_externos.csv

Fuentes aprobadas (decisión del usuario, 2026-10-26) y prioridad dentro del tope:
  1. realwaste  : RealWaste (UCI, CC BY 4.0), sin Textile ni Miscellaneous Trash. Metal pasa por revisión
                  manual (aerosoles -> peligrosos, ver --revision-metal y REV_TXT).
  2. taco_extra : TACO, categorías nuevas (las de importar_taco.py no se tocan), recortes >= 96 px.
  3. trashnet   : TrashNet (MIT), sin trash. Última prioridad.
NO aprobados: Garbage-12 (imágenes sin licencia) y Household Waste de Kaggle (descartado tras revisar una muestra:
fotos de stock, gráficos, ilustraciones y duplicados entre clases; ver datos/fuentes.md).
Cada mapeo está en MAPEO; lo que no está ahí se descarta, no se fuerza.

Revisión de RealWaste Metal (decisión del usuario): la primera ejecución crea mosaicos numerados (48 por hoja) en
reportes/revision_realwaste_metal/ con su indice.csv, y la plantilla reportes/revision_realwaste_metal.txt, que
NUNCA se sobrescribe. El usuario anota ahí los números de los aerosoles (-> peligrosos) y otras correcciones
("N clase" o "N descartar"), y pone "terminado: si". Sin eso la importación real se niega.

Pasos:
  1. Prepara cada fuente en dataset/externo/_fuentes/<fuente>/ (descarga + descompresión; datos no confiables:
     solo se leen como imágenes, nunca se ejecutan; se rechazan rutas que salgan de la carpeta).
  2. Anti-leakage (regla 1d): dHash de cada candidata contra TODOS los splits de publico y juliaca, val_camara y lo que
     ya hay en externo/train (TACO previo). <= 4 -> eliminada. Luego, en orden de prioridad, contra las ya
     aceptadas (duplicados dentro de cada fuente y entre fuentes): la de mayor prioridad se queda.
  3. Tope: como máximo --tope (1200) imágenes NUEVAS por clase, llenando por prioridad; dentro de cada fuente,
     muestra aleatoria con --semilla fija (reproducible). No es un split: todo va a externo/train.
  4. Sin --dry-run: borra SOLO sus propios archivos previos (prefijos de PREFIJO) y copia los elegidos a
     dataset/externo/train/<clase>/<prefijo>_..., sin modificar los píxeles (los de TACO son recortes).
     datos_externos.csv: reemplaza SOLO sus propias filas (fuente que empieza por "<fuente>:").
Informes (en ambos modos): reportes/externo_publicos_dedup.csv (cada candidata: distancia, motivo, si entra),
reportes/externo_publicos_conteos.csv (clase x fuente: candidatas, tras dedup, tras tope) y mosaicos de ~24
imágenes por fuente y clase en reportes/mosaicos_externos/ (no se suben a git).
"""
import argparse
import csv
import json
import random
import re
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES, EXTERNO, JULIACA, PUBLICO, RAIZ, SPLITS, VAL_CAMARA, dhash, imagenes
import importar_taco as taco

FUENTES = EXTERNO / "_fuentes"
TRAIN = EXTERNO / "train"
REGISTRO = RAIZ / "datos_externos.csv"
INF_DEDUP = RAIZ / "reportes" / "externo_publicos_dedup.csv"
INF_CONTEOS = RAIZ / "reportes" / "externo_publicos_conteos.csv"
MOSAICOS = RAIZ / "reportes" / "mosaicos_externos"
PRIORIDAD = ("realwaste", "taco_extra", "trashnet")
PREFIJO = {"realwaste": "realwaste", "taco_extra": "tacox", "trashnet": "trashnet"}
UMBRAL_DHASH = 4
LADO_MIN_TACO = 96  # igual que importar_taco.py
REV_DIR = RAIZ / "reportes" / "revision_realwaste_metal"   # mosaicos (no se suben) + indice.csv
REV_TXT = RAIZ / "reportes" / "revision_realwaste_metal.txt"  # anotaciones del usuario
REV_POR_HOJA, REV_COLS, REV_LADO = 48, 8, 230

URL = {
    "trashnet": "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
    "realwaste": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
}
LICENCIA = {
    "trashnet": "MIT (repositorio garythung/trashnet)",
    "realwaste": "CC BY 4.0",
}
CITA = {
    "trashnet": "Thung & Yang (2016), TrashNet",
    "realwaste": "Single, Iranmanesh & Raad (2023), RealWaste, UCI ML Repository, doi:10.24432/C5SS4G",
    "taco_extra": "TACO, Proença & Simões (2020), arXiv:2003.06975",
}

# categoría original -> clase destino. Lo que no aparece aquí se descarta.
MAPEO = {
    "trashnet": {"glass": "vidrio", "paper": "papel_carton", "cardboard": "papel_carton", "plastic": "plasticos",
                 "metal": "metales"},  # trash: descartado
    "realwaste": {"Cardboard": "papel_carton", "Paper": "papel_carton", "Glass": "vidrio", "Metal": "metales",
                  "Plastic": "plasticos", "Food Organics": "organico", "Vegetation": "organico"},
    # Textile Trash y Miscellaneous Trash: descartados. Metal: revisión manual (aerosoles -> peligrosos).
    "taco_extra": {
        "Drink carton": "papel_carton", "Corrugated carton": "papel_carton", "Other carton": "papel_carton",
        "Egg carton": "papel_carton", "Toilet tube": "papel_carton", "Paper bag": "papel_carton",
        "Normal paper": "papel_carton", "Magazine paper": "papel_carton", "Wrapping paper": "papel_carton",
        "Clear plastic bottle": "plasticos", "Other plastic bottle": "plasticos", "Disposable plastic cup": "plasticos",
        "Plastic lid": "plasticos", "Plastic bottle cap": "plasticos", "Single-use carrier bag": "plasticos",
        "Plastic film": "plasticos", "Disposable food container": "plasticos", "Plastic utensils": "plasticos",
        "Plastic straw": "plasticos", "Garbage bag": "plasticos", "Spread tub": "plasticos",
        "Tupperware": "plasticos", "Polypropylene bag": "plasticos", "Six pack rings": "plasticos",
        "Drink can": "metales", "Food Can": "metales", "Metal bottle cap": "metales", "Metal lid": "metales",
        "Pop tab": "metales", "Scrap metal": "metales",
        "Glass bottle": "vidrio", "Glass jar": "vidrio", "Glass cup": "vidrio", "Broken glass": "vidrio",
        "Tissues": "generales", "Styrofoam piece": "generales", "Foam cup": "generales",
        "Foam food container": "generales",
        "Aerosol": "peligrosos", "Battery": "peligrosos",
        "Food waste": "organico",
    },
    # Cigarette, Unlabeled litter, Other plastic, Meal carton, Pizza box, Shoe, Rope & strings, ... : descartados
}


# ---------- preparación de fuentes (datos no confiables: solo lectura de imágenes) ----------

def descomprimir(zip_path, destino):
    """Extrae rechazando rutas absolutas o con '..' (zip slip). No ejecuta nada."""
    marca = destino / ".extraido"
    if marca.exists():
        return
    destino.mkdir(parents=True, exist_ok=True)
    base = destino.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            final = (destino / info.filename).resolve()
            if base != final and base not in final.parents:
                raise SystemExit(f"{zip_path.name}: ruta sospechosa dentro del zip: {info.filename}")
        z.extractall(destino)
    marca.write_text(zip_path.name, encoding="utf-8")


def es_basura_mac(p):
    """__MACOSX/ y ._archivo son metadatos de macOS dentro de los zip (AppleDouble), no imágenes."""
    return "__MACOSX" in p.parts or p.name.startswith("._")


def fotos_en(d):
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not es_basura_mac(p))


def clase_dir(raiz, nombre):
    """Carpeta (en cualquier profundidad) que se llama `nombre`: la que tiene más imágenes, sin __MACOSX."""
    dirs = [d for d in raiz.rglob(nombre) if d.is_dir() and not es_basura_mac(d) and fotos_en(d)]
    return max(dirs, key=lambda d: len(fotos_en(d))) if dirs else None


def cand(fuente, clase, ruta, categoria, url, licencia, nombre):
    return {"fuente": fuente, "clase": clase, "ruta": ruta, "categoria": categoria, "url": url,
            "licencia": licencia, "nombre": nombre}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def preparar_zip(fuente):
    carpeta = FUENTES / fuente
    zp = carpeta / f"{fuente}.zip"
    if not zp.exists():
        print(f"[{fuente}] descargando {URL[fuente]} ...", flush=True)
        taco.descargar(URL[fuente], zp)
    descomprimir(zp, carpeta / "extraido")
    out, avisos = [], []
    for cat, clase in MAPEO[fuente].items():
        d = clase_dir(carpeta / "extraido", cat)
        if d is None:
            avisos.append(f"[{fuente}] no se encontró la categoría '{cat}'")
            continue
        for p in fotos_en(d):
            out.append(cand(fuente, clase, p, cat, URL[fuente], LICENCIA[fuente],
                            f"{PREFIJO[fuente]}_{slug(cat)}_{slug(p.stem)}{p.suffix.lower()}"))
    return out, avisos


def preparar_taco_extra():
    """Recortes de las categorías nuevas de TACO (mismo recorte y licencias que importar_taco.py)."""
    anot = FUENTES / "taco" / "annotations.json"
    taco.descargar(taco.URL_ANOT, anot)
    d = json.loads(anot.read_text(encoding="utf-8"))
    ims = {i["id"]: i for i in d["images"]}
    nombre = {c["id"]: c["name"] for c in d["categories"]}
    faltan = set(MAPEO["taco_extra"]) - set(nombre.values())
    if faltan:
        raise SystemExit(f"Categorías de TACO no encontradas: {faltan}")
    anots = [a for a in d["annotations"] if nombre[a["category_id"]] in MAPEO["taco_extra"]]
    ok = [a for a in anots if ims[a["image_id"]]["license"] in taco.LICENCIAS]
    avisos = [f"[taco_extra] {len(anots) - len(ok)} objetos excluidos por licencia 'CC' sin especificar"]
    rec_dir = FUENTES / "taco_extra" / "recortes"
    rec_dir.mkdir(parents=True, exist_ok=True)
    out, chicos, n_img = [], 0, len({a["image_id"] for a in ok})
    abiertas = {}
    for k, a in enumerate(sorted(ok, key=lambda a: (a["image_id"], a["id"]))):
        info = ims[a["image_id"]]
        cat = nombre[a["category_id"]]
        destino = rec_dir / f"tacox_{a['image_id']}_{a['id']}.jpg"
        if not destino.exists():
            x, y, w, h = a["bbox"]
            if min(info["width"], info["height"], max(w, h) * (1 + taco.MARGEN)) < LADO_MIN_TACO:
                chicos += 1
                continue
            if a["image_id"] not in abiertas:
                abiertas.clear()  # una imagen en memoria a la vez
                orig = FUENTES / "taco" / "img" / f"{a['image_id']:04d}{Path(info['flickr_url']).suffix.lower() or '.jpg'}"
                if not orig.exists():
                    print(f"  [taco_extra] imagen {a['image_id']} ({k + 1}/{len(ok)} objetos, {n_img} imágenes)",
                          flush=True)
                taco.descargar(info["flickr_url"], orig)
                abiertas[a["image_id"]] = taco.abrir_orientada(orig, info["width"], info["height"])
            im = abiertas[a["image_id"]]
            if im is None:
                continue
            x0, y0, x1, y1, lado = taco.recorte_cuadrado(a["bbox"], *im.size)
            if lado < LADO_MIN_TACO:
                chicos += 1
                continue
            im.crop((x0, y0, x1, y1)).save(destino, quality=95)
        out.append(cand("taco_extra", MAPEO["taco_extra"][cat], destino, cat, info["flickr_url"],
                        taco.LICENCIAS[info["license"]], destino.name))
    avisos.append(f"[taco_extra] {chicos} objetos descartados por recorte < {LADO_MIN_TACO} px")
    return out, avisos


# ---------- dedup, tope, informes ----------

TABLA_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def distancias(banco, h):
    if not len(banco):
        return np.array([], dtype=np.uint8)
    return TABLA_BITS[(banco ^ np.uint64(h)).view(np.uint8).reshape(-1, 8)].sum(axis=1)


def mosaico(fotos, ruta, titulo, n=24, cols=6, lado=180):
    sel = fotos[:n]
    filas = (len(sel) + cols - 1) // cols
    lienzo = Image.new("RGB", (cols * lado, filas * lado + 30), (255, 255, 255))
    d = ImageDraw.Draw(lienzo)
    d.text((6, 8), titulo, fill=(0, 0, 0))
    for i, p in enumerate(sel):
        with Image.open(p) as im:
            t = im.convert("RGB")
            t.thumbnail((lado - 4, lado - 4))
        lienzo.paste(t, ((i % cols) * lado + 2, (i // cols) * lado + 32))
    lienzo.save(ruta, quality=88)


PLANTILLA_REV = """# Revisión manual de RealWaste Metal ({n} imágenes, mosaicos en reportes/revision_realwaste_metal/).
# El número de cada imagen está en su esquina; indice.csv dice qué archivo es cada número.
# - aerosoles: números separados por coma o espacio; pasan a peligrosos.
# - otras correcciones: una por línea, "número clase" (una de: {clases}) o "número descartar".
# - cuando termines, cambia "terminado: no" por "terminado: si". Sin eso no se importa nada.
# Las líneas que empiezan con # se ignoran. Este archivo no lo sobrescribe ningún script.

terminado: no

aerosoles:

# otras correcciones (ejemplos, borra el # para usarlos):
# 123 generales
# 456 descartar
"""


def preparar_revision(metal):
    """Primera vez: indice.csv + mosaicos numerados + plantilla. Después: comprueba que el índice siga valiendo."""
    idx = REV_DIR / "indice.csv"
    nombres = sorted(c["nombre"] for c in metal)
    if idx.exists():
        with open(idx, newline="", encoding="utf-8") as f:
            guardados = [r["archivo"] for r in csv.DictReader(f)]
        if guardados != nombres:
            raise SystemExit(f"{idx.relative_to(RAIZ)} ya no coincide con las imágenes de RealWaste Metal "
                             f"({len(guardados)} en el índice, {len(nombres)} ahora). Los números anotados no se "
                             "pueden aplicar con seguridad: avisa antes de regenerar.")
        return
    REV_DIR.mkdir(parents=True, exist_ok=True)
    por_nombre = {c["nombre"]: c for c in metal}
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["numero", "mosaico", "archivo", "origen"])
        for n, nom in enumerate(nombres, 1):
            w.writerow([n, f"mosaico_{(n - 1) // REV_POR_HOJA + 1:02d}.jpg", nom,
                        por_nombre[nom]["ruta"].relative_to(RAIZ).as_posix()])
    fuente = ImageFont.load_default(size=30)
    for hoja in range(0, len(nombres), REV_POR_HOJA):
        lote = nombres[hoja:hoja + REV_POR_HOJA]
        filas = (len(lote) + REV_COLS - 1) // REV_COLS
        lienzo = Image.new("RGB", (REV_COLS * REV_LADO, filas * REV_LADO), (255, 255, 255))
        d = ImageDraw.Draw(lienzo)
        for i, nom in enumerate(lote):
            with Image.open(por_nombre[nom]["ruta"]) as im:
                t = im.convert("RGB").resize((REV_LADO - 6, REV_LADO - 6))
            x, y = (i % REV_COLS) * REV_LADO + 3, (i // REV_COLS) * REV_LADO + 3
            lienzo.paste(t, (x, y))
            d.rectangle([x, y, x + 72, y + 38], fill=(0, 0, 0))
            d.text((x + 6, y + 3), str(hoja + i + 1), fill=(255, 255, 0), font=fuente)
        lienzo.save(REV_DIR / f"mosaico_{hoja // REV_POR_HOJA + 1:02d}.jpg", quality=90)
    if not REV_TXT.exists():
        REV_TXT.write_text(PLANTILLA_REV.format(n=len(nombres), clases=", ".join(CLASES)), encoding="utf-8")
    print(f"Revisión de RealWaste Metal: {len(nombres)} imágenes en "
          f"{(len(nombres) + REV_POR_HOJA - 1) // REV_POR_HOJA} mosaicos ({REV_DIR.relative_to(RAIZ)}); "
          f"anota en {REV_TXT.relative_to(RAIZ)}")


def leer_revision():
    """{archivo: clase nueva o 'descartar'} y si el usuario marcó 'terminado: si'."""
    if not REV_TXT.exists():
        return {}, False
    with open(REV_DIR / "indice.csv", newline="", encoding="utf-8") as f:
        por_num = {int(r["numero"]): r["archivo"] for r in csv.DictReader(f)}
    cambios, terminado, errores = {}, False, []
    for k, linea in enumerate(REV_TXT.read_text(encoding="utf-8").splitlines(), 1):
        linea = linea.split("#", 1)[0].strip()
        if not linea:
            continue
        clave, _, resto = linea.partition(":")
        if clave.strip().lower() == "terminado":
            terminado = resto.strip().lower() in ("si", "sí")
            continue
        if clave.strip().lower() == "aerosoles":
            pares = [(n, "peligrosos") for n in re.split(r"[,\s]+", resto.strip()) if n]
        else:
            partes = linea.split()
            if len(partes) != 2:
                errores.append(f"línea {k}: '{linea}' (se espera 'número clase' o 'número descartar')")
                continue
            pares = [tuple(partes)]
        for num, clase in pares:
            if not num.isdigit() or int(num) not in por_num:
                errores.append(f"línea {k}: número '{num}' fuera de 1..{len(por_num)}")
            elif clase not in CLASES and clase != "descartar":
                errores.append(f"línea {k}: clase '{clase}' desconocida")
            elif por_num[int(num)] in cambios and cambios[por_num[int(num)]] != clase:
                errores.append(f"línea {k}: el {num} ya estaba anotado como {cambios[por_num[int(num)]]}")
            else:
                cambios[por_num[int(num)]] = clase
    if errores:
        raise SystemExit(f"Errores en {REV_TXT.name} (no se aplicó nada):\n  " + "\n  ".join(errores))
    return cambios, terminado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="no toca externo/train ni datos_externos.csv")
    ap.add_argument("--tope", type=int, default=1200, help="máximo de imágenes NUEVAS por clase")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--excluir", nargs="+", default=[], choices=PRIORIDAD, help="fuentes que no se usan")
    args = ap.parse_args()

    candidatas, avisos = [], []
    for fuente, prep in (("realwaste", lambda: preparar_zip("realwaste")), ("taco_extra", preparar_taco_extra),
                         ("trashnet", lambda: preparar_zip("trashnet"))):
        if fuente in args.excluir:
            print(f"[{fuente}] excluida con --excluir")
            continue
        c, a = prep()
        avisos += a
        candidatas += c
    for a in avisos:
        print(a)
    nombres = Counter(c["nombre"] for c in candidatas)
    if (rep := [n for n, k in nombres.items() if k > 1]):
        raise SystemExit(f"Nombres de destino repetidos: {rep[:5]}")

    # --- anti-leakage: publico y juliaca (todos los splits), val_camara (nunca se entrena: auditar.py da ERROR si
    # algo de train se le parece) y externo/train que no es de este script ---
    propios = tuple(f"{p}_" for p in PREFIJO.values())
    refs = [p for base in (PUBLICO, JULIACA) for s in SPLITS for p in imagenes(base / s)]
    refs += imagenes(VAL_CAMARA)
    refs += [p for p in imagenes(TRAIN) if not p.name.startswith(propios)]
    print(f"\ndHash de {len(refs)} referencias (publico + juliaca, todos los splits, + val_camara + externo/train "
          "previo)...", flush=True)
    ref_h = np.array([dhash(p) for p in refs], dtype=np.uint64)
    orden = {f: i for i, f in enumerate(PRIORIDAD)}
    candidatas.sort(key=lambda c: (orden[c["fuente"]], c["nombre"]))
    print(f"dHash de {len(candidatas)} candidatas...", flush=True)
    aceptadas_h, aceptadas = [], []
    for c in candidatas:
        try:
            h = dhash(c["ruta"])
        except Exception as e:  # noqa: BLE001
            c.update(dist_ref="", mas_cercano="", motivo=f"ilegible: {e}")
            continue
        dr = distancias(ref_h, h)
        j = int(dr.argmin())
        c["dist_ref"], c["mas_cercano"] = int(dr[j]), refs[j].relative_to(RAIZ).as_posix()
        if dr[j] <= UMBRAL_DHASH:
            c["motivo"] = "casi_igual_a_" + refs[j].relative_to(RAIZ).parts[1]  # publico / juliaca / externo
            continue
        da = distancias(np.array(aceptadas_h, dtype=np.uint64), h)
        if len(da) and da.min() <= UMBRAL_DHASH:
            otra = aceptadas[int(da.argmin())]
            c["motivo"] = f"duplicado_de_{otra['fuente']}:{otra['nombre']}"
            continue
        c["motivo"] = "ok"
        aceptadas_h.append(h)
        aceptadas.append(c)

    # --- revisión manual de RealWaste Metal (después de deduplicar, antes del tope) ---
    terminado = True
    if "realwaste" not in args.excluir:
        preparar_revision([c for c in aceptadas if c["fuente"] == "realwaste" and c["categoria"] == "Metal"])
        cambios, terminado = leer_revision()
        for c in [c for c in aceptadas if c["nombre"] in cambios]:
            nueva = cambios[c["nombre"]]
            if nueva == "descartar":
                c["motivo"] = "descartado_revision_usuario"
                aceptadas.remove(c)
            else:
                c["clase"], c["categoria"] = nueva, f"Metal (revisión del usuario: {nueva})"
        print(f"Revisión RealWaste Metal: {len(cambios)} anotaciones "
              f"({Counter(cambios.values()).most_common()}); terminado: {'si' if terminado else 'no'}")
    if not args.dry_run and not terminado:
        raise SystemExit(f"La revisión de RealWaste Metal no está terminada ({REV_TXT.relative_to(RAIZ)}: "
                         "'terminado: si'). No se importa nada.")

    # --- tope por clase, por prioridad; dentro de cada fuente, muestra con semilla fija ---
    por_fc = defaultdict(list)
    for c in aceptadas:
        por_fc[c["fuente"], c["clase"]].append(c)
    elegidas = []
    for clase in CLASES:
        cupo = args.tope
        for fuente in PRIORIDAD:
            grupo = sorted(por_fc[fuente, clase], key=lambda c: c["nombre"])
            random.Random(f"{args.semilla}-{fuente}-{clase}").shuffle(grupo)  # independiente de los demás grupos
            for c in grupo[:cupo]:
                c["elegida"] = True
                elegidas.append(c)
            for c in grupo[cupo:]:
                c["motivo"] = "fuera_del_tope"
            cupo -= min(cupo, len(grupo))

    # --- informes ---
    INF_DEDUP.parent.mkdir(exist_ok=True)
    with open(INF_DEDUP, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fuente", "categoria_origen", "clase", "archivo_destino", "origen", "dist_min_ref",
                    "mas_cercano_ref", "motivo", "elegida"])
        for c in candidatas:
            w.writerow([c["fuente"], c["categoria"], c["clase"], c["nombre"], c["ruta"].relative_to(RAIZ).as_posix(),
                        c.get("dist_ref", ""), c.get("mas_cercano", ""), c["motivo"], "SI" if c.get("elegida") else "NO"])
    cont = defaultdict(Counter)  # (clase, fuente) -> candidatas / tras_dedup / elegidas
    for c in candidatas:
        k = (c["clase"], c["fuente"])
        cont[k]["candidatas"] += 1
        cont[k]["tras_dedup"] += c["motivo"] in ("ok", "fuera_del_tope")
        cont[k]["elegidas"] += bool(c.get("elegida"))
    with open(INF_CONTEOS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", "fuente", "candidatas", "tras_dedup", "elegidas_tras_tope"])
        for clase in CLASES:
            for fuente in PRIORIDAD:
                k = cont[clase, fuente]
                if k:
                    w.writerow([clase, fuente, k["candidatas"], k["tras_dedup"], k["elegidas"]])

    print(f"\n{'clase':13s}" + "".join(f"{f:>24s}" for f in PRIORIDAD) + f"{'TOTAL nuevas':>14s}{'externo previo':>16s}")
    print(f"{'':13s}" + "".join(f"{'cand/dedup/tope':>24s}" for _ in PRIORIDAD))
    previo = {c: len([p for p in imagenes(TRAIN / c) if not p.name.startswith(propios)]) for c in CLASES}
    for clase in CLASES:
        celdas = "".join(f"{cont[clase, f]['candidatas']:>10d}/{cont[clase, f]['tras_dedup']:>5d}/"
                         f"{cont[clase, f]['elegidas']:>5d}  " for f in PRIORIDAD)
        print(f"{clase:13s}{celdas}{sum(cont[clase, f]['elegidas'] for f in PRIORIDAD):>12d}{previo[clase]:>16d}")
    motivos = Counter(c["motivo"].split(":")[0] for c in candidatas)
    print("\nmotivos:", dict(motivos))

    if MOSAICOS.exists():
        shutil.rmtree(MOSAICOS)
    MOSAICOS.mkdir(parents=True)
    rng_m = random.Random(args.semilla)
    for fuente in PRIORIDAD:
        for clase in CLASES:
            grupo = [c for c in elegidas if c["fuente"] == fuente and c["clase"] == clase]
            if grupo:
                rng_m.shuffle(grupo)
                cats = Counter(c["categoria"] for c in grupo)
                titulo = f"{fuente} -> {clase}  ({len(grupo)} elegidas; " + ", ".join(f"{k} {v}" for k, v in cats.most_common()) + ")"
                mosaico([c["ruta"] for c in grupo], MOSAICOS / f"{fuente}__{clase}.jpg", titulo[:200])
    print(f"\nInformes: {INF_DEDUP.relative_to(RAIZ)}, {INF_CONTEOS.relative_to(RAIZ)}, mosaicos en "
          f"{MOSAICOS.relative_to(RAIZ)}/")

    if args.dry_run:
        print("DRY-RUN: no se tocó dataset/externo/train ni datos_externos.csv.")
        return

    # --- copia (solo archivos propios se reemplazan) y registro ---
    for p in imagenes(TRAIN):
        if p.name.startswith(propios):
            p.unlink()
    for c in elegidas:
        destino = TRAIN / c["clase"] / c["nombre"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            raise SystemExit(f"Ya existe {destino}: no se sobrescribe")
        shutil.copy2(c["ruta"], destino)
    hoy = date.today().isoformat()
    campos = ["fuente", "url", "licencia", "fecha", "categoria_origen", "clase_destino", "n_imagenes", "archivos"]
    with open(REGISTRO, newline="", encoding="utf-8") as f:
        otras = [r for r in csv.DictReader(f) if not r["fuente"].startswith(tuple(f"{x}:" for x in PRIORIDAD))]
    grupos = defaultdict(list)
    for c in elegidas:
        grupos[c["fuente"], c["url"] if c["fuente"] == "taco_extra" else "", c["categoria"], c["clase"]].append(c)
    nuevas = [{"fuente": f"{fu}: {CITA[fu]}", "url": g[0]["url"], "licencia": g[0]["licencia"], "fecha": hoy,
               "categoria_origen": cat, "clase_destino": cl, "n_imagenes": len(g),
               "archivos": ";".join(sorted(c["nombre"] for c in g))}
              for (fu, _, cat, cl), g in sorted(grupos.items())]
    with open(REGISTRO, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(otras + nuevas)
    print(f"Copiadas {len(elegidas)} imágenes a {TRAIN.relative_to(RAIZ)} y {len(nuevas)} filas en {REGISTRO.name}.")
    print("Ahora: python scripts/auditar.py")


if __name__ == "__main__":
    main()
