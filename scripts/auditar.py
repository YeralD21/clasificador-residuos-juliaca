"""Auditoría ANTES de entrenar. No modifica nada, solo reporta.

  python scripts/auditar.py

Revisa en dataset/juliaca (tu partición manual):
  - Conteo de fotos y de OBJETOS por clase y split.
  - LEAKAGE: el mismo objeto (subcarpeta) en dos splits distintos.
  - LEAKAGE: fotos idénticas (md5) o casi idénticas (hash perceptual) en splits distintos.
  - Fotos sueltas sin carpeta de objeto (no se puede verificar leakage).
  - Carpetas que no son una de las 8 clases, imágenes corruptas o muy pequeñas.
Y en dataset/publico: conteos, duplicados entre splits y choques con juliaca.
Termina con código 1 si hay ERRORES (para que no entrenes con leakage).
"""
import sys
from collections import defaultdict
from itertools import combinations

from PIL import Image

from comun import CLASES, JULIACA, PUBLICO, SPLITS, dhash, imagenes, md5, objeto_de

UMBRAL_DHASH = 4  # bits distintos (de 64) para considerar dos fotos "casi iguales"


def revisar_base(base, nombre, errores, avisos):
    conteo, md5s = defaultdict(dict), defaultdict(dict)
    for s in SPLITS:
        d = base / s
        if not d.exists():
            errores.append(f"[{nombre}] falta la carpeta {s}/")
            continue
        for sub in d.iterdir():
            if sub.is_dir() and sub.name not in CLASES:
                errores.append(f"[{nombre}] carpeta desconocida: {sub}")
            elif sub.is_file() and sub.name != ".gitkeep":
                avisos.append(f"[{nombre}] archivo fuera de una clase (se ignora): {sub}")
        for c in CLASES:
            fotos = imagenes(d / c)
            conteo[s][c] = fotos
            for p in fotos:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        if min(im.size) < 64:
                            avisos.append(f"[{nombre}] imagen muy pequeña {im.size}: {p}")
                except Exception as e:  # noqa: BLE001
                    errores.append(f"[{nombre}] imagen corrupta: {p} ({e})")
                    continue
                md5s[s][p] = md5(p)
    for a, b in combinations(SPLITS, 2):
        comunes = set(md5s[a].values()) & set(md5s[b].values())
        if comunes:
            errores.append(f"[{nombre}] {len(comunes)} fotos IDÉNTICAS entre {a} y {b} (leakage)")
    return conteo, md5s


def main():
    errores, avisos = [], []
    cj, mj = revisar_base(JULIACA, "juliaca", errores, avisos)
    cp, mp = revisar_base(PUBLICO, "publico", errores, avisos)

    # --- objetos en juliaca ---
    objetos = defaultdict(lambda: defaultdict(set))  # clase -> objeto -> {splits}
    sueltas = 0
    for s in SPLITS:
        for c in CLASES:
            for p in cj[s][c]:
                o = objeto_de(p, JULIACA / s / c)
                if o is None:
                    sueltas += 1
                else:
                    objetos[c][o].add(s)
    for c, objs in objetos.items():
        for o, ss in objs.items():
            if len(ss) > 1:
                errores.append(f"[juliaca] LEAKAGE: objeto '{c}/{o}' aparece en {sorted(ss)}")
    if sueltas:
        avisos.append(f"[juliaca] {sueltas} fotos sueltas sin carpeta de objeto: muévelas a <clase>/objNNN_nombre/ "
                      "para poder verificar que un objeto no esté en dos splits")

    # --- casi duplicados entre splits de juliaca ---
    hashes = {s: [(p, dhash(p)) for p in mj[s]] for s in SPLITS}
    for a, b in combinations(SPLITS, 2):
        for pa, ha in hashes[a]:
            for pb, hb in hashes[b]:
                if bin(ha ^ hb).count("1") <= UMBRAL_DHASH:
                    avisos.append(f"[juliaca] casi iguales entre {a} y {b}: {pa.relative_to(JULIACA)} <-> {pb.relative_to(JULIACA)}")

    # --- choques publico <-> juliaca ---
    tj = {h for d in mj.values() for h in d.values()}
    tp = {h for d in mp.values() for h in d.values()}
    if tj & tp:
        errores.append(f"{len(tj & tp)} fotos están a la vez en publico y juliaca")

    # --- tablas ---
    print("=== dataset/juliaca (tu partición manual) ===")
    print(f"{'clase':14s} {'train':>11s} {'val':>11s} {'test':>11s}   (fotos/objetos)")
    for c in CLASES:
        celdas = []
        for s in SPLITS:
            n_obj = sum(1 for o, ss in objetos[c].items() if s in ss)
            celdas.append(f"{len(cj[s][c]):5d}/{n_obj:<5d}")
        print(f"{c:14s} " + " ".join(celdas))
        if not cj["test"][c]:
            avisos.append(f"[juliaca] clase '{c}' sin fotos en test")

    print("\n=== dataset/publico ===")
    print(f"{'clase':14s} {'train':>7s} {'val':>6s} {'test':>6s}")
    for c in CLASES:
        print(f"{c:14s} {len(cp['train'][c]):7d} {len(cp['val'][c]):6d} {len(cp['test'][c]):6d}")

    print(f"\n=== ERRORES ({len(errores)}) ===")
    for e in errores[:60]:
        print(" X", e)
    print(f"\n=== AVISOS ({len(avisos)}) ===")
    for a in avisos[:60]:
        print(" !", a)
    if len(avisos) > 60:
        print(f" ... y {len(avisos) - 60} más")
    if errores:
        print("\nCorrige los ERRORES antes de entrenar.")
        sys.exit(1)
    print("\nSin errores. Puedes entrenar.")


if __name__ == "__main__":
    main()
