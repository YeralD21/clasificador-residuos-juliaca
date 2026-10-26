"""Auditoría ANTES de entrenar. No modifica nada, solo reporta.

  python scripts/auditar.py

Revisa en dataset/juliaca (tu partición manual):
  - Conteo de fotos y de OBJETOS por clase y split.
  - LEAKAGE: el mismo objeto (subcarpeta) en dos splits distintos.
  - LEAKAGE: fotos idénticas (md5) o casi idénticas (hash perceptual) en splits distintos.
  - Fotos sueltas sin carpeta de objeto (no se puede verificar leakage).
  - Carpetas que no son una de las 8 clases, imágenes corruptas o muy pequeñas.
Y en dataset/publico: conteos, duplicados entre splits y choques con juliaca.
Y en dataset/externo/train (regla 1): conteo por clase, formatos, que no haya val/test,
que cada archivo esté en datos_externos.csv y casi duplicados (dHash <= 4) contra TODOS los
splits de publico y juliaca. Las carpetas que empiezan con "_" (_fuentes, _descartados...) se ignoran.
Y en dataset/val_camara (evaluación aparte, NUNCA se entrena): clases y carpetas de objeto, y que
ninguna foto esté en lo que se entrena. Contra juliaca y publico (todos los splits) y todo externo:
  - foto idéntica (md5) en cualquier parte -> ERROR.
  - casi igual (dHash <= 4) a una foto de train -> ERROR; de val/test o externo/_* -> AVISO.
  - los pares revisados a ojo como falsos positivos (reportes/val_camara_falsos_positivos.csv)
    quedan como AVISO. Informe completo: reportes/val_camara_dedup.csv.
_etapa1 y _etapa2 se regeneran al entrenar solo desde juliaca, publico y externo/train, así que basta con esos.
Termina con código 1 si hay ERRORES (para que no entrenes con leakage).
"""
import csv
import sys
from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np
from PIL import Image

from comun import (CLASES, EXT, EXTERNO, JULIACA, PUBLICO, RAIZ, SPLITS, VAL_CAMARA, imagenes, md5,
                   objeto_de)
from comun import dhash as _dhash

UMBRAL_DHASH = 4  # bits distintos (de 64) para considerar dos fotos "casi iguales"
INFORME_VAL_CAMARA = RAIZ / "reportes" / "val_camara_dedup.csv"
FALSOS_POSITIVOS = RAIZ / "reportes" / "val_camara_falsos_positivos.csv"
TABLA_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
dhash = lru_cache(maxsize=None)(_dhash)  # externo y val_camara comparan contra las mismas referencias


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


def revisar_externo(mj, mp, errores, avisos):
    """dataset/externo: solo train, clases válidas, formatos, registro y anti-leakage contra publico y juliaca."""
    conteo = {c: [] for c in CLASES}
    if not EXTERNO.exists():
        return conteo
    for sub in EXTERNO.iterdir():
        if sub.name.startswith("_") or sub.name == ".gitkeep":
            continue
        if sub.name in ("val", "test") and imagenes(sub):
            errores.append(f"[externo] hay imágenes en {sub.name}/: los datos externos van SOLO a train")
        elif sub.name not in SPLITS:
            avisos.append(f"[externo] carpeta o archivo inesperado (se ignora): {sub}")
    tr = EXTERNO / "train"
    if not tr.exists():
        return conteo
    for sub in tr.iterdir():
        if sub.is_dir() and sub.name not in CLASES:
            errores.append(f"[externo] carpeta desconocida: {sub}")
        elif sub.is_file() and sub.name != ".gitkeep":
            avisos.append(f"[externo] archivo fuera de una clase (se ignora): {sub}")
    registrados = set()
    reg = RAIZ / "datos_externos.csv"
    if reg.exists():
        with open(reg, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("clase_destino") != "descartado":
                    registrados |= set(filter(None, (r.get("archivos") or "").split(";")))
    else:
        errores.append("[externo] falta datos_externos.csv")
    for c in CLASES:
        for p in sorted(x for x in (tr / c).rglob("*") if x.is_file()) if (tr / c).exists() else []:
            if p.suffix.lower() not in EXT:
                avisos.append(f"[externo] formato no soportado (se ignora): {p}")
                continue
            try:
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    if min(im.size) < 64:
                        avisos.append(f"[externo] imagen muy pequeña {im.size}: {p}")
            except Exception as e:  # noqa: BLE001
                errores.append(f"[externo] imagen corrupta: {p} ({e})")
                continue
            if p.name not in registrados:
                errores.append(f"[externo] {p.relative_to(EXTERNO)} no está registrado en datos_externos.csv")
            conteo[c].append(p)
    fotos = [p for c in CLASES for p in conteo[c]]
    if not fotos:
        return conteo
    refs = [p for m in (mj, mp) for s in SPLITS for p in m[s]]
    print(f"[externo] dHash de {len(fotos)} imágenes externas contra {len(refs)} de publico + juliaca...", flush=True)
    ref_h = np.array([dhash(p) for p in refs], dtype=np.uint64)
    tabla = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    for p in fotos:
        x = (ref_h ^ np.uint64(dhash(p))).view(np.uint8).reshape(-1, 8)
        dist = tabla[x].sum(axis=1)
        j = int(dist.argmin())
        if dist[j] <= UMBRAL_DHASH:
            errores.append(f"[externo] LEAKAGE: {p.relative_to(EXTERNO)} casi igual (dHash {dist[j]}) a "
                           f"{refs[j].relative_to(RAIZ)}")
    return conteo


def revisar_val_camara(mj, mp, errores, avisos):
    """dataset/val_camara/<clase>/<objeto>/: que ninguna foto esté (igual o casi igual) en juliaca, publico o externo."""
    conteo = {c: [] for c in CLASES}
    if not VAL_CAMARA.exists():
        avisos.append("[val_camara] no existe dataset/val_camara (se omite el chequeo)")
        return conteo
    for sub in VAL_CAMARA.iterdir():
        if sub.is_dir() and sub.name not in CLASES:
            errores.append(f"[val_camara] carpeta desconocida: {sub}")
        elif sub.is_file() and sub.name != ".gitkeep":
            avisos.append(f"[val_camara] archivo fuera de una clase (se ignora): {sub}")
    for c in CLASES:
        for p in imagenes(VAL_CAMARA / c):
            if objeto_de(p, VAL_CAMARA / c) is None:
                avisos.append(f"[val_camara] foto suelta sin carpeta de objeto: {p.relative_to(VAL_CAMARA)}")
            conteo[c].append(p)
    fotos = [p for c in CLASES for p in conteo[c]]
    if not fotos:
        return conteo

    # referencias: (grupo, es_train, ruta). En externo solo train/ se entrena; _fuentes, _descartados... no.
    refs = [(f"{n}/{s}", s == "train", p) for n, m in (("juliaca", mj), ("publico", mp)) for s in SPLITS for p in m[s]]
    refs += [("externo/" + p.relative_to(EXTERNO).parts[0], p.relative_to(EXTERNO).parts[0] == "train", p)
             for p in imagenes(EXTERNO)]
    ya = {p: h for m in (mj, mp) for d in m.values() for p, h in d.items()}  # md5 ya calculados
    md5_ref = {}
    for g, _, q in refs:
        md5_ref.setdefault(ya.get(q) or md5(q), (g, q))
    for p in fotos:
        h = md5(p)
        if h in md5_ref:
            g, q = md5_ref[h]
            errores.append(f"[val_camara] LEAKAGE: {p.relative_to(VAL_CAMARA)} es IDÉNTICA a {q.relative_to(RAIZ)} ({g})")

    revisados = set()
    if FALSOS_POSITIVOS.exists():
        with open(FALSOS_POSITIVOS, newline="", encoding="utf-8") as f:
            revisados = {(r["foto_val_camara"], r["referencia"]) for r in csv.DictReader(f)}
    print(f"[val_camara] dHash de {len(fotos)} fotos contra {len(refs)} de juliaca + publico + externo...", flush=True)
    ref_h = np.array([dhash(p) for _, _, p in refs], dtype=np.uint64)
    filas = []
    for p in fotos:
        dist = TABLA_BITS[(ref_h ^ np.uint64(dhash(p))).view(np.uint8).reshape(-1, 8)].sum(axis=1)
        cerca = np.flatnonzero(dist <= UMBRAL_DHASH)
        rp = p.relative_to(RAIZ).as_posix()
        for k in cerca:
            g, es_train, q = refs[k]
            rq = q.relative_to(RAIZ).as_posix()
            msg = f"{p.relative_to(VAL_CAMARA)} casi igual (dHash {dist[k]}) a {rq} ({g})"
            if (rp, rq) in revisados:
                avisos.append(f"[val_camara] {msg}: revisado, falso positivo")
            elif es_train:
                errores.append(f"[val_camara] LEAKAGE: {msg}. val_camara NUNCA se entrena")
            else:
                avisos.append(f"[val_camara] {msg}")
        j = int(dist.argmin())
        filas.append([rp, int(dist[j]), refs[j][2].relative_to(RAIZ).as_posix(), refs[j][0], len(cerca)])
    with open(INFORME_VAL_CAMARA, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto_val_camara", "dist_min", "mas_cercano", "base_mas_cercano", "n_refs_dist_le_4"])
        w.writerows(filas)
    print(f"[val_camara] informe: {INFORME_VAL_CAMARA.relative_to(RAIZ)}", flush=True)
    return conteo


def main():
    errores, avisos = [], []
    cj, mj = revisar_base(JULIACA, "juliaca", errores, avisos)
    cp, mp = revisar_base(PUBLICO, "publico", errores, avisos)
    ce = revisar_externo(mj, mp, errores, avisos)
    cv = revisar_val_camara(mj, mp, errores, avisos)

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

    print("\n=== dataset/externo (solo train) ===")
    print(f"{'clase':14s} {'train':>7s}")
    for c in CLASES:
        print(f"{c:14s} {len(ce[c]):7d}")

    print("\n=== dataset/val_camara (evaluación aparte, nunca se entrena) ===")
    print(f"{'clase':14s} {'fotos':>6s} {'objetos':>8s}")
    for c in CLASES:
        n_obj = len({objeto_de(p, VAL_CAMARA / c) for p in cv[c]} - {None})
        print(f"{c:14s} {len(cv[c]):6d} {n_obj:8d}")

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
