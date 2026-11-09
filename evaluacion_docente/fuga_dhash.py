"""Chequeo de fuga: ¿hay imágenes del test del docente (casi) repetidas en mis datos de
entrenamiento? dHash 8x8 (igual que scripts/comun.py), umbral <= 4. Solo LEE, no modifica nada.

Fuentes de entrenamiento comparadas:
  - zip:      dataset/crudo/dataset_ntp900058.zip (origen de dataset/publico; se lee en memoria)
  - publico:  dataset/publico/**     - externo: dataset/externo/train/**     - juliaca: dataset/juliaca/**
Imágenes de test: imagenes/test_docente/** (las que haya) y las 238 en gris del docente.

python -I evaluacion_docente/fuga_dhash.py
Salida: salidas/fuga_dhash.csv (todas las coincidencias) y salidas/fuga_dhash.json (resumen por grupo)
"""
import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

OUT = AQUI / "salidas"
UMBRAL = 4
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
FUENTES = ("publico", "externo", "juliaca")


def dhash_img(im, tam=8):
    g = np.asarray(im.convert("L").resize((tam + 1, tam), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, :-1] > g[:, 1:]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def entrenamiento():
    zp = RAIZ / "dataset" / "crudo" / "dataset_ntp900058.zip"
    if zp.is_file():
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.lower().endswith(EXT):
                    with Image.open(io.BytesIO(z.read(n))) as im:
                        yield "zip", n, dhash_img(im)
    for sub in FUENTES:
        base = RAIZ / "dataset" / sub
        # externo: solo externo/train (lo que entra al entrenamiento); _fuentes, _revision y _descartados no se usan
        raiz_sub = base / "train" if sub == "externo" else base
        for p in raiz_sub.rglob("*"):
            if p.suffix.lower() in EXT and not p.name.startswith("._"):
                try:
                    with Image.open(p) as im:
                        yield sub, p.relative_to(base).as_posix(), dhash_img(im)
                except OSError as e:
                    print(f"  aviso: no se pudo leer {p} ({type(e).__name__})")


def test():
    """(grupo, nombre, hash) de cada imagen de test disponible."""
    grupo_por_archivo = {}
    for f in nm.leer_csv(AQUI / "datos_docente" / "pred_01_best_jhoel.csv"):
        grupo_por_archivo[f["path"].replace("\\", "/").split("/")[-1]] = nm.grupo_de(f)
    for carpeta, pref in ((AQUI / "imagenes" / "test_docente", ""),
                          (AQUI / "imagenes" / "gris_docente_garbage_v1i", "[gris] ")):
        for p in carpeta.rglob("*"):
            if p.suffix.lower() in EXT:
                with Image.open(p) as im:
                    yield pref + grupo_por_archivo[p.name], p.name, dhash_img(im)


def main():
    OUT.mkdir(exist_ok=True)
    tst = list(test())
    tr = list(entrenamiento())
    n_fuente = Counter(f for f, _, _ in tr)
    H = np.array([h for _, _, h in tr], dtype=np.uint64)
    hits = []
    for g, n, h in tst:
        d = np.bitwise_count(H ^ np.uint64(h))
        i = int(d.argmin())
        if d[i] <= UMBRAL:
            fuentes = sorted({tr[j][0] for j in np.flatnonzero(d <= UMBRAL)})
            hits.append({"grupo": g, "test": n, "train_fuente": tr[i][0], "train": tr[i][1],
                         "dist": int(d[i]), "fuentes_con_coincidencia": "|".join(fuentes)})
    with open(OUT / "fuga_dhash.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hits[0]) if hits else ["grupo"])
        w.writeheader()
        w.writerows(hits)
    n_g = Counter(g for g, _, _ in tst)
    h_g = Counter(h["grupo"] for h in hits)
    h_gf = Counter((h["grupo"], h["train_fuente"]) for h in hits)
    res = {"umbral": UMBRAL, "n_train_por_fuente": {k: n_fuente.get(k, 0) for k in ("zip",) + FUENTES},
           "n_test": len(tst), "n_coincidencias": len(hits),
           "por_grupo": {g: {"n": n_g[g], "coinciden": h_g[g],
                             "por_fuente": {k: h_gf[(g, k)] for k in ("zip",) + FUENTES if h_gf[(g, k)]}}
                         for g in sorted(n_g, key=lambda x: (x.startswith("["), x))}}
    (OUT / "fuga_dhash.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
