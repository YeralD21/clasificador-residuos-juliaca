"""Reconstruye las 238 imágenes a color de garbage.v1i (test del benchmark) desde TrashNet.

El zip garbage.v1i.folder del docente no está disponible ("Garbage Collection.v1i" es otro proyecto). Los nombres del
benchmark son de TrashNet (p. ej. cardboard101_jpg.rf.<hash>.jpg -> cardboard101.jpg). Para cada fila de
garbage.v1i en pred_01_best_jhoel.csv:
  1. toma el original de dataset/externo/_fuentes/trashnet/extraido/dataset-resized/<clase>/<original>.jpg (solo lee);
  2. lo redimensiona (stretch) al tamaño de su versión en gris del docente (640x640);
  3. lo verifica contra esa versión en gris: dHash (mínimo sobre las 8 rotaciones/espejos) o SSIM en escala de grises
     (en la orientación del mejor dHash);
  4. lo guarda en imagenes/_descargas/v1i/test/<clase>/<nombre del benchmark> (preparar_test.py lo indexa de ahí).
Las imágenes cuya verificación no pasa, o que el usuario marca como dañadas (--sin-verificar), quedan con estado
"sin verificar" en salidas/reconstruccion_v1i.csv.

python -I evaluacion_docente/reconstruir_v1i.py [--sin-verificar lista.txt]
No toca dataset/ (solo lee TrashNet). No sobrescribe imágenes ya reconstruidas salvo con --forzar.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

TRASHNET = RAIZ / "dataset" / "externo" / "_fuentes" / "trashnet" / "extraido" / "dataset-resized"
GRIS = AQUI / "imagenes" / "gris_docente_garbage_v1i"
DEST = AQUI / "imagenes" / "_descargas" / "v1i" / "test"
OUT = AQUI / "salidas"
UMBRAL_DHASH = 4
UMBRAL_SSIM = 0.80


def dhash(g, tam=8):
    a = np.asarray(Image.fromarray(g).resize((tam + 1, tam), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, :-1] > a[:, 1:]).flatten()), 2)


def variantes(a):
    """Las 8 transformaciones del cuadrado (rotaciones de 90° y espejo)."""
    for k in range(4):
        r = np.rot90(a, k)
        yield f"rot{90 * k}", r
        yield f"rot{90 * k}_espejo", np.fliplr(r)


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    s_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    s_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    s_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_a + s_b + c2))
    return float(m.mean())


def original_de(archivo):
    return archivo.split("_jpg.rf.")[0] + ".jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-verificar", default=None,
                    help="txt con nombres del benchmark (uno por línea) que llegaron dañados: se marcan 'sin verificar'")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    danadas = set()
    if a.sin_verificar:
        danadas = {l.strip() for l in open(a.sin_verificar, encoding="utf-8") if l.strip()}

    filas = [f for f in nm.leer_csv(AQUI / "datos_docente" / "pred_01_best_jhoel.csv")
             if f["dataset"].startswith("garbage.v1i")]
    res = []
    for f in filas:
        partes = f["path"].replace("\\", "/").split("/")
        archivo, clase = partes[-1], partes[-2]
        orig = TRASHNET / clase / original_de(archivo)
        gris_p = GRIS / clase / archivo
        r = {"clase": clase, "archivo": archivo, "original_trashnet": f"{clase}/{orig.name}",
             "grupo": nm.grupo_de(f), "dhash": "", "orientacion": "", "ssim": "", "estado": ""}
        if not orig.is_file():
            r["estado"] = "sin original"
            res.append(r)
            continue
        with Image.open(gris_p) as g:
            tam = g.size
            g_arr = np.asarray(g.convert("L"))
        with Image.open(orig) as im:
            color = im.convert("RGB").resize(tam, Image.BILINEAR)
        c_gris = np.asarray(color.convert("L"))
        hg = dhash(g_arr)
        mejor = min(((bin(dhash(np.ascontiguousarray(v)) ^ hg).count("1"), n, v) for n, v in variantes(c_gris)),
                    key=lambda x: x[0])
        r["dhash"], r["orientacion"] = mejor[0], mejor[1]
        r["ssim"] = round(ssim(np.ascontiguousarray(mejor[2]), g_arr), 4)
        ok = mejor[1] == "rot0" and (mejor[0] <= UMBRAL_DHASH or r["ssim"] >= UMBRAL_SSIM)
        r["estado"] = "sin verificar (dañada)" if archivo in danadas else ("verificada" if ok else "sin verificar")
        dst = DEST / clase / archivo
        dst.parent.mkdir(parents=True, exist_ok=True)
        if a.forzar or not dst.exists():
            color.save(dst, quality=95)
        res.append(r)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "reconstruccion_v1i.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    resumen = {"n": len(res), "umbral_dhash": UMBRAL_DHASH, "umbral_ssim": UMBRAL_SSIM,
               "estado": dict(Counter(r["estado"] for r in res)),
               "por_grupo": {g: dict(Counter(r["estado"] for r in res if r["grupo"] == g))
                             for g in sorted({r["grupo"] for r in res})},
               "orientacion": dict(Counter(r["orientacion"] for r in res))}
    (OUT / "reconstruccion_v1i.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
