"""Importa a dataset/externo/train lo aprobado por el usuario (2026-11-11) de la Fase A de candidatas_tiendas.py.

  python -I scripts/importar_tiendas.py               # DRY-RUN: muestra qué copiaría
  python -I scripts/importar_tiendas.py --importar    # copia y agrega filas a datos_externos.csv

Fuentes (aprobadas por el usuario, regla 1a):
  frg_ : Freiburg Groceries Dataset. Se recortan las bandas grises (128,128,128) y se guarda JPG calidad 95.
  gro_ : Grocery Store Dataset. Se copia el JPG original sin modificar.
Qué se importa: las filas con motivo 'ok' de reportes/externo_candidatas3_dedup.csv (Fase A: envase por foto,
deduplicado contra el benchmark y los datos propios, sin casi duplicados internos, tope 150 por fuente y clase).
Excepción, CEREAL -> papel_carton: decisión del usuario, solo cajas de cartón. Se importan exactamente las 50 de
reportes/externo_candidatas3_cereal_cajas.txt (ok o fuera_del_tope en la Fase A).
Garantías: solo escribe en dataset/externo/train/<clase>/ (regla 1c); nunca sobrescribe ni borra; si ya hay algún
archivo frg_ o gro_ no hace nada; agrega filas a datos_externos.csv sin tocar las existentes.
"""
import argparse
import csv
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES, EXTERNO, RAIZ, imagenes  # noqa: E402
from candidatas_tiendas import quitar_relleno  # noqa: E402

DEDUP = RAIZ / "reportes" / "externo_candidatas3_dedup.csv"
CEREAL = RAIZ / "reportes" / "externo_candidatas3_cereal_cajas.txt"
REGISTRO = RAIZ / "datos_externos.csv"
DEDUP_TXT = ("deduplicado contra el benchmark NTP 900.058 (nombre, dHash<=4 con 8 giros/espejos y ORB) y contra "
             "todos los datos propios; sin casi duplicados internos (dHash<=4 y embedding coseno>=0.95); "
             "informe reportes/externo_candidatas3_{dedup,conteos}.csv")
FUENTES = {
    "frg": dict(
        prefijo="frg_",
        fuente="frg: Freiburg Groceries Dataset. Jund, P., Abdo, N., Eitel, A. y Burgard, W. (2016). The Freiburg "
               "Groceries Dataset. arXiv:1611.05799. Envase decidido por foto (CLIP ViT-B/32 + revisión visual), "
               "bandas grises recortadas; " + DEDUP_TXT,
        url="https://github.com/PhilJd/freiburg_groceries_dataset ; "
            "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/",
        licencia="Sin licencia formal. Condición de uso publicada por los autores: \"It is our hope that machine "
                 "learning and robotics researchers find this dataset of use for training, testing, and "
                 "bootstrapping their approaches.\""),
    "gro": dict(
        prefijo="gro_",
        fuente="gro: Grocery Store Dataset. Klasson, M., Zhang, C. y Kjellström, H. (2019). A Hierarchical Grocery "
               "Store Image Dataset with Visual and Semantic Labels. WACV 2019. Solo imágenes naturales; " + DEDUP_TXT,
        url="https://github.com/marcusklasson/GroceryStoreDataset",
        licencia="MIT (Copyright (c) 2019 Marcus Klasson; archivo LICENSE del repositorio)"),
}


def nombre_destino(f):
    ruta = Path(f["ruta_zip"])
    if f["fuente"] == "frg":
        return f"frg_{f['categoria']}_{ruta.stem}.jpg"
    return f"gro_{f['split']}_{f['categoria']}_{ruta.stem}.jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--importar", action="store_true")
    a = ap.parse_args()

    filas = list(csv.DictReader(open(DEDUP, encoding="utf-8")))
    for f in filas:  # en la Fase A Grocery se llamaba 'gsd'; el prefijo acordado es gro_
        if f["fuente"] == "gsd":
            f["fuente"] = "gro"
    cereal = {x.strip() for x in CEREAL.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")}
    if len(cereal) != 50:
        sys.exit(f"ERROR: {CEREAL.name} debe tener 50 nombres ({len(cereal)})")
    sel = []
    for f in filas:
        if f["fuente"] == "frg" and f["categoria"] == "CEREAL":
            if Path(f["ruta_zip"]).stem in cereal:
                if f["motivo"] not in ("ok", "fuera_del_tope") or f["clase"] != "papel_carton":
                    sys.exit(f"ERROR: {f['ruta_zip']} no estaba limpia en la Fase A ({f['motivo']})")
                sel.append(f)
        elif f["motivo"] == "ok":
            sel.append(f)
    n_cereal = sum(1 for f in sel if f["categoria"] == "CEREAL")
    if n_cereal != 50:
        sys.exit(f"ERROR: se esperaban 50 de CEREAL y hay {n_cereal}")
    for f in sel:
        if f["clase"] not in CLASES:
            sys.exit(f"ERROR: clase inválida {f['clase']} en {f['ruta_zip']}")
        f["destino"] = EXTERNO / "train" / f["clase"] / nombre_destino(f)

    destinos = [f["destino"] for f in sel]
    if len(set(destinos)) != len(destinos):
        sys.exit("ERROR: nombres de destino repetidos")
    previos = [p for c in CLASES for p in imagenes(EXTERNO / "train" / c) if p.name.startswith(("frg_", "gro_"))]
    if previos or any(d.exists() for d in destinos):
        sys.exit(f"ERROR: ya hay archivos frg_/gro_ en externo/train ({len(previos)}); no se sobrescribe nada.")

    agg = defaultdict(list)
    for f in sel:
        agg[(f["fuente"], f["categoria"], f["clase"])].append(f["destino"].name)
    print(f"{'fuente/categoría -> clase':40s}{'n':>5s}")
    for (fu, cat, cl), arch in sorted(agg.items()):
        print(f"{fu + '/' + cat + ' -> ' + cl:40s}{len(arch):>5d}")
    tot = defaultdict(int)
    for f in sel:
        tot[(f["fuente"], f["clase"])] += 1
    print("Total por fuente y clase:", dict(tot), "| total", len(sel))
    if not a.importar:
        print("\nDRY-RUN: no se copió nada. Usa --importar.")
        return

    for f in sel:
        f["destino"].parent.mkdir(parents=True, exist_ok=True)
        if f["fuente"] == "frg":
            quitar_relleno(Image.open(f["ruta_zip"]).convert("RGB")).save(f["destino"], quality=95)
        else:
            shutil.copyfile(f["ruta_zip"], f["destino"])
    hoy = date.today().isoformat()
    campos = list(csv.DictReader(open(REGISTRO, encoding="utf-8")).fieldnames)
    with open(REGISTRO, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=campos)
        for (fu, cat, cl), arch in sorted(agg.items()):
            cfg = FUENTES[fu]
            w.writerow({"fuente": cfg["fuente"], "url": cfg["url"], "licencia": cfg["licencia"], "fecha": hoy,
                        "categoria_origen": cat, "clase_destino": cl, "n_imagenes": len(arch),
                        "archivos": ";".join(sorted(arch))})
    print(f"\nImportadas {len(sel)} imágenes; {len(agg)} filas nuevas en datos_externos.csv")


if __name__ == "__main__":
    main()
