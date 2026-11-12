"""Importa a dataset/externo/train lo aprobado por el usuario (2026-11-12) de la Fase A de candidatas_peligrosos.py.

  python -I scripts/importar_peligrosos.py --bdwaste <dir> --recybat <dir>              # DRY-RUN
  python -I scripts/importar_peligrosos.py --bdwaste <dir> --recybat <dir> --importar   # copia y registra

Aprobado (regla 1a): bdw (BDWaste), msk (MaskNet), ewa (E-Waste v9) y rcb (RecyBat24 v1, variante
rotated_90_bright_1.0, pouch y cylindric, tope 100). Surgical Waste Dataset: descartado por el usuario.
Qué se importa: las piezas limpias (motivo ok o fuera_del_tope) de reportes/externo_candidatas4_dedup.csv, menos las
excluidas a mano, con el tope por fuente y clase (150; rcb 100) aplicado de nuevo tras la exclusión.
Exclusión (decisión del usuario): mascarillas de bdw en paquete nuevo o recién sacadas del paquete (tela roja), que no
son residuos: EXCLUIR_BDW_MASK.
Las piezas se regeneran con candidatas_peligrosos.piezas (mismo recorte y margen que en la Fase A); bdw se guarda con
lado mayor <= 1600 px (fotos de móvil de hasta 4000 px), JPG calidad 95.
Garantías: solo escribe en dataset/externo/train/<clase>/; nunca sobrescribe ni borra; si ya hay archivos bdw_, msk_,
ewa_ o rcb_ no hace nada; agrega filas a datos_externos.csv sin tocar las existentes.
"""
import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import CLASES, EXTERNO, RAIZ, imagenes  # noqa: E402
import candidatas_peligrosos as cp  # noqa: E402

DEDUP = RAIZ / "reportes" / "externo_candidatas4_dedup.csv"
REGISTRO = RAIZ / "datos_externos.csv"
EXCLUIR_BDW_MASK = {f"bdw_Mask_{n}.jpg" for n in
                    (1, 2, 4, 7, 8, 10, 11, 15, 16, 18, 19, 20, 30, 34, 50, 121, 122, 127)}
DEDUP_TXT = ("deduplicado contra el benchmark NTP 900.058 (nombre, dHash<=4 con 8 giros/espejos y ORB), contra la "
             "familia Smart medical/Medical-Waste-4.0/pbw/mwn/rfmed_ y contra todos los datos propios; sin casi "
             "duplicados internos (dHash<=4 y embedding coseno>=0.95); informe reportes/externo_candidatas4_"
             "{dedup,conteos}.csv")
FUENTES = {
    "bdw": dict(
        fuente="bdw: BDWaste. Akter, M., Sultana, N., Farjana, M., Uddin, A., Rahman, M. W. y Rahman, M. M. (2023). "
               "BDWaste: A comprehensive image dataset of digestible and indigestible waste in Bangladesh. Mendeley "
               "Data, V1, doi:10.17632/96g5pgfnfw.1. Sin mascarillas nuevas en paquete; " + DEDUP_TXT,
        url="https://data.mendeley.com/datasets/96g5pgfnfw/1", licencia="CC BY 4.0"),
    "msk": dict(
        fuente="msk: MaskNet. Peyvandi, A., Majidi, B., Peyvandi, S., Patra, J. C. y Moshiri, B. (2022). "
               "Location-aware hazardous litter management for smart emergency governance in urban "
               "eco-cyber-physical systems. Multimedia Tools and Applications, doi:10.1007/s11042-021-11654-w. "
               "Recortes de cajas Mask (margen 10%); " + DEDUP_TXT,
        url="https://github.com/Tenebris97/MaskNet ; https://www.kaggle.com/datasets/tenebris97/masknet",
        licencia="MIT (repositorio GitHub de los autores); Kaggle: Open Database (ODbL), contenido (c) autores"),
    "ewa": dict(
        fuente="ewa: E-Waste Classifications v9 (Roboflow Universe, student-esos5). Recortes de cajas Battery y Cable "
               "(margen 10%), imágenes estiradas a 640x640 por Roboflow; splits train y valid; " + DEDUP_TXT,
        url="https://universe.roboflow.com/student-esos5/e-waste-classifications/dataset/9", licencia="CC BY 4.0"),
    "rcb": dict(
        fuente="rcb: RecyBat24 v1 (Zenodo 13126689). RecyBat24: an Image Dataset for LIB Recycling (2025). Solo la "
               "variante rotated_90_bright_1.0 (giro exacto de 90 grados, brillo original) de pouch y cylindric; "
               "recortes de cajas COCO (margen 10%); " + DEDUP_TXT,
        url="https://zenodo.org/records/13126689", licencia="CC BY 4.0"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdwaste", type=Path, required=True)
    ap.add_argument("--recybat", type=Path, required=True)
    ap.add_argument("--importar", action="store_true")
    a = ap.parse_args()

    filas = [f for f in csv.DictReader(open(DEDUP, encoding="utf-8")) if f["motivo"] in ("ok", "fuera_del_tope")]
    excl = [f for f in filas if f["archivo"] in EXCLUIR_BDW_MASK]
    if len(excl) != len(EXCLUIR_BDW_MASK):
        sys.exit(f"ERROR: se esperaban {len(EXCLUIR_BDW_MASK)} mascarillas a excluir y hay {len(excl)}")
    filas = [f for f in filas if f["archivo"] not in EXCLUIR_BDW_MASK]

    rng = random.Random(42)
    sel = []
    grupos = defaultdict(list)
    for f in filas:
        grupos[(f["fuente"], f["clase"])].append(f)
    for (fu, cl), fs in sorted(grupos.items()):
        ok = [f for f in fs if f["motivo"] == "ok"]
        resto = [f for f in fs if f["motivo"] == "fuera_del_tope"]
        rng.shuffle(resto)
        porcat = defaultdict(list)
        for f in resto:
            porcat[f["categoria"]].append(f)
        colas = [v for _, v in sorted(porcat.items())]
        while len(ok) < cp.TOPE[fu] and any(colas):  # rellena el hueco de lo excluido, repartido por categoría
            for v in colas:
                if v and len(ok) < cp.TOPE[fu]:
                    ok.append(v.pop())
        sel += ok[:cp.TOPE[fu]]

    # regenerar las piezas desde los originales
    desc = Counter()
    cands = cp.cands_bdw(a.bdwaste, desc) + cp.cands_msk(desc) + cp.cands_ewa(desc) + cp.cands_rcb(a.recybat, desc)
    por_ruta = defaultdict(list)
    for c in cands:
        por_ruta[c["ruta_zip"]].append(c)
    quiero = defaultdict(set)
    for f in sel:
        quiero[f["ruta_zip"]].add(f["archivo"])
    piezas = {}
    for ruta, nombres in quiero.items():
        for c in por_ruta[ruta]:
            for p in cp.piezas(c, cp.abrir(c)):
                if p["archivo"] in nombres:
                    piezas[(ruta, p["archivo"])] = p
    faltan = [f["archivo"] for f in sel if (f["ruta_zip"], f["archivo"]) not in piezas]
    if faltan:
        sys.exit(f"ERROR: no se pudieron regenerar {len(faltan)} piezas: {faltan[:5]}")
    for f in sel:
        if f["clase"] not in CLASES:
            sys.exit(f"ERROR: clase inválida {f['clase']}")
        nombre = f["archivo"]
        if f["fuente"] == "ewa":  # Roboflow repite nombres base (11_jpg.rf.<h1> y .rf.<h2>): se añade el hash corto
            nombre = nombre.replace(f"_{f['split']}_", f"_{f['split']}_{f['ruta_zip'].split('.rf.')[1][:6]}_", 1)
        f["destino"] = EXTERNO / "train" / f["clase"] / nombre
    destinos = [f["destino"] for f in sel]
    if len(set(destinos)) != len(destinos):
        sys.exit("ERROR: nombres de destino repetidos")
    previos = [p for c in CLASES for p in imagenes(EXTERNO / "train" / c) if p.name.startswith(("bdw_", "msk_",
                                                                                                "ewa_", "rcb_"))]
    if previos or any(d.exists() for d in destinos):
        sys.exit(f"ERROR: ya hay archivos bdw_/msk_/ewa_/rcb_ en externo/train ({len(previos)}); no se sobrescribe.")

    agg = defaultdict(list)
    for f in sel:
        agg[(f["fuente"], f["categoria"], f["clase"])].append(f["destino"].name)
    print(f"Excluidas a mano (bdw Mask en paquete nuevo): {len(excl)}")
    print(f"{'fuente/categoría -> clase':42s}{'n':>5s}")
    for (fu, cat, cl), arch in sorted(agg.items()):
        print(f"{fu + '/' + cat + ' -> ' + cl:42s}{len(arch):>5d}")
    tot = Counter((f["fuente"], f["clase"]) for f in sel)
    print("Total por fuente y clase:", dict(tot), "| total", len(sel))
    if not a.importar:
        print("\nDRY-RUN: no se copió nada. Usa --importar.")
        return

    for f in sel:
        f["destino"].parent.mkdir(parents=True, exist_ok=True)
        piezas[(f["ruta_zip"], f["archivo"])]["img"].save(f["destino"], quality=95)
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
