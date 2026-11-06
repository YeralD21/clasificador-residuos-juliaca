"""Paso 3: arma el test del docente a partir de los zips de Roboflow ("Folder Structure").

1. Busca en --zips (por defecto ~/Downloads) los zips cuyo nombre empieza por la carpeta de
   exportación de cada dataset (p. ej. "garbage.v1i.folder", "Garbage Classification.v4i.folder").
   Si no se ha descomprimido aún, lo extrae (solo lectura del zip) en imagenes/_descargas/<clave>/.
   Rechaza zips con rutas absolutas o '..'.
2. Indexa: copia a imagenes/test_docente/<dataset>/<clase>/<archivo> SOLO las 1.455 imágenes cuyo
   nombre (con hash .rf.) aparece en datos_docente/pred_01_best_jhoel.csv.
3. Escribe salidas/indexado.json y salidas/faltantes.csv con lo encontrado por grupo.

python -I evaluacion_docente/preparar_test.py              # zips de ~/Downloads + indexado
python -I evaluacion_docente/preparar_test.py --solo-indexar
Volver a correrlo es seguro: no re-extrae ni sobrescribe lo que ya existe.
"""
import argparse
import csv
import json
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

DESC = AQUI / "imagenes" / "_descargas"
TEST = AQUI / "imagenes" / "test_docente"
OUT = AQUI / "salidas"
DATASETS = {"v1i": "garbage.v1i.folder", "v4": "Garbage Classification.v4i.folder",
            "med": "Smart medical waste segregation.v3i.folder"}


def extraer_zips(carpeta):
    for clave, pref in DATASETS.items():
        destino = DESC / clave
        zips = sorted(p for p in Path(carpeta).glob("*.zip") if p.name.lower().startswith(pref.lower()))
        if not zips:
            print(f"[{clave}] no hay zip '{pref}*.zip' en {carpeta}")
            continue
        if destino.is_dir() and any(destino.iterdir()):
            print(f"[{clave}] ya extraído en {destino} ({zips[0].name})")
            continue
        z = zipfile.ZipFile(zips[0])
        malos = [n for n in z.namelist()
                 if n.startswith(("/", "\\")) or ":" in n or ".." in n.replace("\\", "/").split("/")]
        if malos:
            sys.exit(f"[{clave}] {zips[0].name} tiene rutas inseguras, no se extrae: {malos[:3]}")
        destino.mkdir(parents=True, exist_ok=True)
        z.extractall(destino)
        print(f"[{clave}] extraído {zips[0].name} -> {destino}")


def indexar():
    indice = {}
    for p in DESC.rglob("*"):
        if p.is_file():
            indice.setdefault(p.name, []).append(p)
    ref = nm.leer_csv(AQUI / "datos_docente" / "pred_01_best_jhoel.csv")
    esperado, encontrado, faltan, ambiguos = Counter(), Counter(), [], 0
    for f in ref:
        partes = f["path"].replace("\\", "/").split("/")
        archivo, clase = partes[-1], partes[-2]
        g = nm.grupo_de(f)
        esperado[g] += 1
        cands = indice.get(archivo, [])
        pref = [c for c in cands if c.parent.name == clase and c.parent.parent.name == "test"]
        cands = pref or cands
        if not cands:
            faltan.append({"dataset": f["dataset"], "clase": clase, "grupo": g, "archivo": archivo})
            continue
        ambiguos += len(cands) > 1
        dst = TEST / f["dataset"] / clase / archivo
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(cands[0], dst)
        encontrado[g] += 1
    OUT.mkdir(exist_ok=True)
    res = {"n_esperado": len(ref), "n_encontrado": sum(encontrado.values()), "ambiguos": ambiguos,
           "por_grupo": {g: {"encontradas": encontrado[g], "esperadas": esperado[g]} for g in nm.GRUPOS},
           "faltan_por_dataset": dict(Counter(x["dataset"] for x in faltan))}
    (OUT / "indexado.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(OUT / "faltantes.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "clase", "grupo", "archivo"])
        w.writeheader()
        w.writerows(faltan)
    print(f"Encontradas {res['n_encontrado']}/{len(ref)} (ambiguas {ambiguos})")
    for g, v in res["por_grupo"].items():
        marca = "OK" if v["encontradas"] == v["esperadas"] else ("FALTA" if not v["encontradas"] else "PARCIAL")
        print(f"  {g:32s} {v['encontradas']:4d}/{v['esperadas']:<4d} {marca}")
    if faltan:
        print("Faltan por dataset:", res["faltan_por_dataset"], "-> salidas/faltantes.csv")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", default=str(Path.home() / "Downloads"))
    ap.add_argument("--solo-indexar", action="store_true")
    a = ap.parse_args()
    if not a.solo_indexar:
        extraer_zips(a.zips)
    indexar()


if __name__ == "__main__":
    main()
