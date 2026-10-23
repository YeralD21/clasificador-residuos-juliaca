"""ETAPA 1 - Arma dataset/publico/{train,val,test}/<clase>/ desde dataset/crudo/ SIN inventar nada.

  dataset/crudo/<fuente>/<clase_origen>/...   (lo descargas tú de Roboflow/Kaggle, formato carpetas)

Reglas:
  - Solo se usan filas de class_mapping.csv con aprobado=SI (esa columna la cambias TÚ).
  - Clase de origen sin aprobar o EXCLUIR -> no entra, y se reporta.
  - Clase de origen nueva -> se agrega al CSV con aprobado=NO para que decidas. Nunca se adivina.
  - Duplicados exactos se eliminan. Cualquier imagen idéntica a una de dataset/juliaca se descarta.
  - Si la fuente ya trae splits (crudo/<fuente>/train|valid|test/<clase>) se RESPETAN.
    Si no, se divide 80/10/10 con semilla fija (en datos públicos no sabemos qué foto es de qué objeto;
    por eso la evaluación que vale es la de dataset/juliaca/test).

Uso:
  python scripts/preparar_publico.py
  python scripts/preparar_publico.py --max-por-clase 1500
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict

from comun import CLASES, CRUDO, JULIACA, PUBLICO, RAIZ, imagenes, md5

CSV_MAPEO = RAIZ / "class_mapping.csv"
CAMPOS = ["fuente", "clase_origen", "clase_destino", "aprobado", "nota"]
ALIAS_SPLIT = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


def carpetas_de_clase(fuente):
    """Devuelve [(split_original o None, nombre_clase, carpeta)]."""
    subs = [d for d in fuente.iterdir() if d.is_dir()]
    if subs and all(d.name.lower() in ALIAS_SPLIT for d in subs):
        return [(ALIAS_SPLIT[s.name.lower()], c.name, c) for s in subs for c in s.iterdir() if c.is_dir()]
    return [(None, d.name, d) for d in subs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-por-clase", type=int, default=0, help="límite de imágenes de train por clase (0 = sin límite)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(CSV_MAPEO, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    conocidas = {(r["fuente"], r["clase_origen"]) for r in filas}
    aprobadas = {(r["fuente"], r["clase_origen"]): r["clase_destino"].strip()
                 for r in filas if r["aprobado"].strip().upper() == "SI"}
    for k, d in aprobadas.items():
        if d != "EXCLUIR" and d not in CLASES:
            raise SystemExit(f"class_mapping.csv: destino inválido '{d}' en {k}. Válidos: {CLASES} o EXCLUIR")

    hashes_juliaca = {md5(p) for p in imagenes(JULIACA)}

    por_clase = defaultdict(lambda: {"train": [], "val": [], "test": [], None: []})
    excluidas, nuevas = Counter(), set()
    if not CRUDO.exists() or not any(d.is_dir() for d in CRUDO.iterdir()):
        raise SystemExit("dataset/crudo/ está vacío. Descarga los datasets públicos primero (ver README).")

    for fuente in sorted(d for d in CRUDO.iterdir() if d.is_dir()):
        for split, nombre, carpeta in carpetas_de_clase(fuente):
            clave = (fuente.name, nombre)
            if clave not in conocidas:
                nuevas.add(clave)
            destino = aprobadas.get(clave)
            lista = imagenes(carpeta)
            if destino in (None, "EXCLUIR"):
                excluidas[f"{fuente.name}/{nombre} ({'EXCLUIR' if destino else 'sin aprobar'})"] += len(lista)
                continue
            por_clase[destino][split].extend(lista)

    if nuevas:
        with open(CSV_MAPEO, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            for fu, o in sorted(nuevas):
                w.writerow({"fuente": fu, "clase_origen": o, "clase_destino": "", "aprobado": "NO",
                            "nota": "NUEVA: asignar destino y aprobar a mano"})
        print(f"[AVISO] {len(nuevas)} clases nuevas agregadas a class_mapping.csv con aprobado=NO:")
        for fu, o in sorted(nuevas):
            print(f"        {fu}/{o}")

    # limpiar salida (conserva .gitkeep)
    for s in ("train", "val", "test"):
        for c in CLASES:
            d = PUBLICO / s / c
            if d.exists():
                for p in d.iterdir():
                    if p.name != ".gitkeep":
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
            d.mkdir(parents=True, exist_ok=True)

    vistos = set(hashes_juliaca)
    dup = choque_juliaca = 0
    resumen = {}
    for clase in CLASES:
        grupos = por_clase[clase]
        sin_split = grupos[None]
        rng.shuffle(sin_split)
        n = len(sin_split)
        n_val, n_test = int(n * 0.1), int(n * 0.1)
        splits = {
            "test": grupos["test"] + sin_split[:n_test],
            "val": grupos["val"] + sin_split[n_test:n_test + n_val],
            "train": grupos["train"] + sin_split[n_test + n_val:],
        }
        # test y val primero: si hay duplicados, se quedan en test/val y salen de train
        for s in ("test", "val", "train"):
            limpias = []
            for p in splits[s]:
                h = md5(p)
                if h in hashes_juliaca:
                    choque_juliaca += 1
                    continue
                if h in vistos:
                    dup += 1
                    continue
                vistos.add(h)
                limpias.append(p)
            if s == "train" and args.max_por_clase and len(limpias) > args.max_por_clase:
                rng.shuffle(limpias)
                limpias = limpias[:args.max_por_clase]
            for i, p in enumerate(limpias):
                fuente = p.relative_to(CRUDO).parts[0]
                shutil.copy2(p, PUBLICO / s / clase / f"{fuente}_{i:06d}{p.suffix.lower()}")
            splits[s] = limpias
        resumen[clase] = {s: len(v) for s, v in splits.items()}

    print("\n=== dataset/publico (conteos reales) ===")
    print(f"{'clase':14s} {'train':>7s} {'val':>6s} {'test':>6s}")
    for c, r in resumen.items():
        alerta = "  <-- POCAS" if r["train"] < 100 else ""
        print(f"{c:14s} {r['train']:7d} {r['val']:6d} {r['test']:6d}{alerta}")
    print(f"\nDuplicados eliminados: {dup} | Iguales a fotos de juliaca (descartadas): {choque_juliaca}")
    if excluidas:
        print("\nNo entraron:")
        for k, v in excluidas.most_common():
            print(f"  {k}: {v}")
    print("\n'sin_residuo' casi no existe en datos públicos: aporta fotos de fondo vacío en dataset/juliaca.")


if __name__ == "__main__":
    main()
