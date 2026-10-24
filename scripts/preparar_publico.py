"""ETAPA 1 - Arma dataset/publico/{train,val,test}/<clase>/ desde dataset/crudo/ SIN inventar nada.

  dataset/crudo/<fuente>/<clase_origen>/...                (sin split: se divide 80/10/10)
  dataset/crudo/<fuente>/{train,val|valid,test}/<clase>/... (con split: se RESPETA)

Reglas:
  - Solo se usan filas de class_mapping.csv con aprobado=SI (esa columna la cambias TÚ).
  - Clase de origen sin aprobar o EXCLUIR -> no entra, y se reporta.
  - Clase de origen nueva -> se agrega al CSV con aprobado=NO para que decidas. Nunca se adivina.
  - excluir_imagenes.txt: imágenes que TÚ decidiste sacar (mal clasificadas, no son residuo...).
    No se borran de crudo/, solo no se copian.
  - Duplicados exactos (md5) se eliminan. Imágenes idénticas a una de dataset/juliaca se descartan.
  - ANTI-LEAKAGE (hash perceptual): lo casi igual a test se saca de val, y lo casi igual a
    val/test se saca de train. Test nunca se toca.

Uso:
  python scripts/preparar_publico.py
  python scripts/preparar_publico.py --max-por-clase 1500
  python scripts/preparar_publico.py --umbral-casi-dup 0      # desactiva el filtro anti-leakage
"""
import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict

import numpy as np

from comun import CLASES, CRUDO, JULIACA, PUBLICO, RAIZ, dhash, imagenes, md5

CSV_MAPEO = RAIZ / "class_mapping.csv"
TXT_EXCLUIR = RAIZ / "excluir_imagenes.txt"
CAMPOS = ["fuente", "clase_origen", "clase_destino", "aprobado", "nota"]
ALIAS_SPLIT = {"train": "train", "valid": "val", "val": "val", "validation": "val", "test": "test"}


def carpetas_de_clase(fuente):
    """Devuelve [(split_original o None, nombre_clase, carpeta)]."""
    subs = [d for d in fuente.iterdir() if d.is_dir()]
    if subs and all(d.name.lower() in ALIAS_SPLIT for d in subs):
        return [(ALIAS_SPLIT[s.name.lower()], c.name, c) for s in subs for c in s.iterdir() if c.is_dir()]
    return [(None, d.name, d) for d in subs]


def leer_exclusiones():
    """Rutas relativas a dataset/crudo, una por línea. '#' = comentario."""
    if not TXT_EXCLUIR.exists():
        return set()
    rutas = set()
    for linea in TXT_EXCLUIR.read_text(encoding="utf-8").splitlines():
        linea = linea.split("#", 1)[0].strip().replace("\\", "/").strip("/")
        if linea:
            rutas.add(linea.lower())
    return rutas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-por-clase", type=int, default=0, help="límite de imágenes de train por clase (0 = sin límite)")
    ap.add_argument("--umbral-casi-dup", type=int, default=4,
                    help="bits distintos (de 64) para considerar casi iguales train vs val/test; 0 = desactivado")
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

    if not CRUDO.exists() or not any(d.is_dir() for d in CRUDO.iterdir()):
        raise SystemExit("dataset/crudo/ está vacío. Descarga los datasets públicos primero (ver README).")

    excluir = leer_exclusiones()
    usadas_excl = set()
    hashes_juliaca = {md5(p) for p in imagenes(JULIACA)}

    por_clase = defaultdict(lambda: {"train": [], "val": [], "test": [], None: []})
    excluidas, nuevas = Counter(), set()
    n_excl_manual = 0
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
            for p in lista:
                rel = p.relative_to(CRUDO).as_posix().lower()
                if rel in excluir:
                    usadas_excl.add(rel)
                    n_excl_manual += 1
                    continue
                por_clase[destino][split].append(p)

    if nuevas:
        with open(CSV_MAPEO, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS)
            for fu, o in sorted(nuevas):
                w.writerow({"fuente": fu, "clase_origen": o, "clase_destino": "", "aprobado": "NO",
                            "nota": "NUEVA: asignar destino y aprobar a mano"})
        print(f"[AVISO] {len(nuevas)} clases nuevas agregadas a class_mapping.csv con aprobado=NO:")
        for fu, o in sorted(nuevas):
            print(f"        {fu}/{o}")

    # --- 1) armar splits por clase y deduplicar exactos (test > val > train) ---
    vistos = set(hashes_juliaca)
    dup = choque_juliaca = 0
    finales = {c: {} for c in CLASES}
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
            finales[clase][s] = limpias

    # --- 2) anti-leakage: sacar de TRAIN lo casi igual a val/test (de cualquier clase) ---
    quitadas_leak = Counter()
    quitadas_val = 0
    if args.umbral_casi_dup > 0:
        def cerca(h, banco):
            if not len(banco):
                return False
            x = np.bitwise_xor(banco, np.uint64(h))
            return np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1).min() <= args.umbral_casi_dup

        test_paths = [p for c in CLASES for p in finales[c]["test"]]
        print(f"Calculando hash perceptual de {len(test_paths)} imágenes de test ...")
        test_h = np.array([dhash(p) for p in test_paths], dtype=np.uint64)
        val_h = []
        for clase in CLASES:  # val casi igual a test -> sale de val
            quedan = []
            for p in finales[clase]["val"]:
                h = dhash(p)
                if cerca(h, test_h):
                    quitadas_val += 1
                    continue
                quedan.append(p)
                val_h.append(h)
            finales[clase]["val"] = quedan
        eval_h = np.concatenate([test_h, np.array(val_h, dtype=np.uint64)])
        n_train = sum(len(finales[c]["train"]) for c in CLASES)
        print(f"Comparando {n_train} imágenes de train contra val/test (umbral <= {args.umbral_casi_dup} bits) ...")
        for clase in CLASES:
            quedan = []
            for p in finales[clase]["train"]:
                if cerca(dhash(p), eval_h):
                    quitadas_leak[clase] += 1
                    continue
                quedan.append(p)
            finales[clase]["train"] = quedan

    # --- 3) límite por clase y copia ---
    for s in ("train", "val", "test"):
        for c in CLASES:
            d = PUBLICO / s / c
            if d.exists():
                for p in d.iterdir():
                    if p.name != ".gitkeep":
                        shutil.rmtree(p) if p.is_dir() else p.unlink()
            d.mkdir(parents=True, exist_ok=True)

    resumen = {}
    for clase in CLASES:
        tr = finales[clase]["train"]
        if args.max_por_clase and len(tr) > args.max_por_clase:
            rng.shuffle(tr)
            finales[clase]["train"] = tr[:args.max_por_clase]
        for s in ("train", "val", "test"):
            for i, p in enumerate(finales[clase][s]):
                fuente = p.relative_to(CRUDO).parts[0]
                shutil.copy2(p, PUBLICO / s / clase / f"{fuente}_{i:06d}_{p.stem}{p.suffix.lower()}")
        resumen[clase] = {s: len(finales[clase][s]) for s in ("train", "val", "test")}

    print("\n=== dataset/publico (conteos reales) ===")
    print(f"{'clase':14s} {'train':>7s} {'val':>6s} {'test':>6s} {'sacadas_train_por_leakage':>26s}")
    for c, r in resumen.items():
        alerta = "  <-- POCAS" if r["train"] < 100 else ""
        print(f"{c:14s} {r['train']:7d} {r['val']:6d} {r['test']:6d} {quitadas_leak[c]:26d}{alerta}")
    print(f"\nDuplicados exactos eliminados: {dup} | Iguales a fotos de juliaca: {choque_juliaca}")
    print(f"Sacadas de train por ser casi iguales a val/test: {sum(quitadas_leak.values())}")
    print(f"Sacadas de val por ser casi iguales a test: {quitadas_val}")
    print(f"Excluidas por excluir_imagenes.txt: {n_excl_manual}")
    no_encontradas = excluir - usadas_excl
    if no_encontradas:
        print(f"[AVISO] {len(no_encontradas)} rutas de excluir_imagenes.txt no coinciden con ninguna imagen "
              f"(revisa que sean relativas a dataset/crudo). Ejemplos: {sorted(no_encontradas)[:5]}")
    if excluidas:
        print("\nNo entraron:")
        for k, v in excluidas.most_common():
            print(f"  {k}: {v}")
    print("\n'sin_residuo' casi no existe en datos públicos: aporta fotos de fondo vacío en dataset/juliaca.")


if __name__ == "__main__":
    main()
