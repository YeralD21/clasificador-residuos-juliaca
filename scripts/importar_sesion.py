"""Copia a dataset/juliaca las fotos de una sesión YA etiquetada a mano con etiquetar_sesion.py.

  python scripts/importar_sesion.py --dry-run           # solo muestra qué haría; no escribe nada
  python scripts/importar_sesion.py                     # la fecha más reciente de D:\\Proyecto-Residuos-ML\\fotos_laptop
  python scripts/importar_sesion.py --fecha 2026-10-26 --recorte roi

La clase y el split los decidiste TÚ por objeto en etiquetas.csv; este script no decide nada (CLAUDE.md, reglas
2 y 3). Cada objeto va entero a UN split, en su propia carpeta, con el siguiente número libre de juliaca:
  dataset/juliaca/<split>/<clase>/obj046_20261026-obj003/20261026_obj003_01.jpg
Solo train y val (test se arma aparte). Las fotos originales de fotos_laptop no se tocan.
Imagen: por defecto el frame completo, escalado para que el lado mayor sea --lado-max (1600, como las fotos de
juliaca: 1200x1600 / 1600x1200), JPG 95. --recorte roi aplica antes el cuadrado central de capturar_fotos.py
(roi_cuadrado, 75 % del lado menor). Nunca sobrescribe; un objeto ya importado (misma fecha y objNNN) se salta.
Al final imprime fotos/objetos por clase y split antes y después. Luego ejecuta python scripts/auditar.py.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2

from comun import CLASES, JULIACA, imagenes, roi_cuadrado
from etiquetar_sesion import FOTOS, fotos_de, leer, ultima_fecha

SPLITS_OK = ("train", "val")
TODOS = ("train", "val", "test")


def conteos(base):
    """{(clase, split): (fotos, objetos)} de dataset/juliaca, solo lectura."""
    out = {}
    for s in TODOS:
        for c in CLASES:
            d = base / s / c
            objs = [o for o in d.iterdir() if o.is_dir()] if d.exists() else []
            out[c, s] = (len(imagenes(d)), len(objs))
    return out


def tabla(titulo, cont):
    print(f"\n{titulo}  (fotos/objetos)")
    print(f"{'clase':14s}" + "".join(f"{s:>12s}" for s in TODOS))
    for c in CLASES:
        print(f"{c:14s}" + "".join(f"{cont[c, s][0]:>7d}/{cont[c, s][1]:<4d}" for s in TODOS))
    print(f"{'TOTAL':14s}" + "".join(f"{sum(cont[c, s][0] for c in CLASES):>7d}/"
                                     f"{sum(cont[c, s][1] for c in CLASES):<4d}" for s in TODOS))


def preparar(p, recorte, lado_max):
    im = cv2.imread(str(p))
    if im is None:
        raise SystemExit(f"No se pudo leer {p}")
    if recorte == "roi":
        im = roi_cuadrado(im)[0]
    e = lado_max / max(im.shape[:2]) if lado_max else 1.0
    if e < 1:
        im = cv2.resize(im, (round(im.shape[1] * e), round(im.shape[0] * e)), interpolation=cv2.INTER_AREA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", default=str(FOTOS), help="carpeta con las sesiones AAAA-MM-DD")
    ap.add_argument("--fecha", help="AAAA-MM-DD (por defecto la más reciente)")
    ap.add_argument("--dataset", default=str(JULIACA), help="destino (por defecto dataset/juliaca)")
    ap.add_argument("--recorte", choices=("ninguno", "roi"), default="ninguno")
    ap.add_argument("--lado-max", type=int, default=1600, help="lado mayor final en px (0 = sin redimensionar)")
    ap.add_argument("--dry-run", action="store_true", help="solo muestra lo que haría, no escribe nada")
    args = ap.parse_args()

    raiz, base = Path(args.raiz), Path(args.dataset)
    fecha = args.fecha or ultima_fecha(raiz)
    dia, fc = raiz / fecha, fecha.replace("-", "")
    filas = leer(dia / "etiquetas.csv")
    if not filas:
        raise SystemExit(f"No hay {dia / 'etiquetas.csv'}: primero python scripts/etiquetar_sesion.py --fecha {fecha}")

    # --- validar TODO antes de escribir nada ---
    errores, vistos = [], set()
    for f in filas:
        o = f["objeto"]
        if o in vistos:
            errores.append(f"{o} aparece dos veces en etiquetas.csv")
        vistos.add(o)
        if f["clase"] == "descartado":
            continue
        if f["clase"] not in CLASES:
            errores.append(f"{o}: clase desconocida '{f['clase']}'")
        if f["split"] not in SPLITS_OK:
            errores.append(f"{o}: split '{f['split']}' no permitido (solo {', '.join(SPLITS_OK)}; test se arma aparte)")
        if not (dia / o).is_dir() or not fotos_de(dia / o):
            errores.append(f"{o}: no existe {dia / o} o no tiene fotos")
    if errores:
        raise SystemExit("ERRORES en etiquetas.csv (no se copió nada):\n  " + "\n  ".join(errores))

    # objetos de esta sesión ya importados: carpeta obj<N>_<fecha>-<objeto> en cualquier split/clase
    ya, usados = defaultdict(list), [0]
    for s in TODOS:
        for c in CLASES:
            for d in (base / s / c).iterdir() if (base / s / c).exists() else []:
                if d.is_dir() and (m := re.match(r"obj(\d+)", d.name)):
                    usados.append(int(m.group(1)))
                    if (m2 := re.search(rf"_{fc}-(obj\d+)$", d.name)):
                        ya[m2.group(1)].append(f"{s}/{c}/{d.name}")
    siguiente = max(usados) + 1

    antes = conteos(base)
    plan, descartados, saltados = [], 0, []
    for f in filas:
        if f["clase"] == "descartado":
            descartados += 1
            continue
        if ya[f["objeto"]]:
            saltados.append(f"{f['objeto']} (ya está en {', '.join(ya[f['objeto']])})")
            continue
        plan.append((f, base / f["split"] / f["clase"] / f"obj{siguiente:03d}_{fc}-{f['objeto']}"))
        siguiente += 1

    modo = "DRY-RUN (no se escribe nada)" if args.dry_run else "IMPORTANDO"
    print(f"{modo}: sesión {dia} -> {base}")
    print(f"recorte={args.recorte}, lado mayor <= {args.lado_max or 'sin límite'} px, JPG 95")
    for f, dest in plan:
        fotos = fotos_de(dia / f["objeto"])
        h, w = preparar(fotos[0], args.recorte, args.lado_max).shape[:2]
        print(f"  {f['objeto']:10s} {len(fotos):3d} fotos -> {dest.relative_to(base)}/  ({w}x{h})")
    for s in saltados:
        print(f"  saltado: {s}")
    print(f"  descartados en etiquetas.csv: {descartados}")

    despues = dict(antes)
    for f, dest in plan:
        fotos = fotos_de(dia / f["objeto"])
        c, s = f["clase"], f["split"]
        despues[c, s] = (despues[c, s][0] + len(fotos), despues[c, s][1] + 1)
        if args.dry_run:
            continue
        dest.mkdir(parents=True, exist_ok=False)  # carpeta nueva: nunca mezcla ni sobrescribe
        for j, p in enumerate(fotos, 1):
            out = dest / f"{fc}_{f['objeto']}_{j:02d}.jpg"
            if out.exists() or not cv2.imwrite(str(out), preparar(p, args.recorte, args.lado_max),
                                               [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise SystemExit(f"No se pudo escribir {out} (¿ya existía?). Se detuvo aquí.")

    tabla("ANTES", antes)
    if args.dry_run:
        tabla("DESPUÉS (simulado)", despues)
        print("\nDry-run: no se escribió nada. Sin --dry-run para importar.")
        return
    real = conteos(base)
    tabla("DESPUÉS", real)
    if real != despues:
        print("\n[AVISO] Los conteos reales no coinciden con lo esperado: revisa la carpeta.")
    # cada objeto de la sesión debe estar en exactamente UN split
    por_obj = defaultdict(set)
    for s in TODOS:
        for c in CLASES:
            for d in (base / s / c).glob(f"obj*_{fc}-obj*"):
                por_obj[d.name.split("-", 1)[1]].add(s)
    repartidos = {o: ss for o, ss in por_obj.items() if len(ss) > 1}
    print("\n" + (f"[ERROR] objetos en más de un split: {repartidos}" if repartidos
                  else f"OK: {len(por_obj)} objetos de la sesión, cada uno en un solo split."))
    print("Ahora: python scripts/auditar.py")


if __name__ == "__main__":
    main()
