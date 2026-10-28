"""Fase A de la selección curada de datos públicos para la etapa 2 (papel_carton y generales). No mueve nada.

  python scripts/seleccionar_curados.py            # candidatas, mosaicos numerados y plantilla de revisión
  python scripts/seleccionar_curados.py --final 100   # tras la revisión: lista de rutas para entrenar.py --seleccion

Busca SOLO en lo que ya está descargado y aprobado para train: dataset/publico/train y dataset/externo/train
(nunca val ni test). Quiere imágenes parecidas al uso real: un solo objeto, cerca, en la mano o con fondo variado,
no fondo blanco de estudio, y con variedad de tipos (cajas, periódico, Tetra Pak...; envolturas, sachets, tecnopor...).

Filtro automático (solo propone; la selección final la aprueba el usuario):
  1. fuera: taco_* (ya entran con --externo-fuentes taco en la receta), la muestra de replay que tomaría
     entrenar.py (semilla 42, replay 2, juliaca actual) y lado menor < 96 px;
  2. detector YOLOE (modelos/detectores/yoloe-26s-seg.pt) con prompts del tipo de objeto, prompts negativos
     (ropa, zapatos, vasos de papel...) y "hand": la caja principal debe ser de un tipo buscado (conf >= 0.20),
     con como mucho 2 objetos con conf >= 0.25 y caja >= 6 % de la imagen (cerca, no colillas en el suelo). El prompt de la caja principal da el "tipo"; si se ve una mano,
     el puntaje sube x1.3. En TACO extra el tipo sale de su categoría anotada a mano y el detector solo cuenta
     objetos y manos;
  3. publico: fuera el fondo liso (desviación del gris en el borde < 14, típico de estudio) y las texturas a
     pantalla completa (caja >= 90 % de la imagen y poco contraste centro-borde);
  4. anti-leakage: fuera las que estén a dHash <= 4 de juliaca (train/val/test), val_camara o publico val/test,
     y los casi duplicados entre candidatas.
Luego se eligen ~300 por clase con cupo por fuente y reparto por tipo (round-robin, mejor puntuación primero).

Salidas (reportes/seleccion_curada/): candidatas.csv (todas las del pool, con medidas y motivo), dedup.csv,
indice.csv (número -> archivo), <clase>/mosaico_NN.jpg (48 por hoja, no se suben) y la plantilla
reportes/seleccion_curada.txt, que ningún script sobrescribe. Numeración única: papel_carton primero, luego generales.
Si indice.csv ya existe y no coincide con lo que saldría ahora, se detiene en vez de regenerar.

--final N (fase B, sin detector): lee la revisión (terminado: si; descartar: ...) y escribe
reportes/seleccion_curada/seleccion_final.csv con el MISMO número de imágenes aprobadas en las dos clases
(el mínimo de aprobadas, con tope N). Variedad: round-robin por familia de tipo y, dentro de cada familia, por
fuente (en el orden del índice). En generales entran primero todas las envolturas, sachets y servilletas
aprobadas, y el tecnopor completa.
"""
import argparse
import csv
import os
import re
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from comun import CLASES, EXTERNO, JULIACA, PUBLICO, RAIZ, VAL_CAMARA, dhash, imagenes

SALIDA = RAIZ / "reportes" / "seleccion_curada"
PLANTILLA = RAIZ / "reportes" / "seleccion_curada.txt"
DETECTOR = RAIZ / "modelos" / "detectores" / "yoloe-26s-seg.pt"
SEMILLA, REPLAY = 42, 2.0          # los de la receta de etapa2_ext_full (entrenar.py --replay 2 --seed 42)
LADO_MIN, CONF_OBJ, CONF_CONTAR, MAX_OBJ, AREA_MIN = 96, 0.20, 0.25, 2, 0.06
BORDE_LISO, AREA_TEXTURA, CONTRASTE_TEXTURA = 14.0, 0.90, 18.0
DHASH_MAX, POR_HOJA, COLS, LADO = 4, 48, 8, 200

# tipo (texto del mosaico, ASCII) -> prompt en inglés para YOLOE
TIPOS = {
    "papel_carton": {
        "caja": "cardboard box", "corrugado": "corrugated cardboard", "periodico": "newspaper",
        "revista": "magazine", "hoja": "sheet of paper", "papel color": "colored paper",
        "bolsa papel": "paper bag", "tetra pak": "milk carton", "tetra jugo": "juice box",
        "carton huevos": "egg carton",
    },
    "generales": {
        "golosina": "candy wrapper", "snack": "chips bag", "sachet": "sachet", "tecnopor": "styrofoam",
        "vaso tecnopor": "foam cup", "papel higienico": "toilet paper", "servilleta": "tissue paper",
    },
}
# Prompts negativos: si la caja principal es uno de estos, la imagen sale (ropa, zapatos, vasos de papel... que
# hay en publico/train/generales y no son los tipos buscados). Sin "book" (toma periódicos y revistas por libros) ni
# "person"/"furniture" (en el uso real hay una persona sosteniendo el objeto o una mesa). "hand" suma puntos.
NEGATIVOS = {
    "papel_carton": ["plastic bottle", "glass bottle", "can"],
    "generales": ["clothing", "shoe", "paper cup", "diaper", "face mask", "toothbrush", "plastic bottle"],
}
MANO, CONF_MANO, BONUS_MANO = "hand", 0.30, 1.3
# TACO trae la categoría anotada a mano: da el tipo directamente (más fiable que el detector en recortes chicos)
TIPO_TACO = {
    "Corrugated carton": "corrugado", "Drink carton": "tetra pak", "Egg carton": "carton huevos",
    "Magazine paper": "revista", "Normal paper": "hoja", "Other carton": "caja", "Paper bag": "bolsa papel",
    "Toilet tube": "tubo higienico", "Wrapping paper": "papel color",
    "Foam cup": "vaso tecnopor", "Foam food container": "envase tecnopor", "Styrofoam piece": "tecnopor",
    "Tissues": "servilleta",
}
# fuente -> cupo de candidatas por clase (lo que falte lo completa la última fuente de la lista)
CUPOS = {
    "papel_carton": [("tacox", 120), ("realwaste", 90), ("publico", 300)],
    "generales": [("tacox", 125), ("publico", 300)],
}
OBJETIVO = 300


def fuente_de(p):
    if PUBLICO in p.parents:
        return "publico"
    return p.name.split("_", 1)[0]          # taco, tacox, realwaste, trashnet


def muestra_replay():
    """Las mismas fotos de publico/train que pondría entrenar.armar_etapa2(replay=2, seed=42) hoy."""
    rng, fuera = random.Random(SEMILLA), set()
    for c in CLASES:
        n_jul = len(imagenes(JULIACA / "train" / c))
        pub = imagenes(PUBLICO / "train" / c)
        rng.shuffle(pub)
        fuera.update(pub[:min(len(pub), int(round(n_jul * REPLAY)))])
    return fuera


def categorias_externas():
    with open(RAIZ / "datos_externos.csv", newline="", encoding="utf-8") as f:
        return {a: r["categoria_origen"] for r in csv.DictReader(f) for a in r["archivos"].split(";") if a}


def medir_fondo(im, caja):
    """Desviación del gris en un borde del 8 % y contraste de color (Lab) entre el centro de la caja y el borde."""
    h, w = im.shape[:2]
    b = max(2, int(0.08 * min(h, w)))
    gris = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    borde = np.concatenate([gris[:b].ravel(), gris[-b:].ravel(), gris[:, :b].ravel(), gris[:, -b:].ravel()])
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    lb = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                         lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)]).mean(0)
    x1, y1, x2, y2 = caja
    cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, max(1, (x2 - x1) / 4), max(1, (y2 - y1) / 4)
    centro = lab[int(cy - ry):int(cy + ry) + 1, int(cx - rx):int(cx + rx) + 1].reshape(-1, 3).mean(0)
    return float(borde.std()), float(np.linalg.norm(centro - lb))


def detectar(clase, rutas):
    from ultralytics import YOLOE
    det = YOLOE(str(DETECTOR))
    nombres = list(TIPOS[clase]) + [f"NO:{n}" for n in NEGATIVOS[clase]] + ["MANO"]
    prompts = [TIPOS[clase][t] for t in TIPOS[clase]] + NEGATIVOS[clase] + [MANO]
    det.set_classes(prompts, det.get_text_pe(prompts))
    out = {}
    for i in range(0, len(rutas), 32):
        lote = rutas[i:i + 32]
        for p, r in zip(lote, det.predict([str(x) for x in lote], conf=0.05, agnostic_nms=True, imgsz=640,
                                          verbose=False)):
            cajas = r.boxes
            if cajas is None or not len(cajas):
                out[p] = None
                continue
            conf = cajas.conf.cpu().numpy()
            cls = [nombres[int(k)] for k in cajas.cls.cpu().numpy()]
            xyxy = cajas.xyxy.cpu().numpy()
            mano = any(c == "MANO" and s >= CONF_MANO for c, s in zip(cls, conf))
            objs = [k for k, c in enumerate(cls) if c != "MANO"]   # la mano no cuenta como objeto
            if not objs:
                out[p] = dict(conf=0.0, tipo="", area=0.0, n_obj=0, caja=None, mano=mano)
                continue
            j = max(objs, key=lambda k: conf[k])
            h, w = r.orig_shape
            area = float((xyxy[j, 2] - xyxy[j, 0]) * (xyxy[j, 3] - xyxy[j, 1]) / (w * h))
            out[p] = dict(conf=float(conf[j]), tipo=cls[j], area=area, mano=mano, caja=xyxy[j].tolist(),
                          n_obj=int(sum(conf[k] >= CONF_CONTAR for k in objs)))
        print(f"  {clase}: detector {min(i + 32, len(rutas))}/{len(rutas)}", end="\r", flush=True)
    print()
    return out


def banco_leakage():
    """dHash de juliaca (todos los splits), val_camara y publico val/test: las candidatas no pueden parecerse."""
    rutas = imagenes(JULIACA) + imagenes(VAL_CAMARA) + imagenes(PUBLICO / "val") + imagenes(PUBLICO / "test")
    return rutas, np.array([dhash(p) for p in rutas], dtype=np.uint64)


def dist(banco, h):
    if not len(banco):
        return np.array([], dtype=np.int64)
    x = banco ^ np.uint64(h)
    return np.array([bin(int(v)).count("1") for v in x])


def mosaicos(filas, clase):
    d = SALIDA / clase
    d.mkdir(parents=True, exist_ok=True)
    for hoja in range(0, len(filas), POR_HOJA):
        grupo = filas[hoja:hoja + POR_HOJA]
        nf = (len(grupo) + COLS - 1) // COLS
        lienzo = np.full((nf * LADO, COLS * LADO, 3), 255, np.uint8)
        for k, f in enumerate(grupo):
            im = cv2.imread(f["ruta_abs"])
            e = (LADO - 6) / max(im.shape[:2])
            im = cv2.resize(im, (max(1, int(im.shape[1] * e)), max(1, int(im.shape[0] * e))),
                            interpolation=cv2.INTER_AREA)
            y0, x0 = (k // COLS) * LADO + 3, (k % COLS) * LADO + 3
            lienzo[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
            cv2.rectangle(lienzo, (x0, y0), (x0 + 62, y0 + 34), (0, 0, 0), -1)
            cv2.putText(lienzo, str(f["numero"]), (x0 + 4, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2,
                        cv2.LINE_AA)
            pie = f"{f['fuente']} / {f['tipo']}" + (" / mano" if f.get("mano") else "")
            cv2.rectangle(lienzo, (x0, y0 + LADO - 26), (x0 + LADO - 6, y0 + LADO - 6), (0, 0, 0), -1)
            cv2.putText(lienzo, pie, (x0 + 4, y0 + LADO - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        cv2.imwrite(str(d / f"mosaico_{hoja // POR_HOJA + 1:02d}.jpg"), lienzo, [cv2.IMWRITE_JPEG_QUALITY, 88])


PLANTILLA_TXT = """# Revisión de la selección curada para la etapa 2 ({n} candidatas: papel_carton {a}-{b}, generales {c}-{d}).
# Mosaicos en reportes/seleccion_curada/<clase>/ (48 por hoja); indice.csv dice qué archivo es cada número.
# - descartar: números separados por coma o espacio (las que NO quieres en la selección).
# - cuando termines, cambia "terminado: no" por "terminado: si". Sin eso no se entrena con la selección.
# Las líneas que empiezan con # se ignoran. Este archivo no lo sobrescribe ningún script.

terminado: no

descartar:
"""


SELECCION = SALIDA / "seleccion_final.csv"
FAMILIA = {"tetra jugo": "tetra pak", "vaso tecnopor": "tecnopor", "envase tecnopor": "tecnopor"}
PRIMERO = {"generales": {"golosina", "snack", "sachet", "servilleta"}}   # antes que el tecnopor


def leer_revision():
    """Números descartados de reportes/seleccion_curada.txt; exige 'terminado: si' y números del índice."""
    texto = "\n".join(l for l in PLANTILLA.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("#"))
    if not re.search(r"^terminado:\s*si\s*$", texto, re.M):
        raise SystemExit(f"{PLANTILLA.relative_to(RAIZ)}: falta 'terminado: si'.")
    m = re.search(r"^descartar:(.*)$", texto, re.M)
    nums = [int(x) for x in re.findall(r"\d+", m.group(1))] if m else []
    if len(nums) != len(set(nums)):
        raise SystemExit("Hay números repetidos en descartar.")
    return set(nums)


def intercalar(filas, clave):
    """Round-robin entre grupos (orden de primera aparición), conservando el orden dentro de cada grupo."""
    grupos = defaultdict(list)
    for f in filas:
        grupos[clave(f)].append(f)
    colas, out = list(grupos.values()), []
    while any(colas):
        for q in colas:
            if q:
                out.append(q.pop(0))
    return out


def seleccion_final(tope):
    with open(SALIDA / "indice.csv", newline="", encoding="utf-8") as fh:
        indice = list(csv.DictReader(fh))
    descartar = leer_revision()
    if descartar - {int(r["numero"]) for r in indice}:
        raise SystemExit(f"Números fuera del índice: {sorted(descartar - {int(r['numero']) for r in indice})}")
    aprobadas = {c: [r for r in indice if r["clase"] == c and int(r["numero"]) not in descartar]
                 for c in ("papel_carton", "generales")}
    for c, fs in aprobadas.items():
        for r in fs:
            p = RAIZ / r["ruta"]
            ok_raiz = (PUBLICO / "train" / c) in p.parents or (EXTERNO / "train" / c) in p.parents
            if not p.exists() or not ok_raiz:
                raise SystemExit(f"Ruta no válida (debe existir y estar en publico/train o externo/train de {c}): {p}")
    n = min(tope, *(len(v) for v in aprobadas.values()))
    filas = []
    for c, fs in aprobadas.items():
        orden = []
        for bloque in ([r for r in fs if r["tipo"] in PRIMERO.get(c, ())], [r for r in fs if r["tipo"] not in PRIMERO.get(c, ())]):
            por_familia = intercalar(bloque, lambda r: r["fuente"])          # fuentes alternadas
            orden += intercalar(por_familia, lambda r: FAMILIA.get(r["tipo"], r["tipo"]))
        filas += orden[:n]
        cuenta = defaultdict(int)
        for r in orden[:n]:
            cuenta[f"{r['fuente']}/{r['tipo']}"] += 1
        print(f"{c}: aprobadas {len(fs)}, entran {n}: " + ", ".join(f"{k} {v}" for k, v in sorted(cuenta.items())))
    with open(SELECCION, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["numero", "clase", "fuente", "tipo", "categoria", "ruta"])
        w.writeheader()
        w.writerows(filas)
    print(f"Lista para entrenar.py --seleccion: {SELECCION.relative_to(RAIZ)} ({len(filas)} rutas)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=int, metavar="N", help="tras la revisión: N por clase como máximo (mismo n)")
    args = ap.parse_args()
    if args.final:
        return seleccion_final(args.final)
    os.chdir(RAIZ)   # Ultralytics busca mobileclip2_b.ts (texto de YOLOE) en el directorio actual
    SALIDA.mkdir(parents=True, exist_ok=True)
    replay = muestra_replay()
    cat_ext = categorias_externas()
    print("Banco anti-leakage (juliaca, val_camara, publico val/test)...", flush=True)
    rutas_ban, banco = banco_leakage()

    todas, elegidas_por_clase, dedup = [], {}, []
    for clase in ("papel_carton", "generales"):
        pool = imagenes(PUBLICO / "train" / clase) + imagenes(EXTERNO / "train" / clase)
        filas, vivas = [], []
        for p in pool:
            f = dict(clase=clase, fuente=fuente_de(p), ruta=p.relative_to(RAIZ).as_posix(), ruta_abs=str(p),
                     categoria=cat_ext.get(p.name, ""), motivo="")
            if f["fuente"] == "taco":
                f["motivo"] = "ya en la receta (taco)"
            elif p in replay:
                f["motivo"] = "en la muestra de replay"
            else:
                im = cv2.imread(str(p))
                if im is None or min(im.shape[:2]) < LADO_MIN:
                    f["motivo"] = f"lado < {LADO_MIN} px"
            filas.append(f)
            if not f["motivo"]:
                vivas.append(f)
        print(f"{clase}: pool {len(pool)}, al detector {len(vivas)}", flush=True)
        dets = detectar(clase, [Path(f["ruta_abs"]) for f in vivas])
        for f in vivas:
            d = dets[Path(f["ruta_abs"])]
            taco = f["categoria"] in TIPO_TACO and f["fuente"] == "tacox"
            if taco:   # tipo anotado a mano en TACO; el detector solo cuenta objetos y manos
                d = d or dict(conf=0.0, tipo="", area=1.0, n_obj=0, caja=None, mano=False)
                if d["tipo"].startswith("NO:") or d["conf"] < CONF_OBJ:
                    d = dict(d, conf=CONF_OBJ, area=1.0, caja=None)
                d = dict(d, tipo=TIPO_TACO[f["categoria"]])
            elif d is None or d["conf"] < CONF_OBJ:
                f["motivo"] = "sin objeto del tipo buscado"
                continue
            elif d["tipo"].startswith("NO:"):
                f.update(tipo=d["tipo"], conf=round(d["conf"], 3))
                f["motivo"] = f"objeto de otro tipo ({d['tipo'][3:]})"
                continue
            im = cv2.imread(f["ruta_abs"])
            h, w = im.shape[:2]
            std_borde, contraste = medir_fondo(im, d["caja"] or [0, 0, w, h])
            f.update(tipo=d["tipo"], conf=round(d["conf"], 3), area=round(d["area"], 3), n_obj=d["n_obj"],
                     mano=int(d["mano"]), std_borde=round(std_borde, 1), contraste=round(contraste, 1))
            if d["n_obj"] > MAX_OBJ:
                f["motivo"] = f"más de {MAX_OBJ} objetos"
            elif not taco and d["area"] < AREA_MIN:
                f["motivo"] = "objeto chico o lejano"
            elif f["fuente"] == "publico" and std_borde < BORDE_LISO:
                f["motivo"] = "fondo liso (estudio)"
            elif f["fuente"] == "publico" and d["area"] >= AREA_TEXTURA and contraste < CONTRASTE_TEXTURA:
                f["motivo"] = "textura a pantalla completa"
            else:
                f["puntaje"] = round(d["conf"] * (1.0 if taco or 0.08 <= d["area"] <= 0.92 else 0.7)
                                     * (BONUS_MANO if d["mano"] else 1.0), 4)

        # anti-leakage y casi duplicados entre candidatas (mejor puntaje primero)
        cand = sorted((f for f in filas if "puntaje" in f and not f["motivo"]), key=lambda f: -f["puntaje"])
        hashes_cand = []
        for f in cand:
            h = dhash(Path(f["ruta_abs"]))
            dd = dist(banco, h)
            if len(dd) and dd.min() <= DHASH_MAX:
                j = int(dd.argmin())
                f["motivo"] = "casi igual a juliaca/val_camara/publico val-test"
                dedup.append([f["ruta"], rutas_ban[j].relative_to(RAIZ).as_posix(), int(dd[j])])
                continue
            dc = dist(np.array([x for x, _ in hashes_cand], dtype=np.uint64), h)
            if len(dc) and dc.min() <= DHASH_MAX:
                j = int(dc.argmin())
                f["motivo"] = "casi duplicada de otra candidata"
                dedup.append([f["ruta"], hashes_cand[j][1], int(dc[j])])
                continue
            hashes_cand.append((h, f["ruta"]))

        # cupos por fuente y round-robin por tipo
        ok = [f for f in cand if not f["motivo"]]
        elegidas, resto = [], OBJETIVO
        for fuente, cupo in CUPOS[clase]:
            por_tipo = defaultdict(list)
            for f in ok:
                if f["fuente"] == fuente:
                    por_tipo[f["tipo"]].append(f)
            tomar = min(cupo, resto)
            colas = [por_tipo[t] for t in sorted(por_tipo)]
            tomadas = []
            while len(tomadas) < tomar and any(colas):
                for q in colas:
                    if q and len(tomadas) < tomar:
                        tomadas.append(q.pop(0))
            elegidas += tomadas
            resto -= len(tomadas)
        for f in ok:
            if f not in elegidas:
                f["motivo"] = "fuera del cupo (candidata válida)"
        elegidas_por_clase[clase] = elegidas
        todas += filas

    numero = 0
    for clase in ("papel_carton", "generales"):
        for f in elegidas_por_clase[clase]:
            numero += 1
            f["numero"] = numero
            f["motivo"] = "candidata"

    idx = SALIDA / "indice.csv"
    nuevas = [(f["numero"], f["clase"], f["ruta"]) for c in ("papel_carton", "generales") for f in elegidas_por_clase[c]]
    if idx.exists():
        with open(idx, newline="", encoding="utf-8") as fh:
            viejas = [(int(r["numero"]), r["clase"], r["ruta"]) for r in csv.DictReader(fh)]
        if viejas != nuevas:
            raise SystemExit(f"{idx.relative_to(RAIZ)} ya existe y no coincide con las candidatas de ahora: los números "
                             "anotados no valdrían. Avisa antes de regenerar (borra reportes/seleccion_curada/ a mano).")
    campos = ["numero", "clase", "fuente", "tipo", "categoria", "ruta", "conf", "area", "n_obj", "mano", "std_borde",
              "contraste", "puntaje", "motivo"]
    with open(SALIDA / "candidatas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(todas)
    with open(SALIDA / "dedup.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidata", "parecida_a", "dhash"])
        w.writerows(dedup)
    with open(idx, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["numero", "clase", "fuente", "tipo", "categoria", "ruta"],
                           extrasaction="ignore")
        w.writeheader()
        for c in ("papel_carton", "generales"):
            w.writerows(elegidas_por_clase[c])
    for c in ("papel_carton", "generales"):
        mosaicos(elegidas_por_clase[c], c)
    if not PLANTILLA.exists():
        np_, ng = len(elegidas_por_clase["papel_carton"]), len(elegidas_por_clase["generales"])
        PLANTILLA.write_text(PLANTILLA_TXT.format(n=np_ + ng, a=1, b=np_, c=np_ + 1, d=np_ + ng), encoding="utf-8")

    # resumen
    for c in ("papel_carton", "generales"):
        fil = [f for f in todas if f["clase"] == c]
        print(f"\n== {c}: pool {len(fil)}")
        motivos = defaultdict(int)
        for f in fil:
            motivos[(f["fuente"], f["motivo"])] += 1
        for (fu, m), n in sorted(motivos.items()):
            print(f"  {fu:10s} {m:50s} {n:5d}")
        tipos = defaultdict(int)
        for f in elegidas_por_clase[c]:
            tipos[(f["fuente"], f["tipo"])] += 1
        print("  candidatas por fuente/tipo: " + ", ".join(f"{fu}/{t} {n}" for (fu, t), n in sorted(tipos.items())))
    print(f"\nSalidas en {SALIDA.relative_to(RAIZ)} y plantilla {PLANTILLA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
