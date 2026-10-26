"""Evalúa uno o varios modelos en VALIDACIÓN (por defecto) o en TEST (solo al final).

  python scripts/evaluar.py --pesos modelos/etapa1.pt                       # juliaca/val
  python scripts/evaluar.py --pesos modelos/etapa1.pt --datos publico       # publico/val
  python scripts/evaluar.py --pesos modelos/a.pt modelos/b.pt --datos val_camara   # set aparte de la cámara
  python scripts/evaluar.py --pesos modelos/etapa1.pt modelos/etapa2.pt --split test   # EVALUACIÓN FINAL

Regla anti-leakage: los experimentos se comparan SOLO con val. El test se usa una vez,
con el modelo ya elegido. Cada uso de juliaca/test queda registrado en
reportes/registro_test.csv (para que el informe sea transparente).

dataset/val_camara/<clase>/<objeto>/ es una evaluación aparte (objetos nuevos tomados con la
cámara de la laptop): no tiene splits, nunca se usa para entrenar ni para early stopping.
Además de las métricas guarda la predicción de cada modelo por foto en
reportes/val_camara_por_foto.csv.

Imprime accuracy, precision/recall/F1 por clase y matriz de confusión, y guarda
reportes/<modelo>_<datos>_<split>.json con los números tal cual salieron.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from comun import CLASES, JULIACA, PUBLICO, RAIZ, VAL_CAMARA, imagenes, objeto_de

DATOS = {"juliaca": JULIACA, "publico": PUBLICO, "val_camara": VAL_CAMARA}


def metricas(m, nombres):
    """m: matriz de confusión con filas = predicho, columnas = real."""
    filas, f1s = [], []
    for i in range(len(nombres)):
        tp, real, pred = m[i, i], m[:, i].sum(), m[i, :].sum()
        p = tp / pred if pred else 0.0
        r = tp / real if real else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if real:
            f1s.append(f1)
        filas.append({"clase": nombres[i], "n": int(real), "precision": round(float(p), 4),
                      "recall": round(float(r), 4), "f1": round(float(f1), 4)})
    return filas, f1s


def predecir_carpeta(model, carpeta, imgsz, batch):
    """Predice cada foto de <carpeta>/<clase>/<objeto>/ (sin splits). Devuelve matriz y filas por foto."""
    nombres = model.names
    idx = {v: k for k, v in nombres.items()}
    fotos = [(c, p) for c in CLASES for p in imagenes(carpeta / c)]
    m = np.zeros((len(nombres), len(nombres)), dtype=int)
    por_foto, top5 = [], 0
    for i in range(0, len(fotos), batch):
        lote = fotos[i:i + batch]
        for (c, p), r in zip(lote, model.predict([str(p) for _, p in lote], imgsz=imgsz, verbose=False)):
            pr = r.probs.data.cpu().numpy()
            orden = pr.argsort()[::-1]
            m[int(orden[0]), idx[c]] += 1
            top5 += idx[c] in orden[:5]
            por_foto.append({"foto": p.relative_to(carpeta).as_posix(), "clase_real": c,
                             "objeto": objeto_de(p, carpeta / c), "prediccion": nombres[int(orden[0])],
                             "prob": round(float(pr[orden[0]]), 4),
                             "top3": [[nombres[int(k)], round(float(pr[k]), 4)] for k in orden[:3]]})
    n = len(fotos)
    return m, por_foto, (np.trace(m) / n if n else 0.0), (top5 / n if n else 0.0)


def evaluar(pesos, imgsz=224, batch=32, datos=JULIACA, split="val", etiqueta=None):
    from ultralytics import YOLO

    plano = datos == VAL_CAMARA
    raiz = datos if plano else datos / split
    sueltas = [p for p in imagenes(raiz) if p.relative_to(raiz).parts[0] not in CLASES] if plano else []
    if sueltas:
        raise SystemExit(f"[ERROR] {len(sueltas)} fotos fuera de una carpeta de clase válida, p. ej. {sueltas[0]}")
    if not any(imagenes(raiz / c) for c in CLASES):
        print(f"[AVISO] {raiz} no tiene fotos todavía; no se evalúa.")
        return None
    model = YOLO(str(pesos))
    nombres = model.names
    n = len(nombres)
    por_foto = None
    if plano:
        m, por_foto, top1, top5 = predecir_carpeta(model, datos, imgsz, batch)
        nombre_datos = datos.name
    else:
        res = model.val(data=str(datos), split=split, imgsz=imgsz, batch=batch, plots=True,
                        project=str(RAIZ / "runs" / "evaluaciones"),
                        name=f"{Path(pesos).stem}_{datos.name}_{split}", exist_ok=True)
        m = np.asarray(res.confusion_matrix.matrix)[:n, :n]  # filas = predicho, columnas = real
        top1, top5 = res.top1, res.top5
        nombre_datos = f"{datos.name}/{split}"

    filas, f1s = metricas(m, nombres)
    rep = {
        "modelo": str(pesos), "datos": nombre_datos,
        "accuracy_top1": round(float(top1), 4), "accuracy_top5": round(float(top5), 4),
        "f1_macro": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "clases_con_datos": len(f1s),
        "por_clase": filas,
        "matriz_real_x_predicho": m.T.astype(int).tolist(),
        "clases": [nombres[i] for i in range(n)],
    }
    if por_foto is not None:
        rep["por_foto"] = por_foto

    titulo = etiqueta or Path(pesos).name
    print(f"\n===== {titulo} en {nombre_datos} =====")
    print(f"accuracy top1={rep['accuracy_top1']:.4f}  top5={rep['accuracy_top5']:.4f}  "
          f"F1 macro={rep['f1_macro']:.4f} (sobre {len(f1s)} clases con fotos)\n")
    print(f"{'clase':14s} {'n':>5s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for f in filas:
        print(f"{f['clase']:14s} {f['n']:5d} {f['precision']:10.3f} {f['recall']:8.3f} {f['f1']:6.3f}")
    print("\nMatriz de confusión (filas = REAL, columnas = PREDICHO):")
    print(" " * 14 + " ".join(f"{nombres[i][:6]:>6s}" for i in range(n)))
    for i in range(n):
        print(f"{nombres[i]:14s}" + " ".join(f"{int(m[j, i]):6d}" for j in range(n)))

    sufijo = datos.name if plano else f"{datos.name}_{split}"
    salida = RAIZ / "reportes" / f"{Path(pesos).stem}_{sufijo}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado: {salida}")

    if split == "test" and datos == JULIACA:
        reg = RAIZ / "reportes" / "registro_test.csv"
        nuevo = not reg.exists()
        with open(reg, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["fecha", "modelo", "accuracy_top1", "f1_macro"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), Path(pesos).name, rep["accuracy_top1"], rep["f1_macro"]])
        print(f"[REGISTRO] Uso de juliaca/test anotado en {reg}")
    return rep


def tabla_por_foto(reps):
    """Predicción de cada modelo en cada foto (val_camara) -> consola + reportes/val_camara_por_foto.csv."""
    modelos = [Path(r["modelo"]).stem for r in reps]
    salida = RAIZ / "reportes" / "val_camara_por_foto.csv"
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["foto", "clase_real", "objeto"] + [x for mo in modelos for x in (mo, f"{mo}_prob")])
        print(f"\n===== POR FOTO en val_camara (✓ = acierto) =====")
        for filas in zip(*(r["por_foto"] for r in reps)):
            base = filas[0]
            w.writerow([base["foto"], base["clase_real"], base["objeto"]] +
                       [x for fi in filas for x in (fi["prediccion"], fi["prob"])])
            celdas = "  ".join(f"{'✓' if fi['prediccion'] == base['clase_real'] else '✗'} "
                               f"{fi['prediccion']:12s} {fi['prob']:.2f}" for fi in filas)
            print(f"{base['foto']:60s} {celdas}")
    print(f"Columnas: {', '.join(modelos)}\nTabla por foto guardada: {salida}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", nargs="+", required=True)
    ap.add_argument("--datos", choices=DATOS, default="juliaca")
    ap.add_argument("--split", choices=["val", "test"], default="val",
                    help="se ignora con --datos val_camara (no tiene splits)")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if args.datos == "val_camara" and args.split == "test":
        raise SystemExit("val_camara no tiene splits: no uses --split test con ella.")
    if args.split == "test" and args.datos == "juliaca":
        print("*** EVALUACIÓN FINAL EN juliaca/test: úsala solo con el modelo ya elegido por val. ***")
    datos = DATOS[args.datos]
    reps = [r for r in (evaluar(p, args.imgsz, args.batch, datos, args.split) for p in args.pesos) if r]
    nombre = args.datos if args.datos == "val_camara" else f"{args.datos}/{args.split}"
    if len(reps) > 1:
        print(f"\n===== COMPARACIÓN en {nombre} =====")
        print(f"{'modelo':45s} {'top1':>7s} {'F1 macro':>9s}")
        for r in reps:
            print(f"{Path(r['modelo']).name:45s} {r['accuracy_top1']:7.4f} {r['f1_macro']:9.4f}")
    if reps and args.datos == "val_camara":
        tabla_por_foto(reps)


if __name__ == "__main__":
    main()
