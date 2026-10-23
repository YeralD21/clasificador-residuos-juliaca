"""Evalúa uno o varios modelos en dataset/juliaca/test (fotos tuyas, nunca usadas para entrenar).

  python scripts/evaluar.py --pesos modelos/etapa1.pt modelos/etapa2.pt

Imprime accuracy, precision/recall/F1 por clase y la matriz de confusión, y guarda
reportes/<modelo>.json con los números tal cual salieron (para tu informe).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from comun import CLASES, JULIACA, RAIZ, imagenes


def evaluar(pesos, imgsz=224, batch=32, datos=JULIACA, etiqueta=None):
    from ultralytics import YOLO

    if not any(imagenes(datos / "test" / c) for c in CLASES):
        print(f"[AVISO] {datos / 'test'} no tiene fotos todavía; no se evalúa.")
        return None
    model = YOLO(str(pesos))
    res = model.val(data=str(datos), split="test", imgsz=imgsz, batch=batch, plots=True,
                    project=str(RAIZ / "runs" / "evaluaciones"), name=Path(pesos).stem, exist_ok=True)
    nombres = model.names
    n = len(nombres)
    m = np.asarray(res.confusion_matrix.matrix)[:n, :n]  # filas = predicho, columnas = real

    filas, f1s = [], []
    for i in range(n):
        tp, real, pred = m[i, i], m[:, i].sum(), m[i, :].sum()
        p = tp / pred if pred else 0.0
        r = tp / real if real else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if real:
            f1s.append(f1)
        filas.append({"clase": nombres[i], "n": int(real), "precision": round(float(p), 4),
                      "recall": round(float(r), 4), "f1": round(float(f1), 4)})
    rep = {
        "modelo": str(pesos), "datos": str(datos / "test"),
        "accuracy_top1": round(float(res.top1), 4), "accuracy_top5": round(float(res.top5), 4),
        "f1_macro": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "por_clase": filas,
        "matriz_real_x_predicho": m.T.astype(int).tolist(),
        "clases": [nombres[i] for i in range(n)],
    }

    titulo = etiqueta or Path(pesos).name
    print(f"\n===== {titulo} en {datos.name}/test =====")
    print(f"accuracy top1={rep['accuracy_top1']:.4f}  top5={rep['accuracy_top5']:.4f}  F1 macro={rep['f1_macro']:.4f}\n")
    print(f"{'clase':14s} {'n':>5s} {'precision':>10s} {'recall':>8s} {'f1':>6s}")
    for f in filas:
        print(f"{f['clase']:14s} {f['n']:5d} {f['precision']:10.3f} {f['recall']:8.3f} {f['f1']:6.3f}")
    print("\nMatriz de confusión (filas = REAL, columnas = PREDICHO):")
    print(" " * 14 + " ".join(f"{nombres[i][:6]:>6s}" for i in range(n)))
    for i in range(n):
        print(f"{nombres[i]:14s}" + " ".join(f"{int(m[j, i]):6d}" for j in range(n)))

    salida = RAIZ / "reportes" / f"{Path(pesos).stem}.json"
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReporte guardado: {salida}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    reps = [r for r in (evaluar(p, args.imgsz, args.batch) for p in args.pesos) if r]
    if len(reps) > 1:
        print("\n===== COMPARACIÓN en juliaca/test =====")
        print(f"{'modelo':40s} {'top1':>7s} {'F1 macro':>9s}")
        for r in reps:
            print(f"{Path(r['modelo']).name:40s} {r['accuracy_top1']:7.4f} {r['f1_macro']:9.4f}")


if __name__ == "__main__":
    main()
