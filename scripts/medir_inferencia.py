"""Tiempo de inferencia por imagen (batch 1, imgsz 224) de uno o varios clasificadores, en GPU y en CPU.

  python scripts/medir_inferencia.py --pesos modelos/etapa2_nf_s43.pt modelos/experimentos/<otro>.pt

Usa fotos reales de dataset/val_camara (lectura y preprocesado incluidos, como en evaluar.py: model.predict sobre la
ruta), con 10 pasadas de calentamiento y luego N imágenes. Informa la mediana y el p90 en ms, el tamaño del .pt y los
parámetros. Mide en ESTE equipo: la CPU es una referencia; la laptop y la Raspberry hay que medirlas allí.
Guarda reportes/inferencia_<fecha>.json.
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from comun import RAIZ, VAL_CAMARA, imagenes


def medir(modelo, fotos, device, n_cal=10):
    for p in fotos[:n_cal]:
        modelo.predict(str(p), imgsz=224, device=device, verbose=False)
    t = []
    for p in fotos:
        t0 = time.perf_counter()
        modelo.predict(str(p), imgsz=224, device=device, verbose=False)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t)), float(np.percentile(t, 90))


def main():
    import torch
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    fotos = sorted(imagenes(VAL_CAMARA))[: a.n]
    cpu = platform.processor() or platform.machine()
    res = []
    for w in a.pesos:
        fila = {"modelo": Path(w).name, "mb": round(Path(w).stat().st_size / 1e6, 1)}
        m = YOLO(w)
        fila["params_M"] = round(sum(x.numel() for x in m.model.parameters()) / 1e6, 2)
        if torch.cuda.is_available():
            fila["gpu_ms_mediana"], fila["gpu_ms_p90"] = medir(YOLO(w), fotos, 0)
        torch.set_num_threads(max(1, torch.get_num_threads()))
        fila["cpu_ms_mediana"], fila["cpu_ms_p90"] = medir(YOLO(w), fotos, "cpu")
        res.append(fila)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in fila.items()}, flush=True)
    out = {"fecha": datetime.now().isoformat(timespec="minutes"), "n_imagenes": len(fotos), "imgsz": 224,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cpu": cpu,
           "hilos_cpu": torch.get_num_threads(), "resultados": res}
    ruta = RAIZ / "reportes" / f"inferencia_{datetime.now():%Y%m%d_%H%M}.json"
    ruta.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado {ruta}")


if __name__ == "__main__":
    main()
