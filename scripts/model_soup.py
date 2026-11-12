"""Model soup (promedio uniforme de pesos) de varios modelos de clasificación Ultralytics compatibles.

  python scripts/model_soup.py --pesos a.pt b.pt c.pt --salida modelos/experimentos/etapa2_nf_soup.pt

Requisitos que se verifican: mismas claves y formas en el state_dict y mismas clases (mismo orden). Se promedian en
float32 todos los tensores de punto flotante (pesos y estadísticas de BatchNorm); los enteros (num_batches_tracked) se
toman del primero. El checkpoint de salida copia la estructura del primero (model en fp16, ema=None, optimizer=None),
así que carga con YOLO(...) como cualquier .pt. No entrena nada.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pesos", nargs="+", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()

    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in a.pesos]
    modelos = [c["model"].float() for c in cks]
    nombres = [m.names for m in modelos]
    if any(n != nombres[0] for n in nombres):
        raise SystemExit(f"Clases distintas: {nombres}")
    sds = [m.state_dict() for m in modelos]
    claves = list(sds[0])
    for sd in sds[1:]:
        if list(sd) != claves or any(sd[k].shape != sds[0][k].shape for k in claves):
            raise SystemExit("Los modelos no tienen la misma arquitectura")
    sopa = {}
    for k in claves:
        t = sds[0][k]
        sopa[k] = torch.stack([sd[k].float() for sd in sds]).mean(0).to(t.dtype) if t.is_floating_point() else t
    nuevo = copy.deepcopy(modelos[0])
    nuevo.load_state_dict(sopa)
    ck = dict(cks[0])
    ck["model"] = nuevo.half()
    ck["ema"] = None
    ck["optimizer"] = None
    ck["model_soup"] = {"fuentes": [str(Path(p).name) for p in a.pesos], "metodo": "promedio uniforme"}
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.salida)
    print(f"Guardado {a.salida} ({Path(a.salida).stat().st_size / 1e6:.1f} MB), SHA-1 "
          f"{hashlib.sha1(Path(a.salida).read_bytes()).hexdigest()}")
    print(f"Clases: {nombres[0]}")


if __name__ == "__main__":
    main()
