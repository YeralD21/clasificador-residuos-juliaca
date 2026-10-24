"""Entrenamiento en 2 etapas con YOLO26-cls (PyTorch) en TU GPU.

ETAPA 1 - base general con datos públicos (dataset/publico):
  python scripts/entrenar.py --etapa 1
ETAPA 2 - fine-tuning con tus fotos de Juliaca (dataset/juliaca), partiendo del modelo de la etapa 1:
  python scripts/entrenar.py --etapa 2 --pesos modelos/etapa1_yolo26s-cls.pt
Comparación opcional para el informe (solo Juliaca, sin etapa 1):
  python scripts/entrenar.py --etapa 2 --pesos yolo26s-cls.pt --replay 0

En la etapa 2:
  - train = dataset/juliaca/train + una muestra de dataset/publico/train ("replay", para no olvidar)
  - val   = dataset/juliaca/val      (tu partición manual, tal cual)
  - test  = dataset/juliaca/test     (tu partición manual, tal cual; solo se usa al final)
  - learning rate bajo + primeras capas congeladas.
Se imprime una línea por época. Al final se evalúa SOLO en VALIDACIÓN
(etapa 1: publico/val, etapa 2: juliaca/val). El test se usa una sola vez, al final, con
`--evaluar-test` o con `scripts/evaluar.py --split test`.

Con --experimento NOMBRE el modelo va a modelos/experimentos/ (no se sube a git) y el
resumen a reportes/experimentos/NOMBRE.json (lo usa scripts/experimentar.py).
"""
import argparse
import csv
import json
import platform
import random
import shutil
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from comun import CLASES, ETAPA2, JULIACA, PUBLICO, RAIZ, SPLITS, imagenes
from evaluar import evaluar

PERFIL = {"pc": {"batch": 64, "workers": 6}, "laptop": {"batch": 32, "workers": 4}}
_t = {"ini": 0.0}


def ini_epoca(trainer):
    _t["ini"] = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def fin_epoca(trainer):
    if trainer.epoch + 1 > trainer.epochs or trainer.epoch in _t.setdefault("vistas", set()):
        return  # validación final del best.pt, no es una época nueva
    _t["vistas"].add(trainer.epoch)
    m = trainer.metrics or {}
    tl = trainer.tloss
    if tl is None:
        loss_tr = float("nan")
    elif isinstance(tl, dict):  # versiones nuevas de Ultralytics
        loss_tr = sum(float(v.sum()) if hasattr(v, "sum") else float(v) for v in tl.values())
    else:
        loss_tr = float(tl.sum()) if hasattr(tl, "sum") else float(tl)
    loss_val = sum(v for k, v in m.items() if k.startswith("val/"))
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[EPOCA {trainer.epoch + 1:3d}/{trainer.epochs}] {time.time() - _t['ini']:6.1f}s | "
          f"loss_train={loss_tr:.4f} loss_val={loss_val:.4f} | "
          f"top1_val={m.get('metrics/accuracy_top1', 0):.4f} top5_val={m.get('metrics/accuracy_top5', 0):.4f} | "
          f"lr={trainer.optimizer.param_groups[0]['lr']:.2e} | GPU_pico={gpu:.2f}GB", flush=True)


def entorno():
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_torch": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def contar(base, split):
    return {c: len(imagenes(base / split / c)) for c in CLASES}


def armar_etapa2(replay, seed):
    """Copia juliaca (sin tocar el original) + replay de publico en dataset/_etapa2."""
    rng = random.Random(seed)
    if ETAPA2.exists():
        shutil.rmtree(ETAPA2)
    for s in SPLITS:
        for c in CLASES:
            (ETAPA2 / s / c).mkdir(parents=True, exist_ok=True)
            origen = JULIACA / s / c
            for p in imagenes(origen):
                dst = ETAPA2 / s / c / "juliaca" / p.relative_to(origen)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
    resumen = {}
    for c in CLASES:
        n_jul = len(imagenes(JULIACA / "train" / c))
        pub = imagenes(PUBLICO / "train" / c)
        rng.shuffle(pub)
        k = min(len(pub), int(round(n_jul * replay))) if replay > 0 else 0
        dst = ETAPA2 / "train" / c / "publico_replay"
        dst.mkdir(parents=True, exist_ok=True)
        for p in pub[:k]:
            shutil.copy2(p, dst / p.name)
        resumen[c] = (n_jul, k)
    print("\nEtapa 2 - train = juliaca + replay público:")
    for c, (a, b) in resumen.items():
        print(f"  {c:14s} juliaca={a:4d}  replay={b:4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, choices=[1, 2], required=True)
    ap.add_argument("--pesos", default="yolo26s-cls.pt", help="etapa 1: modelo preentrenado; etapa 2: best.pt de la etapa 1")
    ap.add_argument("--equipo", choices=PERFIL, default="pc")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=1e-4, help="solo etapa 2")
    ap.add_argument("--freeze", type=int, default=6, help="solo etapa 2: capas congeladas (0-9 = backbone)")
    ap.add_argument("--replay", type=float, default=1.0, help="solo etapa 2: fotos públicas por cada foto tuya")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout de la cabeza de clasificación")
    ap.add_argument("--experimento", default=None, help="nombre del experimento (ver experimentar.py)")
    ap.add_argument("--evaluar-test", action="store_true",
                    help="evalúa también en juliaca/test (SOLO para el modelo final elegido por val)")
    ap.add_argument("--device", default="0", help="0 = primera GPU")
    args = ap.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("PyTorch no ve la GPU. Ejecuta: python scripts/verificar_entorno.py")
    _t["t0"] = time.time()
    perfil = PERFIL[args.equipo]
    batch = args.batch or perfil["batch"]
    print("Entorno:", entorno())

    comunes = dict(imgsz=args.imgsz, batch=batch, workers=perfil["workers"], device=args.device, amp=True,
                   seed=args.seed, deterministic=True, cos_lr=True, plots=True, dropout=args.dropout,
                   hsv_h=0.03, hsv_s=0.5, hsv_v=0.4,       # variar tono: que no aprenda "blanco = plástico"
                   fliplr=0.5, flipud=0.2, erasing=0.3, auto_augment="randaugment",
                   project=str(RAIZ / "runs"))

    if args.etapa == 1:
        if sum(contar(PUBLICO, "train").values()) == 0:
            raise SystemExit("dataset/publico/train está vacío. Ejecuta: python scripts/preparar_publico.py")
        datos = PUBLICO
        params = dict(data=str(datos), epochs=args.epochs or 60, patience=15, **comunes)
    else:
        tr, va = contar(JULIACA, "train"), contar(JULIACA, "val")
        if sum(tr.values()) == 0 or sum(va.values()) == 0:
            raise SystemExit("dataset/juliaca/train o val está vacío. Coloca tus fotos (ver README).")
        armar_etapa2(args.replay, args.seed)
        datos = ETAPA2
        params = dict(data=str(datos), epochs=args.epochs or 30, patience=10, optimizer="AdamW",
                      lr0=args.lr, lrf=0.1, warmup_epochs=1, freeze=args.freeze, **comunes)

    nombre = f"etapa{args.etapa}_{args.experimento or Path(args.pesos).stem}"
    model = YOLO(args.pesos)
    model.add_callback("on_train_epoch_start", ini_epoca)
    model.add_callback("on_fit_epoch_end", fin_epoca)
    model.train(name=nombre, **params)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    carpeta = RAIZ / "modelos" / ("experimentos" if args.experimento else "")
    carpeta.mkdir(parents=True, exist_ok=True)
    etiqueta = f"etapa{args.etapa}_{args.experimento + '_' if args.experimento else ''}{time.strftime('%Y%m%d_%H%M')}"
    destino = carpeta / f"{etiqueta}.pt"
    shutil.copy2(best, destino)

    # Solo los argumentos que de verdad usó esta etapa (+ los que resolvió Ultralytics)
    ta = vars(model.trainer.args)
    usados = {k: ta.get(k) for k in ("model", "data", "epochs", "patience", "imgsz", "batch", "optimizer", "lr0",
                                     "lrf", "warmup_epochs", "freeze", "dropout", "cos_lr", "seed", "hsv_h",
                                     "hsv_s", "hsv_v", "fliplr", "flipud", "erasing", "auto_augment")}
    if args.etapa == 2:
        usados["replay"] = args.replay
    filas = list(csv.DictReader(open(save_dir / "results.csv", encoding="utf-8")))
    filas = [{k.strip(): v for k, v in f.items()} for f in filas]
    mejor = max(filas, key=lambda f: float(f.get("metrics/accuracy_top1", 0) or 0)) if filas else {}
    historial = {"epocas_corridas": len(filas), "mejor_epoca": int(float(mejor.get("epoch", 0) or 0)),
                 "mejor_top1_val": float(mejor.get("metrics/accuracy_top1", 0) or 0)}
    destino.with_suffix(".json").write_text(json.dumps(
        {"etapa": args.etapa, "pesos_iniciales": args.pesos, "entorno": entorno(),
         "parametros": usados, "historial": historial}, indent=2, default=str))
    print(f"\nModelo copiado a {destino}  (+ {destino.with_suffix('.json').name})")

    datos_val = PUBLICO if args.etapa == 1 else JULIACA
    rep_val = evaluar(destino, args.imgsz, batch, datos_val, "val", etiqueta=f"ETAPA {args.etapa} (validación)")
    if args.etapa == 1 and any(imagenes(JULIACA / "val" / c) for c in CLASES):
        evaluar(destino, args.imgsz, batch, JULIACA, "val", etiqueta="ETAPA 1 en juliaca/val (referencia)")

    if args.experimento:
        d = RAIZ / "reportes" / "experimentos"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{args.experimento}.json").write_text(json.dumps({
            "experimento": args.experimento, "etapa": args.etapa, "modelo": str(destino),
            "parametros": usados, "historial": historial,
            "val": {"datos": f"{datos_val.name}/val",
                    "accuracy_top1": rep_val["accuracy_top1"] if rep_val else None,
                    "f1_macro": rep_val["f1_macro"] if rep_val else None},
            "minutos": round((time.time() - _t["t0"]) / 60, 1)}, indent=2, default=str), encoding="utf-8")

    if args.evaluar_test:
        print("\n*** EVALUACIÓN FINAL EN juliaca/test ***")
        evaluar(destino, args.imgsz, batch, JULIACA, "test", etiqueta=f"ETAPA {args.etapa} (TEST FINAL)")
    else:
        print("\n(No se evaluó en juliaca/test. Hazlo una sola vez con el modelo final: "
              f"python scripts/evaluar.py --pesos {destino.relative_to(RAIZ)} --split test)")


if __name__ == "__main__":
    main()
