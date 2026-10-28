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

Con --externo se suma dataset/externo/train/<clase> (datos externos reales, regla 1 de CLAUDE.md)
SOLO al train: etapa 1 -> dataset/_etapa1 (publico/train + externo, val = copia de publico/val);
etapa 2 -> dataset/_etapa2/train/<clase>/externo (completo, aparte del replay). Nunca a val ni test.
Sin --externo la etapa 1 entrena directo sobre dataset/publico, como siempre.
--externo-fuentes taco realwaste ... usa solo esas fuentes, por prefijo de archivo (PREFIJO_FUENTE), sin mover nada.
Con --seleccion <csv> (solo etapa 2) se suma una lista curada y aprobada de rutas de publico/train o externo/train
(scripts/seleccionar_curados.py --final) a dataset/_etapa2/train/<clase>/seleccion, sin mover nada.

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

from comun import CLASES, ETAPA1, ETAPA2, EXTERNO, JULIACA, PUBLICO, RAIZ, imagenes
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


PREFIJO_FUENTE = {  # fuente -> prefijo del nombre de archivo en dataset/externo/train (ver datos/fuentes.md)
    "taco": "taco_",            # importar_taco.py (Crisp packet, Other plastic wrapper, Aluminium foil)
    "taco_extra": "tacox_",     # importar_publicos.py
    "realwaste": "realwaste_",
    "trashnet": "trashnet_",
}


def copiar_externo(destino_train, fuentes=None):
    """Copia dataset/externo/train/<clase> a <destino_train>/<clase>/externo. Devuelve n por clase.
    fuentes: lista de claves de PREFIJO_FUENTE; None = todo. Filtra por prefijo, sin mover nada."""
    prefijos = tuple(PREFIJO_FUENTE[f] for f in fuentes) if fuentes else None
    n = {}
    for c in CLASES:
        fotos = [p for p in imagenes(EXTERNO / "train" / c) if prefijos is None or p.name.startswith(prefijos)]
        dst = destino_train / c / "externo"
        dst.mkdir(parents=True, exist_ok=True)
        for p in fotos:
            shutil.copy2(p, dst / p.name)
        n[c] = len(fotos)
    if sum(n.values()) == 0:
        raise SystemExit("--externo activo pero dataset/externo/train está vacío"
                         + (f" para las fuentes {fuentes}." if fuentes else "."))
    return n


def armar_etapa1_externo(fuentes=None):
    """dataset/_etapa1 = publico/train + externo/train (train) y copia de publico/val (val).
    publico/test NO se copia, igual que juliaca/test en la etapa 2."""
    if ETAPA1.exists():
        shutil.rmtree(ETAPA1)
    for s in ("train", "val"):
        for c in CLASES:
            dst = ETAPA1 / s / c
            dst.mkdir(parents=True, exist_ok=True)
            for p in imagenes(PUBLICO / s / c):
                shutil.copy2(p, dst / p.name)
    n = copiar_externo(ETAPA1 / "train", fuentes)
    print("\nEtapa 1 - train = publico + externo" + (f" ({', '.join(fuentes)})" if fuentes else "") + ":")
    for c in CLASES:
        print(f"  {c:14s} publico={len(imagenes(PUBLICO / 'train' / c)):5d}  externo={n[c]:4d}")
    return n


def armar_etapa2(replay, seed, externo=False, fuentes=None, seleccion=None):
    """Copia juliaca train y val (sin tocar el original) + replay de publico en dataset/_etapa2.
    juliaca/test NO se copia: así Ultralytics nunca puede usarlo, ni siquiera como respaldo de val."""
    rng = random.Random(seed)
    if ETAPA2.exists():
        shutil.rmtree(ETAPA2)
    for s in ("train", "val"):
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
    n_ext = copiar_externo(ETAPA2 / "train", fuentes) if externo else None
    n_sel = copiar_seleccion(seleccion) if seleccion else None
    print("\nEtapa 2 - train = juliaca + replay público"
          + ((" + externo" + (f" ({', '.join(fuentes)})" if fuentes else "")) if externo else "")
          + (f" + selección ({Path(seleccion).name})" if seleccion else "") + ":")
    for c, (a, b) in resumen.items():
        print(f"  {c:14s} juliaca={a:4d}  replay={b:4d}" + (f"  externo={n_ext[c]:4d}" if externo else "")
              + (f"  seleccion={n_sel['por_clase'][c]:4d}" if seleccion else ""))
    if seleccion and n_sel["saltadas"]:
        print(f"  (selección: {len(n_sel['saltadas'])} ya estaban en train por replay o externo; no se duplican)")
    return n_ext, n_sel


def copiar_seleccion(lista):
    """Copia a _etapa2/train/<clase>/seleccion las rutas de una lista curada y aprobada (seleccionar_curados.py
    --final). Solo acepta archivos de dataset/publico/train/<clase> o dataset/externo/train/<clase> (nunca val, test
    ni juliaca); no mueve nada. Salta los que ya están en el train armado (replay o externo) para no duplicarlos."""
    with open(lista, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    ya = {p.name for c in CLASES for p in imagenes(ETAPA2 / "train" / c) if p.parent.name != "juliaca"}
    por_clase, saltadas = {c: 0 for c in CLASES}, []
    for r in filas:
        c, p = r["clase"], (RAIZ / r["ruta"]).resolve()
        if c not in CLASES or not p.is_file() or not any(
                b.resolve() in p.parents for b in (PUBLICO / "train" / c, EXTERNO / "train" / c)):
            raise SystemExit(f"--seleccion: ruta no permitida o inexistente para {c}: {r['ruta']} "
                             "(solo publico/train o externo/train de su clase)")
        if p.name in ya:
            saltadas.append(r["ruta"])
            continue
        dst = ETAPA2 / "train" / c / "seleccion"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / p.name)
        ya.add(p.name)
        por_clase[c] += 1
    return {"archivo": Path(lista).resolve().relative_to(RAIZ).as_posix(), "por_clase": por_clase,
            "saltadas": saltadas}


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
    ap.add_argument("--externo", action="store_true",
                    help="suma dataset/externo/train al train (etapa 1 y 2); nunca a val ni test")
    ap.add_argument("--externo-fuentes", nargs="+", choices=sorted(PREFIJO_FUENTE), default=None,
                    help="con --externo: solo estas fuentes (por prefijo de archivo); por defecto todas")
    ap.add_argument("--seleccion", default=None,
                    help="solo etapa 2: CSV de rutas curadas y aprobadas (seleccionar_curados.py --final) que se "
                         "suman al train; solo de publico/train o externo/train")
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

    n_sel = None
    if args.seleccion and args.etapa != 2:
        raise SystemExit("--seleccion solo se usa en la etapa 2.")
    if args.etapa == 1:
        if sum(contar(PUBLICO, "train").values()) == 0:
            raise SystemExit("dataset/publico/train está vacío. Ejecuta: python scripts/preparar_publico.py")
        n_externo = armar_etapa1_externo(args.externo_fuentes) if args.externo else None
        datos = ETAPA1 if args.externo else PUBLICO
        params = dict(data=str(datos), epochs=args.epochs or 60, patience=15, **comunes)
    else:
        tr, va = contar(JULIACA, "train"), contar(JULIACA, "val")
        if sum(tr.values()) == 0 or sum(va.values()) == 0:
            raise SystemExit("dataset/juliaca/train o val está vacío. Coloca tus fotos (ver README).")
        n_externo, n_sel = armar_etapa2(args.replay, args.seed, args.externo, args.externo_fuentes, args.seleccion)
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
    usados["externo"] = n_externo  # None = apagado; si no, n imágenes externas por clase en train
    if args.externo_fuentes:
        usados["externo_fuentes"] = args.externo_fuentes
    if n_sel:
        usados["seleccion"] = n_sel  # lista curada: archivo, n por clase y las saltadas por duplicadas
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
