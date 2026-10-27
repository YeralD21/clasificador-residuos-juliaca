"""Corre experimentos en orden y los compara SOLO con validación (nunca con test).

  python scripts/experimentar.py --etapa 1
  python scripts/experimentar.py --etapa 2 --pesos modelos/etapa1_20261024_1434.pt
  python scripts/experimentar.py --etapa 1 --solo e1_base,e1_drop02     # solo algunos
  python scripts/experimentar.py --listar                                # ver la tabla sin entrenar

Diseño: se cambia UN factor a la vez respecto a la configuración base, para poder explicar
en el informe qué efecto tuvo cada cambio. Cada experimento imprime su línea por época,
guarda su modelo en modelos/experimentos/ (no se sube a git) y su resumen en
reportes/experimentos/<nombre>.json. La tabla final queda en reportes/experimentos.csv.
Los experimentos ya hechos se saltan (usa --repetir para volver a correrlos).

Elegido el ganador por F1 macro en val, la evaluación final en juliaca/test se hace UNA vez:
  python scripts/evaluar.py --pesos modelos/experimentos/<ganador>.pt --split test
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from comun import RAIZ

# nombre: (descripción, argumentos para entrenar.py)
EXPERIMENTOS = {
    1: {
        "e1_base":    ("yolo26s-cls, 224 px, sin dropout (referencia)", []),
        "e1_drop02":  ("igual a base + dropout 0.2 (contra el sobreajuste)", ["--dropout", "0.2"]),
        "e1_s320":    ("igual a base con imágenes de 320 px", ["--imgsz", "320"]),
        "e1_m224":    ("modelo más grande yolo26m-cls", ["--pesos", "yolo26m-cls.pt"]),
        "e1_ext":     ("igual a base + datos externos en train (dataset/externo/train)", ["--externo"]),
        "e1_ext2":    ("igual a e1_ext con externo/train del 26-10 (+RealWaste, TACO extra, TrashNet)", ["--externo"]),
    },
    2: {
        "e2_base":     ("lr 1e-4, 6 capas congeladas, replay 1 (referencia)", []),
        "e2_lr3e-5":   ("lr más bajo 3e-5", ["--lr", "3e-5"]),
        "e2_lr3e-4":   ("lr más alto 3e-4", ["--lr", "3e-4"]),
        "e2_freeze0":  ("sin capas congeladas", ["--freeze", "0"]),
        "e2_freeze9":  ("todo el backbone congelado (solo se ajusta la cabeza)", ["--freeze", "9"]),
        "e2_replay0":  ("sin repaso de datos públicos", ["--replay", "0"]),
        "e2_replay2":  ("doble repaso de datos públicos", ["--replay", "2"]),
        "e2_replay2_v2": ("e2_replay2 re-corrido con juliaca/train del 25-10 (+16 fotos generales); control de e2_ext",
                          ["--replay", "2"]),
        "e2_ext":      ("igual a e2_replay2 + datos externos en train", ["--replay", "2", "--externo"]),
        "e2_ext_full": ("igual a e2_ext partiendo del modelo de e1_ext (externo en ambas etapas)",
                        ["--pesos", "@e1_ext", "--replay", "2", "--externo"]),
        "e2_ext2_full": ("igual a e2_ext_full con externo/train del 26-10, partiendo de e1_ext2",
                         ["--pesos", "@e1_ext2", "--replay", "2", "--externo"]),
        "e2_ext2_taco": ("igual a e2_ext_full (externo solo taco_*, 179) partiendo de e1_ext2",
                         ["--pesos", "@e1_ext2", "--replay", "2", "--externo", "--externo-fuentes", "taco"]),
        "e2_ext2_sinext": ("desde e1_ext2, etapa 2 sin externo (replay 2)", ["--pesos", "@e1_ext2", "--replay", "2"]),
        "e2_solo_jul": ("SIN etapa 1: desde ImageNet solo con fotos de Juliaca",
                        ["--pesos", "yolo26s-cls.pt", "--replay", "0", "--freeze", "0", "--lr", "1e-3"]),
    },
}
DIR = RAIZ / "reportes" / "experimentos"
CSV_SALIDA = RAIZ / "reportes" / "experimentos.csv"


def leer_resultados():
    filas = []
    for f in sorted(DIR.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        p, h, v = r.get("parametros", {}), r.get("historial", {}), r.get("val", {})
        filas.append({
            "experimento": r["experimento"], "etapa": r["etapa"],
            "val_datos": v.get("datos"), "val_f1_macro": v.get("f1_macro"), "val_top1": v.get("accuracy_top1"),
            "modelo_base": Path(str(p.get("model"))).name, "imgsz": p.get("imgsz"), "optimizer": p.get("optimizer"),
            "lr0": p.get("lr0"), "freeze": p.get("freeze"), "dropout": p.get("dropout"), "replay": p.get("replay", ""),
            "externo": sum((p.get("externo") or {}).values()),
            "epocas": h.get("epocas_corridas"), "mejor_epoca": h.get("mejor_epoca"), "minutos": r.get("minutos"),
            "modelo": r.get("modelo"),
        })
    return filas


def modelo_de(experimento):
    """Ruta del modelo que dejó un experimento ya corrido (para encadenar etapa 1 -> etapa 2)."""
    f = DIR / f"{experimento}.json"
    if not f.exists():
        raise SystemExit(f"Primero corre {experimento}: falta {f}")
    return json.loads(f.read_text(encoding="utf-8"))["modelo"]


def tabla(filas, etapa=None):
    filas = [f for f in filas if etapa is None or f["etapa"] == etapa]
    filas.sort(key=lambda f: (f["etapa"], -(f["val_f1_macro"] or 0), -(f["val_top1"] or 0)))
    if not filas:
        print("Todavía no hay experimentos terminados.")
        return filas
    print(f"\n{'experimento':14s} {'etapa':>5s} {'F1_val':>7s} {'top1_val':>9s} {'épocas':>7s} {'mejor':>6s} {'min':>6s}  val en")
    for f in filas:
        f1 = f"{f['val_f1_macro']:.4f}" if f["val_f1_macro"] is not None else "  -  "
        t1 = f"{f['val_top1']:.4f}" if f["val_top1"] is not None else "  -  "
        print(f"{f['experimento']:14s} {f['etapa']:5d} {f1:>7s} {t1:>9s} {str(f['epocas']):>7s} "
              f"{str(f['mejor_epoca']):>6s} {str(f['minutos']):>6s}  {f['val_datos']}")
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", type=int, choices=[1, 2])
    ap.add_argument("--pesos", help="etapa 2: modelo de la etapa 1 del que se parte")
    ap.add_argument("--solo", help="lista de experimentos separados por coma")
    ap.add_argument("--equipo", default="pc")
    ap.add_argument("--repetir", action="store_true")
    ap.add_argument("--epochs", type=int, help="opcional: limita épocas (p. ej. para una prueba rápida)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--listar", action="store_true")
    args = ap.parse_args()

    if args.listar or not args.etapa:
        for e, exps in EXPERIMENTOS.items():
            print(f"\nEtapa {e}:")
            for n, (d, a) in exps.items():
                print(f"  {n:14s} {d}   {' '.join(a)}")
        tabla(leer_resultados())
        return
    if args.etapa == 2 and not args.pesos:
        raise SystemExit("La etapa 2 necesita --pesos con el modelo de la etapa 1.")

    exps = EXPERIMENTOS[args.etapa]
    elegidos = args.solo.split(",") if args.solo else list(exps)
    for n in elegidos:
        if n not in exps:
            raise SystemExit(f"Experimento desconocido: {n}. Opciones: {list(exps)}")

    DIR.mkdir(parents=True, exist_ok=True)
    for i, n in enumerate(elegidos, 1):
        if (DIR / f"{n}.json").exists() and not args.repetir:
            print(f"\n[{i}/{len(elegidos)}] {n}: ya existe, se salta (usa --repetir para volver a correrlo)")
            continue
        desc, extra = exps[n]
        extra = [modelo_de(a[1:]) if a.startswith("@") else a for a in extra]  # @exp = modelo de ese experimento
        cmd = [sys.executable, str(RAIZ / "scripts" / "entrenar.py"), "--etapa", str(args.etapa),
               "--equipo", args.equipo, "--experimento", n, "--device", args.device]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.etapa == 2 and "--pesos" not in extra:
            cmd += ["--pesos", args.pesos]
        cmd += extra
        print(f"\n{'=' * 90}\n[{i}/{len(elegidos)}] {n}: {desc}\n  {' '.join(cmd[1:])}\n{'=' * 90}", flush=True)
        r = subprocess.run(cmd, cwd=RAIZ)
        if r.returncode != 0:
            print(f"[ERROR] {n} terminó con código {r.returncode}; sigo con el siguiente.")

    filas = tabla(leer_resultados(), args.etapa)
    todas = leer_resultados()
    if todas:
        with open(CSV_SALIDA, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(todas[0]))
            w.writeheader()
            w.writerows(todas)
        print(f"\nTabla completa: {CSV_SALIDA}")
    if filas:
        g = filas[0]
        print(f"\nMejor en validación (F1 macro): {g['experimento']} -> {g['modelo']}")
        print("Decide TÚ el ganador (puedes preferir uno casi igual de bueno pero más simple o rápido).")
        if args.etapa == 2:
            print("Evaluación final, UNA sola vez:\n"
                  f"  python scripts/evaluar.py --pesos {g['modelo']} --split test")


if __name__ == "__main__":
    main()
