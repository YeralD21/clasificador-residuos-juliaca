"""Pasos 4-6: evalúa un modelo YOLO-cls propio con el protocolo NTP 900.058 del docente.

Uso (desde la raíz del repo):
  python -I evaluacion_docente/evaluar_docente.py                                   # modelo por defecto: modelos/etapa2_nf_s43.pt
  python -I evaluacion_docente/evaluar_docente.py --modelo modelos/etapa2_j1028_s43.pt
  python -I evaluacion_docente/evaluar_docente.py --modelo modelos/etapa2_ext_full.pt

Lee las imágenes de evaluacion_docente/imagenes/test_docente/ (las arma descargar_roboflow.py)
y las de grises del docente en evaluacion_docente/imagenes/gris_docente_garbage_v1i/.
No toca dataset/ ni entrena nada.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

DD = AQUI / "datos_docente"
TEST = AQUI / "imagenes" / "test_docente"
GRIS = AQUI / "imagenes" / "gris_docente_garbage_v1i"
OUT = AQUI / "salidas"

COLOR = {"papel_carton": "Azul", "metales": "Amarillo", "plasticos": "Blanco", "vidrio": "Gris",
         "organico": "Marrón", "peligrosos": "Rojo", "generales": "Negro", "sin_residuo": nm.SIN_DET}
VARIANTES = ("estricta", "top2")  # (a) sin_residuo = error; (b) sin_residuo -> 2.ª clase


def referencia():
    """Filas del test del docente (orden, dataset, clase_gt...) con el nombre de archivo."""
    filas = nm.leer_csv(DD / "pred_01_best_jhoel.csv")
    for f in filas:
        partes = f["path"].replace("\\", "/").split("/")
        f["archivo"] = partes[-1]
        f["carpeta_clase"] = partes[-2]
    return filas


def localizar(filas, raiz):
    """Busca cada imagen en raiz/<dataset>/<clase>/<archivo> (estructura de descargar_roboflow.py)."""
    faltan = []
    for f in filas:
        p = raiz / f["dataset"] / f["carpeta_clase"] / f["archivo"]
        f["ruta_local"] = p
        if not p.is_file():
            faltan.append(p)
    return faltan


def predecir(model, rutas, imgsz, batch):
    nombres = model.names
    out = []
    for i in range(0, len(rutas), batch):
        lote = [str(p) for p in rutas[i:i + batch]]
        for r in model.predict(lote, imgsz=imgsz, verbose=False):
            pr = r.probs.data.cpu().numpy()
            o = pr.argsort()[::-1]
            out.append([(nombres[int(k)], float(pr[k])) for k in o[:3]])
    return out


def color_de(top, variante):
    c, p = top[0]
    if c == "sin_residuo" and variante == "top2":
        c, p = top[1]
    return COLOR[c], c, p


def matriz(filas):
    cols = nm.COLORES + [nm.SIN_DET]
    m = {g: {c: 0 for c in cols} for g in nm.GRUPOS}
    for f in filas:
        m[nm.grupo_de(f)][f["color_pred"]] += 1
    return cols, m


def guardar_png_matriz(cols, m, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gs = [g for g in m if sum(m[g].values())]  # grupos sin imágenes disponibles no se dibujan
    a = np.array([[m[g][c] for c in cols] for g in gs], dtype=float)
    fr = a / a.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(fr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)), [c.replace(nm.SIN_DET, "Sin det./\nsin_residuo") for c in cols])
    ax.set_yticks(range(len(gs)), [f"{g} (n={int(a[i].sum())})" for i, g in enumerate(gs)])
    for i in range(len(gs)):
        for j in range(len(cols)):
            if a[i, j]:
                ax.text(j, i, f"{int(a[i, j])}\n{fr[i, j]:.0%}", ha="center", va="center", fontsize=7,
                        color="white" if fr[i, j] > 0.5 else "black")
    ax.set_xlabel("Color predicho")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="modelos/etapa2_nf_s43.pt",
                    help="por defecto el modelo actual; alternativa: modelos/etapa2_j1028_s43.pt")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--test", default=str(TEST))
    a = ap.parse_args()

    from ultralytics import YOLO
    pesos = (RAIZ / a.modelo) if not Path(a.modelo).is_absolute() else Path(a.modelo)
    nombre = pesos.stem
    OUT.mkdir(exist_ok=True)

    ref_total = referencia()
    n_esperado = Counter(nm.grupo_de(f) for f in ref_total)
    faltan = localizar(ref_total, Path(a.test))
    ref = [f for f in ref_total if f["ruta_local"].is_file()]
    if faltan:
        print(f"AVISO: faltan {len(faltan)} de {len(ref_total)} imágenes -> evaluación PARCIAL "
              f"({len(ref)} disponibles). Ver salidas/faltantes.csv")
    v1i_gris = [f for f in ref_total if f["dataset"].startswith("garbage.v1i")]
    gris = [GRIS / f["carpeta_clase"] / f["archivo"] for f in v1i_gris]
    if not all(p.is_file() for p in gris):
        sys.exit(f"ERROR: faltan imágenes en {GRIS}")
    leak = set()
    rev = OUT / "fuga_revision_manual.csv"
    if rev.is_file():
        leak = {r["test"] for r in nm.leer_csv(rev) if r["misma_foto"] == "si"}

    model = YOLO(str(pesos))
    assert set(model.names.values()) == set(COLOR), model.names
    top_test = predecir(model, [f["ruta_local"] for f in ref], a.imgsz, a.batch)
    top_gris = predecir(model, gris, a.imgsz, a.batch)

    resumen = {"modelo": nombre, "pesos": str(pesos), "imgsz": a.imgsz, "n_test_esperado": len(ref_total),
               "n_test_disponible": len(ref), "n_gris": len(gris), "variantes": {}}
    for var in VARIANTES:
        filas = []
        for f, top in zip(ref, top_test):
            col, cls, p = color_de(top, var)
            filas.append({"path": f["path"], "dataset": f["dataset"], "clase_gt": f["clase_gt"],
                          "colores_ok": f["colores_ok"], "tipo_gt": f["tipo_gt"], "clase_pred": cls,
                          "conf": round(p, 6), "color_pred": col,
                          "correcto": col in f["colores_ok"].split("|"),
                          "top1": top[0][0], "p_top1": round(top[0][1], 6),
                          "top2": top[1][0], "p_top2": round(top[1][1], 6)})
        # ablación en gris sobre las mismas 238 imágenes de garbage.v1i (las en gris son del docente)
        ok_color = [r["correcto"] for r in filas if r["dataset"].startswith("garbage.v1i")]
        ok_gris = [color_de(t, var)[0] in f["colores_ok"].split("|") for f, t in zip(v1i_gris, top_gris)]
        acc_g = float(np.mean(ok_gris))
        gris_g = {}
        for f, ok in zip(v1i_gris, ok_gris):
            gris_g.setdefault(nm.grupo_de(f), []).append(ok)
        gris_g = {g: float(np.mean(v)) for g, v in gris_g.items()}
        if len(ok_color) == len(v1i_gris):
            acc_c = float(np.mean(ok_color))
            caida = 100 * (acc_c - acc_g)
        else:
            acc_c = caida = None  # falta garbage.v1i a color: no se calcula la caída

        rp, m = nm.rubrica_parcial(filas, n_esperado, caida, 7, topes=True)
        rp_st, _ = nm.rubrica_parcial(filas, n_esperado, caida, 7, topes=False)
        cols, cm = matriz(filas)
        # sensibilidad: grupos sin las imágenes que son la misma foto que una de entrenamiento
        sin_fuga = {}
        for g in nm.GRUPOS:
            v = [r["correcto"] for r in filas if nm.grupo_de(r) == g and r["path"].replace("\\", "/").split("/")[-1] not in leak]
            sin_fuga[g] = {"n": len(v), "acc": float(np.mean(v)) if v else None}

        with open(OUT / f"pred_{nombre}_{var}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        guardar_png_matriz(cols, cm, f"{nombre} · variante {var} · grupo NTP vs color predicho",
                           OUT / f"cm_{nombre}_{var}.png")
        resumen["variantes"][var] = {
            "metricas": m, "rubrica": rp, "rubrica_sin_topes": rp_st,
            "ablacion_gris": {"acc_color": acc_c, "acc_gris": acc_g, "caida_pp": caida, "acc_gris_por_grupo": gris_g},
            "matriz_confusion": {"columnas": cols, "filas": cm},
            "grupos_sin_fuga": sin_fuga,
            "clase_pred_cuenta": {c: sum(r["clase_pred"] == c for r in filas) for c in COLOR},
            "n_top1_sin_residuo": sum(r["top1"] == "sin_residuo" for r in filas),
        }
        tag = f"nota {rp['nota_20']}" if rp["completa"] else \
            f"PARCIAL: asegurados {rp['pts_asegurados']:.3f}, en juego {rp['pts_en_juego']:.3f}"
        print(f"[{nombre} · {var}] {tag} | org {m['organico']:.4f} rojoF1 {m['rojo_f1']:.4f} "
              f"sin_det(disp.) {m['sin_det']:.2f}% | gris v1i {acc_g:.4f}")
        for g, e in rp["estado_grupos"].items():
            acc = m["grupos"][g]
            print(f"    {g:32s} {rp['n_grupos'].get(g, 0):4d}/{n_esperado[g]:<4d} {e:9s} "
                  f"{'' if acc is None else f'{acc:.4f}'}")
    (OUT / f"resultado_{nombre}.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
