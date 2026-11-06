"""Paso 2: reproduce la rúbrica del docente y las métricas desde los pred_*.csv.

python -I evaluacion_docente/reproducir_rubrica.py
Escribe evaluacion_docente/salidas/rubrica_reproducida.csv y .json
"""
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

DD = AQUI / "datos_docente"
OUT = AQUI / "salidas"
OUT.mkdir(exist_ok=True)
TOL = 0.1

rank = nm.leer_csv(DD / "ranking_dossier.csv")
F = lambda r, k: float(r[k])  # noqa: E731

# --- (1) ajuste libre por mínimos cuadrados (sin suponer la fórmula) -------
X_rob = np.array([[1, F(r, "gris_caida"), F(r, "sin_det")] for r in rank])
y_rob = np.array([F(r, "pts_rob") for r in rank])
c_rob = np.linalg.lstsq(X_rob, y_rob, rcond=None)[0]
X_seg = np.array([[F(r, "organico"), F(r, "rojo_f1")] for r in rank])
y_seg = np.array([F(r, "pts_seg") for r in rank])
c_seg = np.linalg.lstsq(X_seg, y_seg, rcond=None)[0]
ajuste = {
    "pts_bal/bal_acc": [F(r, "pts_bal") / F(r, "bal_acc") for r in rank],
    "pts_mf1/macroF1": [F(r, "pts_mf1") / F(r, "macroF1") for r in rank],
    "pts_ntp*7/n_colores": [F(r, "pts_ntp") * 7 / F(r, "n_colores") for r in rank],
    "pts_rob = a + b*gris_caida + c*sin_det": c_rob.tolist(),
    "pts_rob residuo max": float(np.abs(X_rob @ c_rob - y_rob).max()),
    "pts_seg = a*organico + b*rojo_f1": c_seg.tolist(),
    "pts_seg residuo max": float(np.abs(X_seg @ c_seg - y_seg).max()),
}

# --- (2) fórmula final aplicada a las métricas del docente -----------------
filas_out, max_err = [], {}
for r in rank:
    m = {k: F(r, k) for k in ("bal_acc", "macroF1", "n_colores", "gris_caida",
                              "sin_det", "organico", "rojo_f1")}
    p = nm.rubrica(m)
    fila = {"modelo": r["modelo"], "nota_docente": F(r, "nota_20"), "nota_reprod": p["nota_20"],
            "nota_reprod_exacta": round(p["nota_20_sin_redondeo"], 4)}
    for k in ("pts_bal", "pts_mf1", "pts_ntp", "pts_rob", "pts_seg"):
        fila[f"{k}_docente"] = round(F(r, k), 4)
        fila[f"{k}_reprod"] = round(p[k], 4)
        max_err[k] = max(max_err.get(k, 0), abs(p[k] - F(r, k)))
    max_err["nota_20"] = max(max_err.get("nota_20", 0), abs(p["nota_20"] - F(r, "nota_20")))
    filas_out.append(fila)

# --- (3) métricas recalculadas desde pred_*.csv vs ranking_dossier ---------
err_met = {}
for r in rank:
    filas = nm.leer_csv(DD / f"pred_{r['modelo']}.csv")
    m = nm.metricas(filas)
    for k in ("bal_acc", "acc_global", "macroF1", "v1i", "v4", "organico",
              "rojo_rec", "rojo_prec", "rojo_f1", "sin_det"):
        err_met[k] = max(err_met.get(k, 0), abs(m[k] - F(r, k)))
    for g, v in m["grupos"].items():
        err_met["grupos"] = max(err_met.get("grupos", 0), abs(v - F(r, f"g::{g}")))
    # columna 'correcto' del docente vs nuestra regla
    dif = sum((f["correcto"] == "True") != nm.correcto(f) for f in filas)
    err_met["filas_correcto_distinto"] = err_met.get("filas_correcto_distinto", 0) + dif

# ablación: caída = 100*(acc_color - acc_gris)
abl = {a["modelo"]: a for a in nm.leer_csv(DD / "ablacion_gris.csv")}
err_met["gris_caida"] = max(abs(100 * (F(abl[r["modelo"]], "acc_color") - F(abl[r["modelo"]], "acc_gris"))
                                - F(r, "gris_caida")) for r in rank)
err_met["v1i_vs_acc_color_ablacion"] = max(abs(F(abl[r["modelo"]], "acc_color") - F(r, "v1i")) for r in rank)

with open(OUT / "rubrica_reproducida.csv", "w", encoding="utf-8", newline="") as f:
    import csv
    w = csv.DictWriter(f, fieldnames=list(filas_out[0]))
    w.writeheader()
    w.writerows(filas_out)
res = {"ajuste_libre": ajuste, "error_max_pts": max_err, "error_max_metricas": err_met,
       "tolerancia": TOL, "coincide": all(v <= TOL for v in max_err.values())}
(OUT / "rubrica_reproducida.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

print("Ajuste libre:")
for k, v in ajuste.items():
    print(f"  {k}: {np.round(v, 5).tolist() if isinstance(v, list) else round(v, 6)}")
print("Error máx. pts (fórmula final vs docente):", {k: round(v, 6) for k, v in max_err.items()})
print("Error máx. métricas (pred_*.csv vs ranking_dossier):", {k: round(v, 6) if isinstance(v, float) else v for k, v in err_met.items()})
print(f"{'modelo':28s} docente reprod exacta")
for fo in filas_out:
    print(f"{fo['modelo']:28s} {fo['nota_docente']:6.1f} {fo['nota_reprod']:6.1f} {fo['nota_reprod_exacta']:8.4f}")
print("COINCIDE (tol 0.1):", res["coincide"])
