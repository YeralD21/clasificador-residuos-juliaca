"""Métricas y rúbrica del benchmark NTP 900.058 del docente (reconstruidas).

Todo se calcula a partir de filas con las columnas de los pred_*.csv del docente:
dataset, clase_gt, colores_ok, tipo_gt, color_pred.
"""
import csv
from collections import Counter, OrderedDict

SIN_DET = "SIN_DETECCION"
COLORES = ["Azul", "Amarillo", "Blanco", "Gris", "Marrón", "Rojo", "Negro"]

# 9 grupos de verdad de terreno (dossier, sección "Verdad de terreno")
GRUPOS = OrderedDict([
    ("Azul (papel/cartón)", lambda d, c: d.startswith("garbage.v1i") and c in ("cardboard", "paper")),
    ("Amarillo (metal)", lambda d, c: d.startswith("garbage.v1i") and c == "metal"),
    ("Blanco (plástico)", lambda d, c: d.startswith("garbage.v1i") and c == "plastic"),
    ("Gris (vidrio)", lambda d, c: d.startswith("garbage.v1i") and c == "glass"),
    ("Marrón (Organik, grises)", lambda d, c: c == "Organik"),
    ("Inorg. reciclable (Anorganik)", lambda d, c: c == "Anorganik"),
    ("Negro/Rojo (Residu)", lambda d, c: c == "Residu"),
    ("Rojo (biocontaminado)", lambda d, c: c == "Infectious"),
    ("No-Rojo (médico general)", lambda d, c: c == "General"),
])


def leer_csv(ruta):
    with open(ruta, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def grupo_de(fila):
    for g, f in GRUPOS.items():
        if f(fila["dataset"], fila["clase_gt"]):
            return g
    raise ValueError(f"fila sin grupo: {fila['dataset']} {fila['clase_gt']}")


def correcto(fila):
    return fila["color_pred"] in fila["colores_ok"].split("|")


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def macro_f1_estricta(filas):
    """Macro-F1 sobre las filas tipo_gt == 'estricta' (un solo color correcto).
    Promedia sobre los colores presentes en la verdad de terreno estricta
    (Azul, Amarillo, Blanco, Gris, Marrón, Rojo); 'Sin detección' y Negro
    cuentan solo como falsos negativos. Verificado contra ranking.csv."""
    est = [f for f in filas if f["tipo_gt"] == "estricta"]
    labels = sorted({f["colores_ok"] for f in est})
    f1s = []
    for c in labels:
        tp = sum(f["colores_ok"] == c and f["color_pred"] == c for f in est)
        fp = sum(f["colores_ok"] != c and f["color_pred"] == c for f in est)
        fn = sum(f["colores_ok"] == c and f["color_pred"] != c for f in est)
        f1s.append(_f1(tp, fp, fn)[2])
    return sum(f1s) / len(f1s)


def metricas(filas, gris_caida_pp=None, n_colores=None):
    por_g = {g: [] for g in GRUPOS}
    for f in filas:
        por_g[grupo_de(f)].append(correcto(f))
    acc_g = {g: (sum(v) / len(v) if v else None) for g, v in por_g.items()}
    n_g = {g: len(v) for g, v in por_g.items()}
    completos = None not in acc_g.values()

    med = [f for f in filas if f["dataset"].startswith("Smart medical")]
    tp = sum(f["clase_gt"] == "Infectious" and f["color_pred"] == "Rojo" for f in med)
    fp = sum(f["clase_gt"] != "Infectious" and f["color_pred"] == "Rojo" for f in med)
    fn = sum(f["clase_gt"] == "Infectious" and f["color_pred"] != "Rojo" for f in med)
    rp, rr, rf = _f1(tp, fp, fn)

    m = OrderedDict()
    m["bal_acc"] = sum(acc_g.values()) / len(acc_g) if completos else None
    m["acc_global"] = sum(map(correcto, filas)) / len(filas)
    m["macroF1"] = macro_f1_estricta(filas) if any(f["tipo_gt"] == "estricta" for f in filas) else None
    v1i = [correcto(f) for f in filas if f["dataset"].startswith("garbage.v1i")]
    v4 = [correcto(f) for f in filas if f["dataset"].startswith("Garbage Classification")]
    m["v1i"] = sum(v1i) / len(v1i) if v1i else None
    m["v4"] = sum(v4) / len(v4) if v4 else None
    m["organico"] = acc_g["Marrón (Organik, grises)"]
    m["rojo_rec"], m["rojo_prec"], m["rojo_f1"] = rr, rp, rf
    m["sin_det"] = 100 * sum(f["color_pred"] == SIN_DET for f in filas) / len(filas)
    m["gris_caida"] = gris_caida_pp
    m["n_colores"] = n_colores
    m["grupos"] = acc_g
    m["n_grupos"] = n_g
    return m


# ---------------------------------------------------------------- rúbrica
# Coeficientes ajustados contra ranking_dossier.csv (ver reproducir_rubrica.py).
K_GRIS = 0.05       # pts de robustez perdidos por cada pp de caída en grises
K_SINDET = 0.015    # pts de robustez perdidos por cada pp de 'sin detección'
K_ORG = 3.0         # pts de seguridad por unidad de acierto en orgánicos
K_ROJO = 28 / 9     # pts de seguridad por unidad de F1 de Rojo (= 1.4 / 0.45)


MAX = {"pts_bal": 8.0, "pts_mf1": 4.0, "pts_ntp": 3.0, "pts_rob": 3.0, "pts_seg": 2.0}
GRUPOS_SEG = ("Marrón (Organik, grises)", "Rojo (biocontaminado)", "No-Rojo (médico general)")
GRUPOS_V1I = ("Azul (papel/cartón)", "Amarillo (metal)", "Blanco (plástico)", "Gris (vidrio)")


def rubrica_parcial(filas, n_esperado, gris_caida_pp, n_colores, topes=True):
    """Rúbrica con solo los grupos disponibles. NO extrapola: cada criterio es
    'completo' (calculado igual que el docente), 'parcial' (solo el aporte asegurado
    de los grupos completos, más lo que queda en juego) o 'pendiente'.
    n_esperado: dict grupo -> n del test del docente. gris_caida_pp puede ser None."""
    n = Counter(grupo_de(f) for f in filas)
    estado = {g: ("completo" if n[g] == n_esperado[g] else "falta" if n[g] == 0 else "parcial")
              for g in GRUPOS}
    completos = [g for g in GRUPOS if estado[g] == "completo"]
    todo = len(completos) == len(GRUPOS)
    m = metricas(filas, gris_caida_pp=gris_caida_pp, n_colores=n_colores) if filas else None

    def clip(x, hi):
        return max(0.0, min(hi, x)) if topes else x
    c = OrderedDict()
    # bal: cada grupo aporta 8/9 · acierto
    aporte = sum(8 / 9 * m["grupos"][g] for g in completos)
    c["pts_bal"] = {"estado": "completo" if todo else "parcial", "pts": clip(aporte, 8),
                    "en_juego": 8 / 9 * (len(GRUPOS) - len(completos)),
                    "nota": f"{len(completos)}/9 grupos completos"}
    # macro-F1 estricta: necesita todos los grupos estrictos (incluye los 4 de garbage.v1i)
    est_ok = all(estado[g] == "completo" for g in GRUPOS_V1I + GRUPOS_SEG)
    c["pts_mf1"] = ({"estado": "completo", "pts": clip(4 * m["macroF1"], 4), "en_juego": 0.0}
                    if est_ok else {"estado": "pendiente", "pts": None, "en_juego": 4.0,
                                    "nota": "necesita los 6 grupos estrictos (4 de garbage.v1i)"})
    c["pts_ntp"] = {"estado": "completo", "pts": clip(3 * n_colores / 7, 3), "en_juego": 0.0}
    rob_ok = todo and gris_caida_pp is not None
    c["pts_rob"] = ({"estado": "completo", "en_juego": 0.0,
                     "pts": clip(3 - K_GRIS * gris_caida_pp - K_SINDET * m["sin_det"], 3)}
                    if rob_ok else {"estado": "pendiente", "pts": None, "en_juego": 3.0,
                                    "nota": "necesita garbage.v1i a color (caída en gris) y las 1.455 imágenes (% sin detección)"})
    seg_ok = all(estado[g] == "completo" for g in GRUPOS_SEG)
    c["pts_seg"] = ({"estado": "completo", "en_juego": 0.0,
                     "pts": clip(K_ORG * m["organico"] + K_ROJO * m["rojo_f1"], 2)}
                    if seg_ok else {"estado": "pendiente", "pts": None, "en_juego": 2.0})
    asegurado = sum(v["pts"] for v in c.values() if v["pts"] is not None)
    en_juego = sum(v["en_juego"] for v in c.values())
    res = {"estado_grupos": estado, "n_grupos": dict(n), "criterios": c, "completa": todo and rob_ok,
           "pts_asegurados": asegurado, "pts_en_juego": en_juego}
    if res["completa"]:
        res["nota_20"] = round(asegurado, 1)
    return res, m


def rubrica(m, topes=True):
    """Devuelve dict con pts_* y nota_20. `topes` aplica los máximos declarados
    en el dossier (8/4/3/3/2) y el mínimo 0; en los 11 modelos del docente
    ningún tope se activa, así que no se pueden verificar con sus datos."""
    def clip(x, hi):
        return max(0.0, min(hi, x)) if topes else x
    p = OrderedDict()
    p["pts_bal"] = clip(8 * m["bal_acc"], 8)
    p["pts_mf1"] = clip(4 * m["macroF1"], 4)
    p["pts_ntp"] = clip(3 * m["n_colores"] / 7, 3)
    p["pts_rob"] = clip(3 - K_GRIS * m["gris_caida"] - K_SINDET * m["sin_det"], 3)
    p["pts_seg"] = clip(K_ORG * m["organico"] + K_ROJO * m["rojo_f1"], 2)
    p["nota_20"] = round(sum(p.values()), 1)
    p["nota_20_sin_redondeo"] = sum(v for k, v in p.items() if k.startswith("pts_"))
    return p
