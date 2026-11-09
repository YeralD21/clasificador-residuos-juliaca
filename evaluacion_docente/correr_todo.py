"""Corre todo el benchmark de nuevo. Úsalo cada vez que añadas zips a Descargas:

  .venv\\Scripts\\python.exe -I evaluacion_docente\\correr_todo.py

0. reconstruir_v1i.py : reconstruye las 238 de garbage.v1i desde TrashNet (verificadas contra el gris del docente)
1. preparar_test.py   : extrae zips nuevos de ~/Downloads e indexa las 1.455 imágenes
2. fuga_dhash.py      : chequeo de fuga (dHash <= 4) contra el zip de publico, publico, externo, juliaca
3. evaluar_docente.py : etapa2_j1028_s43 y etapa2_ext_full, variantes estricta y top2
4. escribe salidas/resultados.md con tablas por grupo, criterios, matriz y ranking
   (PARCIAL mientras falte algún grupo; nota completa y ranking final cuando estén las 1.455).
No entrena nada ni toca dataset/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
sys.path.insert(0, str(AQUI))
import ntp_metricas as nm  # noqa: E402

OUT = AQUI / "salidas"
MODELOS = ("etapa2_j1028_s43", "etapa2_ext_full")
NOMBRE_CRIT = {"pts_bal": "Exactitud balanceada (9 grupos)", "pts_mf1": "Macro-F1 estricta",
               "pts_ntp": "Cobertura 7 colores NTP", "pts_rob": "Robustez (gris + sin detección)",
               "pts_seg": "Seguridad (Rojo + orgánicos)"}


def correr(script, *args):
    print(f"\n>>> {script} {' '.join(args)}")
    subprocess.run([sys.executable, "-I", str(AQUI / script), *args], cwd=RAIZ, check=True)


def f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def informe():
    ix = json.loads((OUT / "indexado.json").read_text(encoding="utf-8"))
    fuga = json.loads((OUT / "fuga_dhash.json").read_text(encoding="utf-8"))
    rank = nm.leer_csv(AQUI / "datos_docente" / "ranking_dossier.csv")
    res = {m: json.loads((OUT / f"resultado_{m}.json").read_text(encoding="utf-8")) for m in MODELOS}
    L = []
    completa = all(res[m]["variantes"]["estricta"]["rubrica"]["completa"] for m in MODELOS)
    L.append(f"# Resultados {'COMPLETOS' if completa else 'PARCIALES'} (generado por correr_todo.py)\n")
    L.append(f"Imágenes de test disponibles: **{ix['n_encontrado']}/{ix['n_esperado']}**.\n")

    # --- grupos
    L.append("## Acierto por grupo (variante estricta)\n")
    L.append("| grupo | n disp./esperado | estado | " + " | ".join(MODELOS) + " | gris " + " | gris ".join(MODELOS) + " |")
    L.append("|---" * (3 + 2 * len(MODELOS)) + "|")
    for g in nm.GRUPOS:
        v0 = res[MODELOS[0]]["variantes"]["estricta"]
        e = v0["rubrica"]["estado_grupos"][g]
        accs = [f(res[m]["variantes"]["estricta"]["metricas"]["grupos"][g]) for m in MODELOS]
        gr = [f(res[m]["variantes"]["estricta"]["ablacion_gris"]["acc_gris_por_grupo"].get(g)) for m in MODELOS]
        L.append(f"| {g} | {ix['por_grupo'][g]['encontradas']}/{ix['por_grupo'][g]['esperadas']} | {e} | "
                 + " | ".join(accs) + " | " + " | ".join(gr) + " |")
    L.append("")

    # --- criterios
    for m in MODELOS:
        L.append(f"## {m}: rúbrica\n")
        L.append("| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |")
        L.append("|---|---|---|---|---|---|---|---|")
        filas = []
        for k, mx in nm.MAX.items():
            ce = res[m]["variantes"]["estricta"]["rubrica"]["criterios"][k]
            cs = res[m]["variantes"]["estricta"]["rubrica_sin_topes"]["criterios"][k]
            ct = res[m]["variantes"]["top2"]["rubrica"]["criterios"][k]
            perd = None if ce["pts"] is None else mx - ce["en_juego"] - ce["pts"]
            filas.append((perd if perd is not None else -1, k, mx, ce, cs, ct, perd))
        for _, k, mx, ce, cs, ct, perd in sorted(filas, key=lambda x: -x[0]):
            L.append(f"| {NOMBRE_CRIT[k]} | {mx:g} | {f(ce['pts'], 3)} | {f(cs['pts'], 3)} | {f(ct['pts'], 3)} | "
                     f"{ce['estado']} | {f(ce['en_juego'], 3)} | {f(perd, 3)} |")
        for var in ("estricta", "top2"):
            r = res[m]["variantes"][var]["rubrica"]
            if r["completa"]:
                L.append(f"\n**{var}: nota {r['nota_20']} / 20**")
            else:
                L.append(f"\n**{var}: PARCIAL** — asegurados {r['pts_asegurados']:.3f}; quedan {r['pts_en_juego']:.3f} "
                         f"pts sin evaluar (no se estiman).")
        mt = res[m]["variantes"]["estricta"]["metricas"]
        ab = res[m]["variantes"]["estricta"]["ablacion_gris"]
        L.append(f"\nOrgánicos {f(mt['organico'])} · Rojo recall {f(mt['rojo_rec'])} · precisión {f(mt['rojo_prec'])} · "
                 f"F1 {f(mt['rojo_f1'])} · sin detección (sin_residuo, disponibles) {f(mt['sin_det'], 2)} % · "
                 f"gris v1i {f(ab['acc_gris'])} · color v1i {f(ab['acc_color'])} · caída {f(ab['caida_pp'], 2)} pp\n")
        cm = res[m]["variantes"]["estricta"]["matriz_confusion"]
        L.append("Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):\n")
        L.append("| grupo | " + " | ".join(c.replace(nm.SIN_DET, "Sin det.") for c in cm["columnas"]) + " |")
        L.append("|---" * (len(cm["columnas"]) + 1) + "|")
        for g, row in cm["filas"].items():
            if sum(row.values()):
                L.append(f"| {g} | " + " | ".join(str(row[c]) for c in cm["columnas"]) + " |")
        L.append("")

    # --- ranking
    if completa:
        L.append("## Ranking completo (nota del docente vs mi nota reproducida)\n")
        filas = [(r["modelo"].split("_")[0], float(r["nota_20"])) for r in rank]
        filas += [(f"**{m}**", res[m]["variantes"]["estricta"]["rubrica"]["nota_20"]) for m in MODELOS]
    else:
        L.append("## Ranking PARCIAL comparable\n")
        L.append("Mismos criterios calculables para todos: aporte de los grupos completos a `pts_bal` + `pts_ntp` + "
                 "`pts_seg` (para los 11 modelos se calcula con sus aciertos por grupo de `ranking_dossier.csv`). "
                 "No es la nota: faltan los criterios pendientes.\n")
        comp = [g for g, e in res[MODELOS[0]]["variantes"]["estricta"]["rubrica"]["estado_grupos"].items()
                if e == "completo"]
        seg_ok = all(g in comp for g in nm.GRUPOS_SEG)
        filas = []
        for r in rank:
            s = sum(8 / 9 * float(r["g::" + g]) for g in comp) + float(r["pts_ntp"]) + (float(r["pts_seg"]) if seg_ok else 0)
            filas.append((r["modelo"].split("_")[0] + f" (nota {r['nota_20']})", s))
        for m in MODELOS:
            c = res[m]["variantes"]["estricta"]["rubrica"]["criterios"]
            filas.append((f"**{m}**", sum(c[k]["pts"] for k in ("pts_bal", "pts_ntp", "pts_seg") if c[k]["pts"] is not None)))
    L.append("| # | modelo | puntos |")
    L.append("|---|---|---|")
    for i, (n, s) in enumerate(sorted(filas, key=lambda x: -x[1]), 1):
        L.append(f"| {i} | {n} | {s:.3f} |" if not completa else f"| {i} | {n} | {s:.1f} |")
    L.append("")

    # --- fuga
    L.append("## Chequeo de fuga (dHash ≤ 4)\n")
    L.append("Imágenes de entrenamiento comparadas: " + ", ".join(f"{k} {v}" for k, v in fuga["n_train_por_fuente"].items()) + "\n")
    rev = {}
    if (OUT / "fuga_revision_manual.csv").is_file():
        for r in nm.leer_csv(OUT / "fuga_revision_manual.csv"):
            rev[r["test"]] = r["misma_foto"]
    hits = nm.leer_csv(OUT / "fuga_dhash.csv") if (OUT / "fuga_dhash.csv").stat().st_size > 10 else []
    L.append("| grupo | n | coincidencias dHash | revisadas: misma foto | revisadas: falso positivo | sin revisar |")
    L.append("|---|---|---|---|---|---|")
    for g, v in fuga["por_grupo"].items():
        hs = [h for h in hits if h["grupo"] == g]
        si = sum(rev.get(h["test"]) == "si" for h in hs)
        no = sum(rev.get(h["test"]) == "no" for h in hs)
        L.append(f"| {g} | {v['n']} | {v['coinciden']} | {si} | {no} | {len(hs) - si - no} |")
    (OUT / "resultados.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nEscrito {OUT / 'resultados.md'}")


def main():
    global MODELOS
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-informe", action="store_true")
    ap.add_argument("--modelos", nargs="+", default=[],
                    help="pesos extra a evaluar (ruta desde la raíz); las líneas base se reevalúan solo si falta su "
                         "resultado_<modelo>.json")
    a = ap.parse_args()
    extra = {Path(p).stem: p for p in a.modelos}
    base = {m: f"modelos/{m}.pt" for m in MODELOS}
    MODELOS = tuple(base) + tuple(m for m in extra if m not in base)
    if not a.solo_informe:
        correr("reconstruir_v1i.py")
        correr("preparar_test.py")
        correr("fuga_dhash.py")
        for m, ruta in {**base, **extra}.items():
            if m in extra or not (OUT / f"resultado_{m}.json").is_file():
                correr("evaluar_docente.py", "--modelo", ruta)
    informe()


if __name__ == "__main__":
    main()
