"""Predicción de cada modelo de etapa 2 para las 3 fotos de la bolsa de panetón (juliaca/val/generales).
Guarda reportes/etapa2_ext_paneton_val.json con los números tal cual salieron.
Uso: python scripts/predecir_paneton_val.py  (solo juliaca/val; nunca test)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comun import JULIACA, RAIZ, imagenes
from ultralytics import YOLO

fotos = imagenes(JULIACA / "val" / "generales" / "obj004_bolsa_paneton_milano")
salida = {"objeto": "juliaca/val/generales/obj004_bolsa_paneton_milano", "clase_real": "generales", "modelos": {}}
for exp in ("e2_replay2", "e2_replay2_v2", "e2_ext", "e2_ext_full"):
    pesos = json.loads((RAIZ / "reportes" / "experimentos" / f"{exp}.json").read_text(encoding="utf-8"))["modelo"]
    m = YOLO(pesos)
    filas = []
    for p, r in zip(fotos, m.predict([str(f) for f in fotos], imgsz=224, verbose=False, device=0)):
        pr = r.probs.data.cpu().numpy()
        top = pr.argsort()[::-1][:3]
        filas.append({"foto": p.name, "prediccion": r.names[int(top[0])],
                      "top3": [[r.names[int(k)], round(float(pr[k]), 4)] for k in top]})
        print(f"{exp:14s} {p.name:42s} -> {filas[-1]['top3']}")
    salida["modelos"][exp] = {"pesos": Path(pesos).name, "fotos": filas}
(RAIZ / "reportes" / "etapa2_ext_paneton_val.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
