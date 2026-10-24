# CLAUDE.md — Clasificador de residuos Juliaca (NTP 900.058:2019)

Clasificación de imagen (un residuo por imagen) con Ultralytics YOLO26-cls (PyTorch), en 2 etapas de fine-tuning.
Lee README.md para el plan completo. Comandos desde la raíz del repo: `python scripts/<script>.py`.

## Clases (nombres de carpeta exactos, orden alfabético = índices del modelo)
generales, metales, organico, papel_carton, peligrosos, plasticos, sin_residuo, vidrio
El modelo clasifica MATERIAL; el color es el tacho donde va (ver tabla en README). Nunca nombres carpetas por color.

## Reglas de integridad (OBLIGATORIAS)
1. NUNCA generes, sintetices ni descargues imágenes por tu cuenta para rellenar clases. Solo fotos reales que el usuario coloca.
2. NUNCA muevas, copies, renombres ni borres archivos dentro de `dataset/juliaca/`. La clase y el split de cada foto los decide el usuario a mano. Si detectas un problema (leakage, foto en clase dudosa), repórtalo y detente.
3. La partición de `dataset/juliaca` es MANUAL y por objeto (`<split>/<clase>/<objeto>/`). Nunca propongas ni ejecutes un split aleatorio sobre ella.
4. En `class_mapping.csv` la columna `aprobado` la cambia SOLO el usuario. No la edites.
5. `dataset/juliaca/test` solo se usa en la evaluación final. No lo uses para entrenar, elegir épocas ni ajustar hiperparámetros.
6. Nunca simules un entrenamiento ni escribas métricas que no salgan de una ejecución real. Copia números de `reportes/*.json` o `runs/*/results.csv`.
7. Antes de entrenar, ejecuta `python scripts/auditar.py`. Si hay ERRORES, detente y muéstralos.
8. No borres nada en `dataset/crudo/`. `dataset/publico/` y `dataset/_etapa2/` son generados y se pueden regenerar.
9. Los experimentos se comparan SOLO con validación (F1 macro). dataset/juliaca/test se evalúa UNA sola vez, con el modelo final que yo elija, mediante `python scripts/evaluar.py --split test`. Nunca uses --split test ni --evaluar-test sin que te lo pida explícitamente.

## Hardware
- PC (entrenamiento): RTX 3060 12GB → `--equipo pc` (batch 64, imgsz 224)
- Laptop (inferencia con MX Brio): RTX 3050 4GB → `--equipo laptop` (batch 32)
- Mismas versiones en ambos: `requirements-lock.txt` + `scripts/verificar_entorno.py`.
