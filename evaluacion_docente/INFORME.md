# Réplica del benchmark NTP 900.058 del docente

Estado (2026-11-08): **evaluación COMPLETA — 1.455 de 1.455 imágenes, con garbage.v1i RECONSTRUIDO** (ver §0).
Las secciones 1–7 son el informe parcial del 2026-11-05 (1.217 imágenes) y se dejan como estaban; los números
vigentes son los de §0 y [`salidas/resultados.md`](salidas/resultados.md).

## 0. Actualización 2026-11-08: línea base completa (garbage.v1i reconstruido)

**garbage.v1i se reconstruyó; no es el archivo original del docente.** El zip `garbage.v1i.folder` no está
disponible y "Garbage Collection.v1i" (`amcsalazar/garbage-collection-celih`) es **otro** proyecto: **0 de los 238
nombres** del benchmark están en él (ni en test ni en train), y sus conteos de test no coinciden (45/57/41/54/46
frente a 44/57/43/57/37). Los nombres del benchmark son de TrashNet (`cardboard101_jpg.rf.<hash>.jpg`), así que
`reconstruir_v1i.py`:
1. toma cada original de `dataset/externo/_fuentes/trashnet/extraido/dataset-resized/<clase>/<nombre>.jpg`
   (quitando `_jpg.rf.<hash>`);
2. lo redimensiona (stretch) al tamaño de su versión en gris del docente (640×640, la misma exportación de Roboflow);
3. lo verifica contra esa versión en gris: dHash (probando las 8 rotaciones/espejos) **o** SSIM en grises ≥ 0.80.

Resultado (`salidas/reconstruccion_v1i.{csv,json}`): **238 de 238 pares coinciden**, todos en la orientación
original; dHash 0–4 en 237 (119 con distancia 0) y 5 en uno (`metal250`, SSIM 0.9905); SSIM entre 0.971 y 0.997.
Ninguna imagen en gris llegó ilegible: no hay ninguna marcada "sin verificar". Diferencia posible con el original
del docente: la compresión JPEG y el remuestreo de Roboflow (se redimensionó con bilineal y se guardó a calidad 95).
`correr_todo.py` corre ahora `reconstruir_v1i.py` antes de `preparar_test.py`. `datos_docente/` se reconstruyó en la
PC desde `resultados.rar` (no se sube a git).

**Nota completa (rúbrica reproducida, con topes = sin topes):**

| criterio | máx | etapa2_j1028_s43 | etapa2_ext_full |
|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 5.268 | 5.044 |
| Macro-F1 estricta | 4 | 1.895 | 1.915 |
| Robustez (gris + sin det.) | 3 | 2.723 | 2.635 |
| Cobertura 7 colores | 3 | 3.000 | 3.000 |
| Seguridad (Rojo + orgánicos) | 2 | 0.521 | 0.722 |
| **Nota /20 (estricta = top2)** | 20 | **13.4** | **13.3** |
| **Puesto entre 13** (11 del curso + mis 2) | | **4.º** (tras 15: 16.0, 01: 14.0, 21: 13.6) | 5.º |

garbage.v1i a color, j1028: papel/cartón 0.9505, metal 0.9767, plástico 0.8649, vidrio 0.8772 (global 0.9244); en
gris 0.8697, caída **5.46 pp**. Los demás grupos no cambian respecto a §5. Lo que más puntos cuesta sigue siendo
biocontaminado (0.034) y orgánicos en gris (0.107).

**Fuga (dHash ≤ 4, `salidas/fuga_dhash.csv`; ahora contra `publico` 12.220, `externo/train` 5.684 y `juliaca` 624;
el zip `dataset_ntp900058.zip` no está en esta PC):**
- `externo/train/trashnet_*` contiene 35 de metal, 49 de vidrio y 4 de papel/cartón de garbage.v1i (son los mismos
  originales de TrashNet). **No afecta a la línea base:** `etapa2_j1028_s43` y `etapa2_ext_full` usaron externo
  solo `taco` en las dos etapas. Cualquier experimento nuevo debe excluir `trashnet_`.
- `publico/train` (lo que sí usa la línea base) coincide con 29 de las 238 de garbage.v1i (papel/cartón 12, metal 7,
  vidrio 7, plástico 3) — **sin revisar a ojo**; SSIM automático ≥ 0.80 en 23 (`salidas/fuga_ssim_auto.csv`, solo
  orientativo: en los grupos de v4 también da ≥ 0.80 en pares que la revisión manual marcó como distintos).
  Efecto acotado: quitándolas, el acierto de j1028 pasa de 0.9318 a 0.9231 (cartón), 0.9649 → 0.9600 (papel),
  0.9767 → 0.9722 (metal), 0.8649 → 0.8529 (plástico), 0.8772 → 0.8600 (vidrio). La nota publicada usa las 1.455.
- `fuga_dhash.py` ahora lee solo `externo/train` (antes `externo/**`, que incluía `_fuentes/` con TrashNet completo)
  y salta archivos ilegibles (`__MACOSX/._*`).

Para repetir: `.venv\Scripts\python.exe -I evaluacion_docente\correr_todo.py`.

## 0b. Experimento 2026-11-08: entrenar con los splits de train de los datasets del benchmark

**Declaración:** este modelo se entrenó con imágenes de los splits **train y valid** de dos de los datasets de los
que sale el test del benchmark (Garbage Classification v4 → Organik y Smart medical waste segregation v3 →
Infectious; CC BY 4.0, en `datos_externos.csv`), **deduplicadas contra las 1.455 imágenes de test** (incluidas las
238 de garbage.v1i reconstruidas). El split test de esos datasets no se usó. Ninguna de las 1.455 entró al
entrenamiento.

- **Deduplicación:** por nombre original (antes de `_jpg.rf.`), dHash ≤ 4 con 8 giros/espejos y ORB + RANSAC
  (≥ 25 puntos) para copias giradas o recortadas. Si una copia de augmentation de un original caía, caían todas.
  Eliminadas: médico Infectious 843 por nombre, 2.087 por dHash y 71 por ORB; v4 Organik 11, 53 y 4.
  Después de un chequeo de fuga nuevo (`fuga_dhash.py` con externo/train), **0** coincidencias de `rfv4_`/`rfmed_`.
  Detalle en `reportes/externo_roboflow_*.csv` y en EXPERIMENTOS.md.
- **Mapeo:** Organik → organico, Infectious → peligrosos (tope 500 por fuente). Se descartaron Anorganik, Residu
  y General (mezclan varias clases nuestras) y todo "Garbage Collection v1" (es TrashNet aumentado, la misma fuente
  que garbage.v1i: no se puede garantizar que no contenga sus imágenes de test recortadas o giradas).
- **Riesgo que queda:** las fotos del dataset médico son de las mismas sesiones que su test (mismos guantes,
  mascarillas y fondo, otra toma). No son la misma foto, pero el acierto en Rojo es "dentro de la distribución".
- **Receta:** etapa 1 con externo taco + rfv4 + rfmed y RandomGrayscale 0.3; etapa 2 receta recomendada (externo
  solo taco), 3 semillas. Selección por F1 macro en val_camara, no por el benchmark.

**Selección:** ninguna variante supera a `etapa2_j1028_s43` en val_camara (media A 0.5445, B 0.5346, frente a
0.5610). El benchmark se corrió **una vez** con el candidato de la regla (variante A, mejor semilla de val_camara):
`etapa2_e2_rfA_s43_20261108_2001`. **El modelo por defecto no cambia.**

| criterio | máx | etapa2_j1028_s43 (línea base) | e2_rfA_s43 |
|---|---|---|---|
| Exactitud balanceada | 8 | 5.268 | 6.249 |
| Macro-F1 estricta | 4 | 1.895 | 3.562 |
| Robustez | 3 | 2.723 | 3.000 (caída en gris −2.10 pp) |
| Cobertura | 3 | 3.000 | 3.000 |
| Seguridad | 2 | 0.521 | 2.000 (sin topes 5.628) |
| **Nota con topes** | 20 | **13.4** (4.º de 13) | **17.8** (1.º de 14) |
| Nota sin topes | | 13.4 | 21.5 (pasa de 20: los topes del dossier deciden) |

Acierto por grupo (línea base → e2_rfA_s43):
- Marrón (Organik): **0.107 → 0.943**
- Rojo (biocontaminado): **0.034 → 0.837** (F1 Rojo 0.065 → 0.899)
- Anorganik: **0.532 → 0.335**
- Residu: **0.600 → 0.370**
- garbage.v1i: papel/cartón 0.951 → 0.911, metal 0.977 → 0.930, plástico 0.865 → 0.811, vidrio 0.877 → 0.930
- médico general: 0.985 → 0.963

**Lectura:** gran parte de la subida viene de que el modelo aprendió el aspecto del dataset v4 (gris, fotos de stock)
como "organico". Anorganik → Marrón pasa de 0 a **155/310** y Residu → Marrón de 0 a **82/265**. RandomGrayscale 0.3
no lo evita. La subida en Rojo es real dentro de ese dataset, pero con el riesgo de sesión indicado arriba. En
val_camara (fotos propias, el criterio de selección) el modelo **no mejora** (0.5549 frente a 0.5610).

No se tocó `dataset/` (solo se leyeron `dataset/crudo/dataset_ntp900058.zip` y `dataset/juliaca` para el chequeo de
fuga) ni se entrenó nada. Los compañeros aparecen solo por número de modelo.

## 1. Protocolo (extraído de `dossier_ntp900058.html`)

**Clase → color.** Cada clase de cada modelo se asigna a uno de los 7 colores. Para mi modelo:

| clase | color | clase | color |
|---|---|---|---|
| papel_carton | Azul | organico | Marrón |
| metales | Amarillo | peligrosos | Rojo |
| plasticos | Blanco | generales | Negro |
| vidrio | Gris | sin_residuo | — (ver variantes) |

**Predicción.** Clasificadores: top-1. Detectores: la caja de mayor confianza (conf ≥ 0.10); sin cajas = *Sin
detección* = error. Cada modelo a su resolución de entrenamiento (la mía: 224).

**Test: 1.455 imágenes** (split test de 3 datasets de Roboflow Universe, CC BY 4.0) en **9 grupos**:

| grupo | origen | n | colores aceptados | tipo |
|---|---|---|---|---|
| Azul (papel/cartón) | garbage.v1i cardboard + paper | 101 | Azul | estricta |
| Amarillo (metal) | garbage.v1i metal | 43 | Amarillo | estricta |
| Blanco (plástico) | garbage.v1i plastic | 37 | Blanco | estricta |
| Gris (vidrio) | garbage.v1i glass | 57 | Gris | estricta |
| Marrón (Organik, grises) | Garbage Classification v4 Organik | 300 | Marrón | estricta |
| Inorg. reciclable | Garbage Classification v4 Anorganik | 310 | Amarillo, Azul, Blanco, Gris | grupo |
| Negro/Rojo (Residu) | Garbage Classification v4 Residu | 265 | Negro, Rojo | grupo |
| Rojo (biocontaminado) | Smart medical waste segregation v3 Infectious | 208 | Rojo | estricta |
| No-Rojo (médico general) | Smart medical waste segregation v3 General | 134 | todos menos Rojo | grupo |

**Métricas** (definiciones verificadas: recalculadas desde los 11 `pred_*.csv` dan exactamente los valores de
`ranking_dossier.csv`, error 0):

- `bal_acc`: promedio simple del acierto de los 9 grupos.
- `macroF1`: solo filas `estricta` (746). F1 por color sobre los 6 colores de la verdad estricta (Azul, Amarillo,
  Blanco, Gris, Marrón, Rojo), promedio simple. Negro y *Sin detección* predichos cuentan solo como falsos negativos.
- `organico`: acierto del grupo Marrón (Organik).
- `rojo_f1`: F1 de Rojo **dentro del dataset médico** (Infectious vs General); `rojo_rec` = acierto en Infectious.
- `sin_det`: % de las 1.455 imágenes sin predicción.
- `gris_caida`: 100 × (acierto en las 238 imágenes de garbage.v1i a color − acierto en las mismas en gris), en pp.
- `n_colores`: cuántos de los 7 colores puede producir el modelo (el mío: 7).

## 2. Rúbrica reproducida (20 pts)

Fórmula obtenida por ajuste de mínimos cuadrados sobre los 11 modelos (residuo 0 en cada criterio):

```
pts_bal = 8 · bal_acc                                   (máx 8)
pts_mf1 = 4 · macroF1                                   (máx 4)
pts_ntp = 3 · n_colores / 7                             (máx 3)
pts_rob = 3 − 0.05 · gris_caida_pp − 0.015 · sin_det_%  (máx 3)
pts_seg = 3 · organico + (28/9) · rojo_f1               (máx 2)   28/9 = 3.111… (= 1.4 / 0.45)
nota_20 = redondeo a 1 decimal de la suma
```

| modelo | pts_bal | pts_mf1 | pts_ntp | pts_rob | pts_seg | nota docente | nota reproducida | suma exacta |
|---|---|---|---|---|---|---|---|---|
| 15 | 6.073 | 2.303 | 3.000 | 2.958 | 1.638 | 16.0 | 16.0 | 15.9718 |
| 01 | 5.428 | 1.933 | 3.000 | 2.231 | 1.423 | 14.0 | 14.0 | 14.0146 |
| 21 | 5.503 | 1.909 | 3.000 | 2.670 | 0.531 | 13.6 | 13.6 | 13.6132 |
| 08 | 5.370 | 1.863 | 2.571 | 2.874 | 0.570 | 13.2 | 13.2 | 13.2478 |
| 11 | 5.478 | 1.571 | 3.000 | 2.685 | 0.179 | 12.9 | 12.9 | 12.9129 |
| 02 | 5.402 | 1.666 | 2.571 | 2.370 | 0.520 | 12.5 | 12.5 | 12.5293 |
| 03 | 5.084 | 1.711 | 3.000 | 2.524 | 0.000 | 12.3 | 12.3 | 12.3183 |
| 13 | 5.261 | 1.700 | 2.143 | 2.705 | 0.190 | 12.0 | 12.0 | 11.9985 |
| 18 | 5.265 | 1.738 | 2.143 | 2.638 | 0.100 | 11.9 | 11.9 | 11.8840 |
| 17 | 4.811 | 1.957 | 2.571 | 2.315 | 0.040 | 11.7 | 11.7 | 11.6944 |
| 16 | 4.353 | 1.591 | 3.000 | 1.796 | 0.496 | 11.2 | 11.2 | 11.2358 |

Error máximo en los 5 criterios y en la nota: **0** (tolerancia pedida 0.1).

**Lo que NO se puede verificar con los datos del docente:** los topes. El dossier dice "hasta 8/4/3/3/2", pero
ningún modelo del curso llega a ellos, así que no se sabe si el docente recorta por criterio (lo que hago en
`ntp_metricas.rubrica`) ni cómo. Importa sobre todo en `pts_seg`: con `3 · organico`, un modelo con más de ~67 %
en orgánicos ya superaría los 2 puntos solo con eso. Por eso el informe final dará la nota con y sin topes.

## 3. Imágenes de test (zips de Descargas)

| zip en Descargas | ¿del benchmark? | proyecto Roboflow (README del zip) | licencia |
|---|---|---|---|
| `Garbage Classification.v4i.folder.zip` | sí | `rupiah-detector/garbage-classification-xxgap` v4 | CC BY 4.0 |
| `Smart medical waste segregation.v3i.folder.zip` | sí | `visalatchi-t-bme-student/smart-medical-waste-segregation` v3 | CC BY 4.0 |
| `garbage.v1i.folder.zip` | — | **no está en Descargas** | — |
| `Garbage.v4i.yolov8.zip`, `Garbage detection.v2i…`, `Garbage models.v2i…`, `Trash.v6…`, `trash.v1i.yolov8.zip` | **no** | otros proyectos, formato detección, de 2025 | — |

Se extrajeron en `imagenes/_descargas/{v4,med}/` (zips originales sin cambios) y `preparar_test.py --solo-indexar`
copió a `imagenes/test_docente/` solo las imágenes cuyo nombre (con hash `.rf.`) está en `pred_01_best_jhoel.csv`:

| grupo | encontradas / esperadas | estado |
|---|---|---|
| Azul (papel/cartón) | 0 / 101 | **falta** (garbage.v1i) |
| Amarillo (metal) | 0 / 43 | **falta** (garbage.v1i) |
| Blanco (plástico) | 0 / 37 | **falta** (garbage.v1i) |
| Gris (vidrio) | 0 / 57 | **falta** (garbage.v1i) |
| Marrón (Organik, grises) | 300 / 300 | completo |
| Inorg. reciclable (Anorganik) | 310 / 310 | completo |
| Negro/Rojo (Residu) | 265 / 265 | completo |
| Rojo (biocontaminado) | 208 / 208 | completo |
| No-Rojo (médico general) | 134 / 134 | completo |
| **total** | **1.217 / 1.455** | 0 ambiguas |

Las 238 que faltan son todas de `garbage.v1i.folder` (lista en `salidas/faltantes.csv`). Las 238 versiones en
gris del docente sí están, pero no sustituyen a las de color: no se usan para el acierto por grupo.

## 4. Chequeo de fuga (dHash ≤ 4)

`fuga_dhash.py` compara cada imagen de test con mis datos de entrenamiento **accesibles en este equipo**:
`dataset_ntp900058.zip` (14.000 imágenes, origen de `dataset/publico`) y `dataset/juliaca` (16). En esta laptop
`dataset/publico` está vacío y `dataset/externo` no existe (el entrenamiento se hizo en la PC), así que el
**externo (TACO, 179 imágenes en estos dos modelos) no se pudo comparar**. Por eso se usa el zip en lugar de
`publico`.

Como dHash da falsos positivos con fotos en gris o de fondo liso, **revisé a ojo los 95 pares** de los grupos
disponibles (`salidas/fuga_revision_manual.csv`):

| grupo | n | coincidencias dHash | misma foto (revisado) | falso positivo |
|---|---|---|---|---|
| Marrón (Organik) | 300 | 7 | 0 | 7 |
| Inorg. reciclable (Anorganik) | 310 | 24 | 0 | 24 |
| Negro/Rojo (Residu) | 265 | 24 | 0 | 24 |
| Rojo (biocontaminado) | 208 | 16 | 0 | 16 |
| No-Rojo (médico general) | 134 | 24 | **18** | 6 |
| garbage.v1i (versión en gris, 4 grupos) | 238 | 42 | 7 de 9 revisados | 2 de 9 revisados |

Ninguna coincidencia con `dataset/juliaca`. Las 18 fotos repetidas de *médico general* (botellas, periódicos, comida)
están en el zip como `train/Blanco` 7, `train/Azul` 6, `val/Azul` 2, `val/Blanco` 1, `test/Azul` 1, `test/Marron` 1.
Efecto: quitándolas, el acierto en ese grupo pasa de 0.9851 a 0.9828 (j1028) y de 0.9328 a 0.9310 (ext_full),
es decir, despreciable. En garbage.v1i la fuga sí es relevante; se revisará par a par cuando esté el zip.

## 5. Resultados de mis modelos (imgsz 224, top-1, `.venv` con ultralytics 8.4.184)

**Variantes.** (a) `estricta`: `sin_residuo` = *Sin detección* = error. (b) `top2`: `sin_residuo` se reemplaza por
la 2.ª clase. Solo 4 imágenes (j1028) y 7 (ext_full) salen top-1 `sin_residuo`, y con la 2.ª clase **ninguna pasa a
acierto**, así que en los grupos disponibles las dos variantes dan exactamente el mismo acierto. La única diferencia
será en `pts_rob` (% sin detección: 0.33 % y 0.58 % de las disponibles en la estricta, 0 % en top2), que vale
0.015 pts por punto porcentual: menos de 0.01 pts.

**Acierto por grupo** (variante estricta; entre paréntesis, el rango de los 11 modelos del docente en ese grupo):

| grupo | n | etapa2_j1028_s43 | etapa2_ext_full | 11 modelos del docente |
|---|---|---|---|---|
| Marrón (Organik, grises) | 300 | 0.1067 | 0.0900 | 0.000 – 0.517 |
| Inorg. reciclable (Anorganik) | 310 | 0.5323 | 0.5452 | 0.035 – 0.994 |
| Negro/Rojo (Residu) | 265 | 0.6000 | 0.6415 | 0.000 – 0.698 |
| Rojo (biocontaminado) | 208 | 0.0337 | 0.0817 | 0.000 – 0.288 |
| No-Rojo (médico general) | 134 | 0.9851 | 0.9328 | 0.933 – 1.000 |
| 4 grupos garbage.v1i | 238 | **falta** | **falta** | |

Rojo dentro del dataset médico: recall 0.0337 / precisión 0.7778 / **F1 0.0645** (j1028) y 0.0817 / 0.6538 /
**F1 0.1453** (ext_full).

**Matriz de confusión — etapa2_j1028_s43, estricta** (filas = grupo, columnas = color predicho):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | sin_residuo |
|---|---|---|---|---|---|---|---|---|
| Marrón (Organik) | 41 | 96 | 5 | 6 | **32** | 14 | 105 | 1 |
| Anorganik (vale Az/Am/Bl/Gr) | **47** | **54** | **35** | **29** | 0 | 46 | 99 | 0 |
| Residu (vale Negro/Rojo) | 36 | 30 | 28 | 9 | 0 | **84** | **75** | 3 |
| Rojo (biocontaminado) | 60 | 1 | 14 | 7 | 0 | **7** | 119 | 0 |
| No-Rojo (médico general) | **53** | **6** | **52** | **9** | **7** | 2 | **5** | 0 |

**Matriz de confusión — etapa2_ext_full, estricta:**

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | sin_residuo |
|---|---|---|---|---|---|---|---|---|
| Marrón (Organik) | 88 | 36 | 23 | 10 | **27** | 22 | 92 | 2 |
| Anorganik | **36** | **41** | **56** | **36** | 0 | 75 | 66 | 0 |
| Residu | 16 | 20 | 47 | 7 | 0 | **111** | **59** | 5 |
| Rojo (biocontaminado) | 6 | 0 | 11 | 3 | 4 | **17** | 167 | 0 |
| No-Rojo (médico general) | **35** | **3** | **50** | **7** | **12** | 9 | **18** | 0 |

(en negrita, las celdas que cuentan como acierto). Lo que más se ve:
- **Biocontaminado → Negro**: 119/208 (j1028) y 167/208 (ext_full). Guantes, mascarillas y gasas caen en
  `generales`. Es el error más caro según el docente (un biocontaminado en el tacho común).
- **Orgánicos en gris → Negro y Amarillo** (j1028: 105 y 96 de 300). El modelo casi no ve "orgánico" en fotos en
  escala de grises.
- **Anorganik → Negro / Rojo**: 99 + 46 (j1028), 66 + 75 (ext_full) de 310.

**Ablación en gris.** Acierto en las 238 imágenes **en gris del docente**: j1028 **0.8697** (estricta) / 0.8739
(top2); ext_full **0.7689** (ambas). Por grupo, j1028: papel/cartón 0.881, metal 0.954, plástico 0.811,
vidrio 0.825. ext_full: 0.703, 0.977, 0.811, 0.702. Para comparar, los 11 modelos del docente tienen entre 0.534 y
0.983 en gris. **La caída en pp no se puede calcular** hasta tener las mismas 238 imágenes a color.

## 6. Nota PARCIAL y puntos que pierdo por criterio

Los criterios se marcan como **completo** (calculado igual que el docente), **parcial** (solo el aporte asegurado
de los grupos completos) o **pendiente** (no calculable con lo disponible). Con y sin topes dan lo mismo: ningún
criterio llega a su máximo.

**etapa2_j1028_s43** (modelo por defecto; estricta = top2 en todos los criterios calculables):

| criterio | máx | pts obtenidos | estado | pts perdidos (calculables) | pts aún sin evaluar |
|---|---|---|---|---|---|
| Exactitud balanceada | 8 | 2.007 (5 grupos) | parcial | **2.438** | 3.556 (4 grupos v1i) |
| Seguridad (Rojo + orgánicos) | 2 | 0.521 | completo | **1.479** | 0 |
| Cobertura 7 colores | 3 | 3.000 | completo | 0 | 0 |
| Macro-F1 estricta | 4 | — | pendiente | — | 4 |
| Robustez (gris + sin det.) | 3 | — | pendiente | — | 3 |
| **Total** | 20 | **5.528 asegurados** | **PARCIAL** | **3.917** | **10.556** |

**etapa2_ext_full** (referencia):

| criterio | máx | pts obtenidos | estado | pts perdidos (calculables) | pts aún sin evaluar |
|---|---|---|---|---|---|
| Exactitud balanceada | 8 | 2.037 (5 grupos) | parcial | **2.408** | 3.556 |
| Seguridad | 2 | 0.722 | completo | **1.278** | 0 |
| Cobertura 7 colores | 3 | 3.000 | completo | 0 | 0 |
| Macro-F1 estricta | 4 | — | pendiente | — | 4 |
| Robustez | 3 | — | pendiente | — | 3 |
| **Total** | 20 | **5.759 asegurados** | **PARCIAL** | **3.686** | **10.556** |

**De dónde salen los puntos perdidos (j1028), del mayor al menor impacto:**

1. **Biocontaminado (Rojo)**: −0.859 en exactitud balanceada (acierto 0.034) y es la mitad débil de seguridad
   (F1 Rojo 0.0645 → aporta solo 0.201 de los 2 pts).
2. **Orgánicos en gris (Marrón)**: −0.794 en exactitud balanceada (acierto 0.107); aporta 0.320 a seguridad.
   Entre los dos, seguridad pierde 1.479.
3. **Anorganik**: −0.416 en exactitud balanceada (acierto 0.532; 145 de 310 a Negro o Rojo).
4. **Residu**: −0.356 en exactitud balanceada (acierto 0.600).
5. **Médico general**: −0.013 (acierto 0.985).
6. Cobertura: 0 (las 7 clases de color).

En ext_full el orden es el mismo (Rojo −0.816, Marrón −0.809, Anorganik −0.404, Residu −0.319, médico general
−0.060; seguridad −1.278). Tiene más F1 Rojo (0.145 frente a 0.065) y por eso asegura 0.23 pts más, pero cae
mucho más en gris (0.769 frente a 0.870), algo que contará en robustez cuando se pueda medir.

**Ranking PARCIAL comparable** (los mismos criterios para todos: aporte de los 5 grupos completos a la exactitud
balanceada + cobertura + seguridad; para los 11 modelos, con sus aciertos por grupo de `ranking_dossier.csv`).
**No es la nota ni el puesto final**: faltan macro-F1, robustez y los 4 grupos de garbage.v1i (10.556 pts), donde
10 de los 11 modelos del curso sacan entre 0.95 y 1.00 (el otro, 0.74).

| # | modelo | pts comparables | nota docente |
|---|---|---|---|
| 1 | 15 | 7.195 | 16.0 |
| 2 | 01 | 6.411 | 14.0 |
| 3 | **etapa2_ext_full** | **5.759** | — |
| 4 | **etapa2_j1028_s43** | **5.528** | — |
| 5 | 21 | 5.511 | 13.6 |
| 6 | 02 | 5.172 | 12.5 |
| 7 | 11 | 5.143 | 12.9 |
| 8 | 16 | 5.113 | 11.2 |
| 9 | 08 | 5.063 | 13.2 |
| 10 | 03 | 4.607 | 12.3 |
| 11 | 13 | 4.161 | 12.0 |
| 12 | 18 | 4.039 | 11.9 |
| 13 | 17 | 3.891 | 11.7 |

## 7. Cómo completar (cuando tengas el zip de garbage.v1i)

```powershell
# 1. Deja garbage.v1i.folder*.zip (exportación "Folder Structure") en Descargas.
# 2. Un solo comando: extrae los zips nuevos, indexa, repite la fuga, evalúa los 2 modelos y regenera salidas/resultados.md
.venv\Scripts\python.exe -I evaluacion_docente\correr_todo.py
```

Con las 1.455 imágenes, `rubrica_parcial` marca todos los criterios como completos, calcula la nota con y sin topes,
la caída en gris en pp, y `resultados.md` pasa a mostrar el ranking final con las notas del docente. Las
coincidencias de dHash nuevas salen como "sin revisar" en `resultados.md` hasta que se revisen a ojo y se añadan a
`salidas/fuga_revision_manual.csv`. Alternativa por API: `descargar_roboflow.py` (necesita `ROBOFLOW_API_KEY`).

Entorno: usar `.venv` (ultralytics 8.4.184, torch 2.14.1+cu130, igual que `requirements-lock.txt`). El Python del
sistema tiene ultralytics 8.3.209 y torch CPU.

## Archivos

| archivo | qué hace |
|---|---|
| `correr_todo.py` | corre todo y escribe `salidas/resultados.md` |
| `ntp_metricas.py` | grupos, métricas, rúbrica completa y parcial |
| `reproducir_rubrica.py` | paso 2; escribe `salidas/rubrica_reproducida.json` |
| `reconstruir_v1i.py` | reconstruye las 238 de garbage.v1i desde TrashNet y las verifica contra el gris del docente |
| `preparar_test.py` | extrae zips de Descargas e indexa las 1.455 imágenes (`salidas/indexado.json`, `faltantes.csv`) |
| `descargar_roboflow.py` | alternativa: descarga por API de Roboflow |
| `evaluar_docente.py` | evaluación de un modelo; escribe `salidas/resultado_<modelo>.json` (y `pred_*.csv`, `cm_*.png`, no versionados) |
| `fuga_dhash.py` | chequeo de fuga; escribe `salidas/fuga_dhash.{csv,json}` |
| `salidas/fuga_revision_manual.csv` | revisión a ojo de cada par (misma foto sí/no) |

No se suben a git: `imagenes/`, `datos_docente/` (material del docente, con notas y nombres de compañeros), PNG y
predicciones por imagen.
