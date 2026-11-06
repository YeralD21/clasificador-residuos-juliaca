# Resultados PARCIALES (generado por correr_todo.py)

Imágenes de test disponibles: **1217/1455**.

## Acierto por grupo (variante estricta)

| grupo | n disp./esperado | estado | etapa2_j1028_s43 | etapa2_ext_full | gris etapa2_j1028_s43 | gris etapa2_ext_full |
|---|---|---|---|---|---|---|
| Azul (papel/cartón) | 0/101 | falta | — | — | 0.8812 | 0.7030 |
| Amarillo (metal) | 0/43 | falta | — | — | 0.9535 | 0.9767 |
| Blanco (plástico) | 0/37 | falta | — | — | 0.8108 | 0.8108 |
| Gris (vidrio) | 0/57 | falta | — | — | 0.8246 | 0.7018 |
| Marrón (Organik, grises) | 300/300 | completo | 0.1067 | 0.0900 | — | — |
| Inorg. reciclable (Anorganik) | 310/310 | completo | 0.5323 | 0.5452 | — | — |
| Negro/Rojo (Residu) | 265/265 | completo | 0.6000 | 0.6415 | — | — |
| Rojo (biocontaminado) | 208/208 | completo | 0.0337 | 0.0817 | — | — |
| No-Rojo (médico general) | 134/134 | completo | 0.9851 | 0.9328 | — | — |

## etapa2_j1028_s43: rúbrica

| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |
|---|---|---|---|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 2.007 | 2.007 | 2.007 | parcial | 3.556 | 2.438 |
| Seguridad (Rojo + orgánicos) | 2 | 0.521 | 0.521 | 0.521 | completo | 0.000 | 1.479 |
| Cobertura 7 colores NTP | 3 | 3.000 | 3.000 | 3.000 | completo | 0.000 | 0.000 |
| Macro-F1 estricta | 4 | — | — | — | pendiente | 4.000 | — |
| Robustez (gris + sin detección) | 3 | — | — | — | pendiente | 3.000 | — |

**estricta: PARCIAL** — asegurados 5.528; quedan 10.556 pts sin evaluar (no se estiman).

**top2: PARCIAL** — asegurados 5.528; quedan 10.556 pts sin evaluar (no se estiman).

Orgánicos 0.1067 · Rojo recall 0.0337 · precisión 0.7778 · F1 0.0645 · sin detección (sin_residuo, disponibles) 0.33 % · gris v1i 0.8697 · color v1i — · caída — pp

Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | Sin det. |
|---|---|---|---|---|---|---|---|---|
| Marrón (Organik, grises) | 41 | 96 | 5 | 6 | 32 | 14 | 105 | 1 |
| Inorg. reciclable (Anorganik) | 47 | 54 | 35 | 29 | 0 | 46 | 99 | 0 |
| Negro/Rojo (Residu) | 36 | 30 | 28 | 9 | 0 | 84 | 75 | 3 |
| Rojo (biocontaminado) | 60 | 1 | 14 | 7 | 0 | 7 | 119 | 0 |
| No-Rojo (médico general) | 53 | 6 | 52 | 9 | 7 | 2 | 5 | 0 |

## etapa2_ext_full: rúbrica

| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |
|---|---|---|---|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 2.037 | 2.037 | 2.037 | parcial | 3.556 | 2.408 |
| Seguridad (Rojo + orgánicos) | 2 | 0.722 | 0.722 | 0.722 | completo | 0.000 | 1.278 |
| Cobertura 7 colores NTP | 3 | 3.000 | 3.000 | 3.000 | completo | 0.000 | 0.000 |
| Macro-F1 estricta | 4 | — | — | — | pendiente | 4.000 | — |
| Robustez (gris + sin detección) | 3 | — | — | — | pendiente | 3.000 | — |

**estricta: PARCIAL** — asegurados 5.759; quedan 10.556 pts sin evaluar (no se estiman).

**top2: PARCIAL** — asegurados 5.759; quedan 10.556 pts sin evaluar (no se estiman).

Orgánicos 0.0900 · Rojo recall 0.0817 · precisión 0.6538 · F1 0.1453 · sin detección (sin_residuo, disponibles) 0.58 % · gris v1i 0.7689 · color v1i — · caída — pp

Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | Sin det. |
|---|---|---|---|---|---|---|---|---|
| Marrón (Organik, grises) | 88 | 36 | 23 | 10 | 27 | 22 | 92 | 2 |
| Inorg. reciclable (Anorganik) | 36 | 41 | 56 | 36 | 0 | 75 | 66 | 0 |
| Negro/Rojo (Residu) | 16 | 20 | 47 | 7 | 0 | 111 | 59 | 5 |
| Rojo (biocontaminado) | 6 | 0 | 11 | 3 | 4 | 17 | 167 | 0 |
| No-Rojo (médico general) | 35 | 3 | 50 | 7 | 12 | 9 | 18 | 0 |

## Ranking PARCIAL comparable

Mismos criterios calculables para todos: aporte de los grupos completos a `pts_bal` + `pts_ntp` + `pts_seg` (para los 11 modelos se calcula con sus aciertos por grupo de `ranking_dossier.csv`). No es la nota: faltan los criterios pendientes.

| # | modelo | puntos |
|---|---|---|
| 1 | 15 (nota 16.0) | 7.195 |
| 2 | 01 (nota 14.0) | 6.411 |
| 3 | **etapa2_ext_full** | 5.759 |
| 4 | **etapa2_j1028_s43** | 5.528 |
| 5 | 21 (nota 13.6) | 5.511 |
| 6 | 02 (nota 12.5) | 5.172 |
| 7 | 11 (nota 12.9) | 5.143 |
| 8 | 16 (nota 11.2) | 5.113 |
| 9 | 08 (nota 13.2) | 5.063 |
| 10 | 03 (nota 12.3) | 4.607 |
| 11 | 13 (nota 12.0) | 4.161 |
| 12 | 18 (nota 11.9) | 4.039 |
| 13 | 17 (nota 11.7) | 3.891 |

## Chequeo de fuga (dHash ≤ 4)

Imágenes de entrenamiento comparadas: zip 14000, publico 0, externo 0, juliaca 16

| grupo | n | coincidencias dHash | revisadas: misma foto | revisadas: falso positivo | sin revisar |
|---|---|---|---|---|---|
| Inorg. reciclable (Anorganik) | 310 | 24 | 0 | 24 | 0 |
| Marrón (Organik, grises) | 300 | 7 | 0 | 7 | 0 |
| Negro/Rojo (Residu) | 265 | 24 | 0 | 24 | 0 |
| No-Rojo (médico general) | 134 | 24 | 18 | 6 | 0 |
| Rojo (biocontaminado) | 208 | 16 | 0 | 16 | 0 |
| [gris] Amarillo (metal) | 43 | 8 | 0 | 0 | 8 |
| [gris] Azul (papel/cartón) | 101 | 20 | 0 | 0 | 20 |
| [gris] Blanco (plástico) | 37 | 5 | 0 | 0 | 5 |
| [gris] Gris (vidrio) | 57 | 9 | 0 | 0 | 9 |
