# Resultados COMPLETOS (generado por correr_todo.py)

Imágenes de test disponibles: **1455/1455**.

## Acierto por grupo (variante estricta)

| grupo | n disp./esperado | estado | etapa2_j1028_s43 | etapa2_ext_full | etapa2_e2_nf_s43_20261110_2132 | gris etapa2_j1028_s43 | gris etapa2_ext_full | gris etapa2_e2_nf_s43_20261110_2132 |
|---|---|---|---|---|---|---|---|---|
| Azul (papel/cartón) | 101/101 | completo | 0.9505 | 0.8218 | 0.9505 | 0.8812 | 0.7030 | 0.9604 |
| Amarillo (metal) | 43/43 | completo | 0.9767 | 0.9535 | 0.9767 | 0.9535 | 0.9767 | 0.9535 |
| Blanco (plástico) | 37/37 | completo | 0.8649 | 0.7838 | 0.8108 | 0.8108 | 0.8108 | 0.8649 |
| Gris (vidrio) | 57/57 | completo | 0.8772 | 0.8246 | 0.8070 | 0.8246 | 0.7018 | 0.8070 |
| Marrón (Organik, grises) | 300/300 | completo | 0.1067 | 0.0900 | 0.5000 | — | — | — |
| Inorg. reciclable (Anorganik) | 310/310 | completo | 0.5323 | 0.5452 | 0.5161 | — | — | — |
| Negro/Rojo (Residu) | 265/265 | completo | 0.6000 | 0.6415 | 0.5396 | — | — | — |
| Rojo (biocontaminado) | 208/208 | completo | 0.0337 | 0.0817 | 0.0962 | — | — | — |
| No-Rojo (médico general) | 134/134 | completo | 0.9851 | 0.9328 | 0.9701 | — | — | — |

## etapa2_j1028_s43: rúbrica

| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |
|---|---|---|---|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 5.268 | 5.268 | 5.268 | completo | 0.000 | 2.732 |
| Macro-F1 estricta | 4 | 1.895 | 1.895 | 1.895 | completo | 0.000 | 2.105 |
| Seguridad (Rojo + orgánicos) | 2 | 0.521 | 0.521 | 0.521 | completo | 0.000 | 1.479 |
| Robustez (gris + sin detección) | 3 | 2.723 | 2.723 | 2.748 | completo | 0.000 | 0.277 |
| Cobertura 7 colores NTP | 3 | 3.000 | 3.000 | 3.000 | completo | 0.000 | 0.000 |

**estricta: nota 13.4 / 20**

**top2: nota 13.4 / 20**

Orgánicos 0.1067 · Rojo recall 0.0337 · precisión 0.7778 · F1 0.0645 · sin detección (sin_residuo, disponibles) 0.27 % · gris v1i 0.8697 · color v1i 0.9244 · caída 5.46 pp

Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | Sin det. |
|---|---|---|---|---|---|---|---|---|
| Azul (papel/cartón) | 96 | 1 | 1 | 1 | 0 | 1 | 1 | 0 |
| Amarillo (metal) | 0 | 42 | 0 | 0 | 0 | 0 | 1 | 0 |
| Blanco (plástico) | 2 | 0 | 32 | 3 | 0 | 0 | 0 | 0 |
| Gris (vidrio) | 0 | 2 | 4 | 50 | 0 | 1 | 0 | 0 |
| Marrón (Organik, grises) | 41 | 96 | 5 | 6 | 32 | 14 | 105 | 1 |
| Inorg. reciclable (Anorganik) | 47 | 54 | 35 | 29 | 0 | 46 | 99 | 0 |
| Negro/Rojo (Residu) | 36 | 30 | 28 | 9 | 0 | 84 | 75 | 3 |
| Rojo (biocontaminado) | 60 | 1 | 14 | 7 | 0 | 7 | 119 | 0 |
| No-Rojo (médico general) | 53 | 6 | 52 | 9 | 7 | 2 | 5 | 0 |

## etapa2_ext_full: rúbrica

| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |
|---|---|---|---|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 5.044 | 5.044 | 5.053 | completo | 0.000 | 2.956 |
| Macro-F1 estricta | 4 | 1.915 | 1.915 | 1.917 | completo | 0.000 | 2.085 |
| Seguridad (Rojo + orgánicos) | 2 | 0.722 | 0.722 | 0.722 | completo | 0.000 | 1.278 |
| Robustez (gris + sin detección) | 3 | 2.635 | 2.635 | 2.622 | completo | 0.000 | 0.365 |
| Cobertura 7 colores NTP | 3 | 3.000 | 3.000 | 3.000 | completo | 0.000 | 0.000 |

**estricta: nota 13.3 / 20**

**top2: nota 13.3 / 20**

Orgánicos 0.0900 · Rojo recall 0.0817 · precisión 0.6538 · F1 0.1453 · sin detección (sin_residuo, disponibles) 0.55 % · gris v1i 0.7689 · color v1i 0.8403 · caída 7.14 pp

Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | Sin det. |
|---|---|---|---|---|---|---|---|---|
| Azul (papel/cartón) | 83 | 0 | 2 | 1 | 0 | 1 | 13 | 1 |
| Amarillo (metal) | 0 | 41 | 0 | 0 | 0 | 0 | 2 | 0 |
| Blanco (plástico) | 0 | 2 | 29 | 2 | 0 | 1 | 3 | 0 |
| Gris (vidrio) | 0 | 6 | 3 | 47 | 0 | 1 | 0 | 0 |
| Marrón (Organik, grises) | 88 | 36 | 23 | 10 | 27 | 22 | 92 | 2 |
| Inorg. reciclable (Anorganik) | 36 | 41 | 56 | 36 | 0 | 75 | 66 | 0 |
| Negro/Rojo (Residu) | 16 | 20 | 47 | 7 | 0 | 111 | 59 | 5 |
| Rojo (biocontaminado) | 6 | 0 | 11 | 3 | 4 | 17 | 167 | 0 |
| No-Rojo (médico general) | 35 | 3 | 50 | 7 | 12 | 9 | 18 | 0 |

## etapa2_e2_nf_s43_20261110_2132: rúbrica

| criterio | máx | estricta (con topes) | estricta (sin topes) | top2 (con topes) | estado | pts en juego | pts perdidos (calculables) |
|---|---|---|---|---|---|---|---|
| Exactitud balanceada (9 grupos) | 8 | 5.482 | 5.482 | 5.499 | completo | 0.000 | 2.518 |
| Macro-F1 estricta | 4 | 2.521 | 2.521 | 2.521 | completo | 0.000 | 1.479 |
| Cobertura 7 colores NTP | 3 | 3.000 | 3.000 | 3.000 | completo | 0.000 | 0.000 |
| Robustez (gris + sin detección) | 3 | 3.000 | 3.036 | 3.000 | completo | 0.000 | 0.000 |
| Seguridad (Rojo + orgánicos) | 2 | 2.000 | 2.036 | 2.000 | completo | 0.000 | 0.000 |

**estricta: nota 16.0 / 20**

**top2: nota 16.0 / 20**

Orgánicos 0.5000 · Rojo recall 0.0962 · precisión 0.8333 · F1 0.1724 · sin detección (sin_residuo, disponibles) 0.41 % · gris v1i 0.9076 · color v1i 0.8992 · caída -0.84 pp

Matriz de confusión (estricta; filas = grupo, columnas = color predicho; Sin det. = sin_residuo):

| grupo | Azul | Amarillo | Blanco | Gris | Marrón | Rojo | Negro | Sin det. |
|---|---|---|---|---|---|---|---|---|
| Azul (papel/cartón) | 96 | 0 | 3 | 1 | 0 | 0 | 1 | 0 |
| Amarillo (metal) | 0 | 42 | 0 | 0 | 0 | 0 | 1 | 0 |
| Blanco (plástico) | 2 | 1 | 30 | 3 | 0 | 1 | 0 | 0 |
| Gris (vidrio) | 0 | 4 | 7 | 46 | 0 | 0 | 0 | 0 |
| Marrón (Organik, grises) | 26 | 5 | 10 | 3 | 150 | 5 | 100 | 1 |
| Inorg. reciclable (Anorganik) | 69 | 28 | 32 | 31 | 19 | 13 | 118 | 0 |
| Negro/Rojo (Residu) | 45 | 16 | 42 | 6 | 8 | 66 | 77 | 5 |
| Rojo (biocontaminado) | 58 | 0 | 21 | 1 | 2 | 20 | 106 | 0 |
| No-Rojo (médico general) | 49 | 4 | 49 | 13 | 10 | 4 | 5 | 0 |

## Ranking completo (nota del docente vs mi nota reproducida)

| # | modelo | puntos |
|---|---|---|
| 1 | 15 | 16.0 |
| 2 | **etapa2_e2_nf_s43_20261110_2132** | 16.0 |
| 3 | 01 | 14.0 |
| 4 | 21 | 13.6 |
| 5 | **etapa2_j1028_s43** | 13.4 |
| 6 | **etapa2_ext_full** | 13.3 |
| 7 | 08 | 13.2 |
| 8 | 11 | 12.9 |
| 9 | 02 | 12.5 |
| 10 | 03 | 12.3 |
| 11 | 13 | 12.0 |
| 12 | 18 | 11.9 |
| 13 | 17 | 11.7 |
| 14 | 16 | 11.2 |

## Chequeo de fuga (dHash ≤ 4)

Imágenes de entrenamiento comparadas: zip 0, publico 12220, externo 7026, juliaca 624

| grupo | n | coincidencias dHash | revisadas: misma foto | revisadas: falso positivo | sin revisar |
|---|---|---|---|---|---|
| Amarillo (metal) | 43 | 43 | 0 | 0 | 43 |
| Azul (papel/cartón) | 101 | 25 | 0 | 0 | 25 |
| Blanco (plástico) | 37 | 5 | 0 | 0 | 5 |
| Gris (vidrio) | 57 | 57 | 0 | 0 | 57 |
| Inorg. reciclable (Anorganik) | 310 | 23 | 0 | 22 | 1 |
| Marrón (Organik, grises) | 300 | 5 | 0 | 5 | 0 |
| Negro/Rojo (Residu) | 265 | 22 | 0 | 22 | 0 |
| No-Rojo (médico general) | 134 | 25 | 18 | 6 | 1 |
| Rojo (biocontaminado) | 208 | 13 | 0 | 13 | 0 |
| [gris] Amarillo (metal) | 43 | 43 | 0 | 0 | 43 |
| [gris] Azul (papel/cartón) | 101 | 23 | 0 | 0 | 23 |
| [gris] Blanco (plástico) | 37 | 5 | 0 | 0 | 5 |
| [gris] Gris (vidrio) | 57 | 56 | 0 | 0 | 56 |
