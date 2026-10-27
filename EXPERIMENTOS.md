# Metodología de experimentos (sin leakage)

## Regla de oro
| Conjunto | Para qué se usa | Cuántas veces |
|---|---|---|
| **train** | Ajustar los pesos del modelo | Siempre |
| **val** | Comparar experimentos y elegir el mejor (y early stopping) | Todas las que haga falta |
| **test** (`dataset/juliaca/test`) | Medir el modelo FINAL ya elegido | **Una sola vez** |

Si se elige el modelo mirando el test, el test deja de ser una medida honesta (leakage por selección).
Cada uso de `juliaca/test` queda anotado en `reportes/registro_test.csv`.

## Métrica de selección
**F1 macro en validación**: promedia el F1 de cada clase por igual, así una clase con pocas fotos
(peligrosos, sin_residuo) pesa lo mismo que una con muchas. En empate se mira top-1.

## Diseño: un factor a la vez
Cada experimento cambia **una sola cosa** respecto a la referencia, para poder explicar su efecto.

**Etapa 1** (datos públicos; val = `publico/val`):
| Experimento | Cambio | Hipótesis |
|---|---|---|
| e1_base | yolo26s-cls, 224 px | Referencia |
| e1_drop02 | dropout 0.2 | Reduce el sobreajuste (loss_train ≪ loss_val) |
| e1_s320 | 320 px | Más detalle (textura del material) |
| e1_m224 | yolo26m-cls | Más capacidad |

**Etapa 2** (fine-tuning con Juliaca; val = `juliaca/val`), partiendo del ganador de la etapa 1:
| Experimento | Cambio | Hipótesis |
|---|---|---|
| e2_base | lr 1e-4, freeze 6, replay 1 | Referencia |
| e2_lr3e-5 / e2_lr3e-4 | learning rate | Muy bajo: aprende poco; muy alto: olvida la etapa 1 |
| e2_freeze0 / e2_freeze9 | capas congeladas | Con pocas fotos, congelar más evita memorizar |
| e2_replay0 / e2_replay2 | repaso de datos públicos | Sin repaso puede olvidar; con mucho, se adapta menos a Juliaca |
| e2_solo_jul | sin etapa 1 (desde ImageNet) | Mide cuánto aporta la etapa 1 |

## Comandos
```bash
python scripts/experimentar.py --listar                      # ver experimentos y resultados
python scripts/experimentar.py --etapa 1                     # corre todos los de la etapa 1
python scripts/experimentar.py --etapa 2 --pesos modelos/experimentos/<ganador_etapa1>.pt
python scripts/evaluar.py --pesos modelos/a.pt modelos/b.pt --datos val_camara   # set aparte de la cámara (por foto)
python scripts/evaluar.py --pesos <ganador_etapa1>.pt <ganador_etapa2>.pt --split test   # UNA vez, al final
```
Resultados: `reportes/experimentos.csv` (tabla para el informe) y `reportes/experimentos/*.json`.
Curvas de cada entrenamiento: `runs/etapaN_<experimento>/results.png`.

## Qué reportar en el informe
1. Tabla de experimentos con F1 macro y top-1 en val, y el ganador de cada etapa.
2. Resultado en `juliaca/test` del ganador de la etapa 1 y del de la etapa 2 (mejora por datos locales).
3. Matriz de confusión y F1 por clase del modelo final: qué residuos confunde y por qué.
4. Limitaciones: la partición pública venía aleatoria por foto (se quitaron casi-duplicados por hash
   perceptual, pero puede quedar correlación); por eso la métrica válida es `juliaca/test`, separado por objeto.
5. Que el data augmentation (giros, color, RandAugment) transforma fotos reales al entrenar; no se generaron datos.
6. Datos externos, si se usan (solo en train, ver README §7.1 y `datos_externos.csv`), con su atribución:
   TACO, Proença & Simões (2020), arXiv:2003.06975 (anotaciones CC BY 4.0; imágenes CC BY 4.0 u ODbL);
   imágenes ODbL © OpenLitterMap & Contributors. Informe anti-leakage: `reportes/externo_dedup.csv`.

## Decisión etapa 1
Resultados reales en `publico/val` (copiados de `reportes/experimentos.csv`):

| Experimento | F1 macro val | top-1 val | Modelo base | imgsz | dropout | Épocas | Mejor época | Minutos |
|---|---|---|---|---|---|---|---|---|
| e1_m224 | 0.9139 | 0.9126 | yolo26m-cls.pt | 224 | 0.0 | 60 | 58 | 16.9 |
| e1_base | 0.9129 | 0.9116 | yolo26s-cls.pt | 224 | 0.0 | 60 | 56 | 9.5 |
| e1_s320 | 0.9113 | 0.911 | yolo26s-cls.pt | 320 | 0.0 | 60 | 47 | 15.8 |
| e1_drop02 | 0.9086 | 0.9073 | yolo26s-cls.pt | 224 | 0.2 | 60 | 54 | 9.6 |

**Ganador: e1_base.** Criterio: empate técnico con e1_m224 (diferencia de F1 macro 0.001 < 0.005);
se prefiere el modelo más simple y rápido (yolo26s, 9.5 min frente a 16.9 min de yolo26m).
Copiado a `modelos/etapa1_ganador.pt` (+ `.json`) desde `modelos/experimentos/etapa1_e1_base_20261024_1453.pt`.

## Avance etapa 2 (provisional)
Resultados reales en `juliaca/val` (copiados de `reportes/experimentos.csv`; todos parten de
`etapa1_ganador.pt` salvo e2_solo_jul):

| Experimento | F1 macro val | top-1 val | Modelo base | lr0 | freeze | replay | Épocas | Mejor época* | Minutos |
|---|---|---|---|---|---|---|---|---|---|
| e2_replay2 | 0.8139 | 0.75 | etapa1_ganador.pt | 0.0001 | 6 | 2.0 | 17 | 7 | 0.9 |
| e2_solo_jul | 0.7827 | 0.8929 | yolo26s-cls.pt | 0.001 | 0 | 0.0 | 15 | 5 | 0.8 |
| e2_lr3e-4 | 0.7479 | 0.7143 | etapa1_ganador.pt | 0.0003 | 6 | 1.0 | 15 | 5 | 0.8 |
| e2_freeze9 | 0.7228 | 0.6786 | etapa1_ganador.pt | 0.0001 | 9 | 1.0 | 20 | 10 | 0.8 |
| e2_base | 0.6982 | 0.6429 | etapa1_ganador.pt | 0.0001 | 6 | 1.0 | 20 | 7 | 0.8 |
| e2_freeze0 | 0.6846 | 0.6429 | etapa1_ganador.pt | 0.0001 | 0 | 1.0 | 22 | 11 | 1.2 |
| e2_replay0 | 0.6262 | 0.6071 | etapa1_ganador.pt | 0.0001 | 6 | 0.0 | 12 | 2 | 0.7 |
| e2_lr3e-5 | 0.6016 | 0.5714 | etapa1_ganador.pt | 3e-05 | 6 | 1.0 | 21 | 2 | 1.1 |

\* "Mejor época" = primera época con el máximo top-1 en `results.csv`. Ultralytics guarda `best.pt`
según (top1+top5)/2, así que en e2_base (10), e2_lr3e-5 (11) y e2_freeze0 (12) la época de `best.pt`
es otra (ver log en `reportes/evidencias/etapa2/`). F1 y top-1 de la tabla sí son de `best.pt`.

Modelo provisional: e2_replay2 (máximo F1 macro en juliaca/val = 0.8139). Val tiene 28
fotos/9 objetos; e2_solo_jul tiene mayor top-1 (0.8929) pero menor F1 por 1 foto de
organico. juliaca/test NO se ha usado: se evaluará una sola vez cuando el dataset esté
completo.

Copiado a `modelos/etapa2_avance.pt` (+ `.json`) desde `modelos/experimentos/etapa2_e2_replay2_20261024_2101.pt`.

## Datos externos TACO (`--externo`, sin ganador elegido)
`dataset/externo/train`: generales 139 (Crisp packet + Other plastic wrapper revisados), metales 40
(Aluminium foil). Solo train; `auditar.py` sin errores (dHash <= 4 contra publico y juliaca).

Etapa 1 en `publico/val` (de `reportes/experimentos.csv` y `reportes/etapa1_e1_*_publico_val.json`):

| Experimento | F1 macro | top-1 | F1 generales | F1 metales |
|---|---|---|---|---|
| e1_base | 0.9129 | 0.9116 | 0.933 | 0.907 |
| e1_ext  | 0.9153 | 0.9143 | 0.928 | 0.916 |

Diferencia de F1 macro 0.0024 (< 0.005, empate técnico con el criterio de la etapa 1).

Etapa 2 en `juliaca/val` (28 fotos / 9 objetos). Desde e2_replay2, `juliaca/train/generales` pasó de 15
a 31 fotos (obj037, obj038 del 25-10), por eso se re-corrió e2_replay2 como control (`e2_replay2_v2`).
F1 por clase de `reportes/etapa2_<exp>_*_juliaca_val.json` (aciertos/n):

| Experimento | Parte de | juliaca/train | F1 macro | top-1 | generales | metales | organico | peligrosos | plasticos | sin_residuo | vidrio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e2_replay2    | e1_base | 24-10 | 0.8139 | 0.7500 | 0.444 (2/3) | 0.615 (4/9) | 1.000 (1/1) | 1.000 (2/2) | 0.923 (6/6) | 0.857 (3/4) | 0.857 (3/3) |
| e2_replay2_v2 | e1_base | 25-10 | 0.8516 | 0.8214 | 0.667 (3/3) | 0.714 (5/9) | 1.000 (1/1) | 0.800 (2/2) | 0.923 (6/6) | 0.857 (3/4) | 1.000 (3/3) |
| e2_ext        | e1_base | 25-10 | 0.7074 | 0.7857 | 0.600 (3/3) | 0.714 (5/9) | 0.000 (0/1) | 1.000 (2/2) | 0.923 (6/6) | 0.857 (3/4) | 0.857 (3/3) |
| e2_ext_full   | e1_ext  | 25-10 | 0.8186 | 0.7500 | 0.444 (2/3) | 0.571 (4/9) | 1.000 (1/1) | 1.000 (2/2) | 0.857 (6/6) | 0.857 (3/4) | 1.000 (3/3) |

e2_ext pierde F1 macro casi solo por la única foto de organico (predicha generales): con 1 foto esa
clase pasa de F1 1.0 a 0.0. Comparación de un factor: e2_ext vs e2_replay2_v2 (mismo juliaca, misma base).

Bolsa de panetón (`juliaca/val/generales/obj004_bolsa_paneton_milano`, multicapa), predicción top-1
(`reportes/etapa2_ext_paneton_val.json`):

| Foto | e2_replay2 | e2_replay2_v2 | e2_ext | e2_ext_full |
|---|---|---|---|---|
| 14.50.24 (2) | papel_carton 0.83 | generales 0.76 | generales 0.63 | metales 0.87 |
| 14.50.25 (1) | generales 0.96 | generales 0.98 | generales 1.00 | generales 1.00 |
| 14.50.25     | generales 0.52 | generales 0.92 | generales 1.00 | generales 0.93 |

Modelos para prueba en vivo: `modelos/etapa2_ext.pt` (e2_ext) y `modelos/etapa2_ext_full.pt` (e2_ext_full),
con su `.json`. Control sin externos: `modelos/etapa2_v2.pt` (e2_replay2_v2, desde
`modelos/experimentos/etapa2_e2_replay2_v2_20261025_1848.pt`). juliaca/test NO se ha usado.

`dataset/val_camara/<clase>/<objeto>/`: objetos nuevos tomados con la cámara de la laptop. Evaluación
aparte (`evaluar.py --datos val_camara`), nunca para entrenar ni early stopping.

### val_camara (Grupo 3, 2026-10-26): 29 objetos nuevos, 293 fotos
Copia de `Paquetes de FOTOS\Grupo 3\val_camara_paquete-clasificadas\val_camara` (README §9).
Reclasificados por el usuario antes de evaluar: `plasticos/tetra_bonle` → `papel_carton/obj128_caja_tetra_bonle`
(decisión tetra) y `plasticos/cinta_peligro_obras` → `plasticos/obj129_cinta_peligro_obras`.

Anti-leakage (dHash <= 4 contra juliaca train/val/test, publico y externo; `reportes/val_camara_dedup.csv`):
1 coincidencia (dist 3), `peligrosos/obj106_estuche_audifonos/…13.53.00 (1).jpeg` vs
`publico/train/organico/companero_000974_dl_marron_01283.jpg`. Revisada a ojo: falso positivo
(estuche negro sobre sofá gris vs papa con moho sobre fondo blanco), no es la misma foto.

| modelo | top1 | F1 macro (7 clases) |
|---|---|---|
| etapa2_v2 | 0.4403 | 0.4309 |
| etapa2_ext | 0.4881 | 0.4606 |
| etapa2_ext_full | 0.5529 | 0.5183 |
| etapa1_ganador | 0.3515 | 0.3760 |

F1 por clase: `reportes/*_val_camara.json`. Por foto: `reportes/val_camara_por_foto.csv`. Accuracy
por objeto (% de fotos acertadas): `reportes/val_camara_por_objeto.csv`. Sin ganador elegido.

## Modelo actual
**`modelos/etapa2_ext_full.pt` (e2_ext_full)**, decisión del usuario (2026-10-26): máximo F1 macro en
val_camara (0.5183). Es el `--modelo` por defecto de `inferencia_webcam.py` e `inferencia_deteccion.py`.

| modelo | top1 val_camara | F1 macro val_camara (7 clases) |
|---|---|---|
| etapa2_v2 | 0.4403 | 0.4309 |
| etapa2_ext | 0.4881 | 0.4606 |
| **etapa2_ext_full** | **0.5529** | **0.5183** |
| etapa1_ganador | 0.3515 | 0.3760 |

F1 por clase en val_camara (de `reportes/*_val_camara.json`):

| clase | n | etapa2_v2 | etapa2_ext | etapa2_ext_full | etapa1_ganador |
|---|---|---|---|---|---|
| generales | 122 | 0.578 | 0.614 | 0.691 | 0.134 |
| metales | 13 | 0.259 | 0.174 | 0.298 | 0.062 |
| organico | 20 | 0.737 | 0.757 | 0.769 | 0.788 |
| papel_carton | 26 | 0.509 | 0.415 | 0.475 | 0.489 |
| peligrosos | 22 | 0.384 | 0.464 | 0.429 | 0.360 |
| plasticos | 73 | 0.216 | 0.311 | 0.467 | 0.408 |
| vidrio | 17 | 0.333 | 0.490 | 0.500 | 0.390 |

> **Advertencia:** val_camara se usó para ELEGIR el modelo, así que su 0.5183 es optimista
> (sesgo de selección; además son solo 29 objetos y 293 fotos, sin `sin_residuo`). No es la medida
> final. La medida final será `dataset/juliaca/test`, evaluado UNA sola vez con este modelo
> (`python scripts/evaluar.py --split test`, cuando el usuario lo pida). juliaca/test NO se ha usado.

## Datos externos nuevos (`e1_ext2` → `e2_ext2_full`, 2026-10-26)
**Un solo factor:** `dataset/externo/train` pasa de 179 imágenes (TACO: generales 139, metales 40) a 5.684
(+ RealWaste, TACO extra y TrashNet; ver `datos/fuentes.md`). La receta es la misma que `e1_ext` → `e2_ext_full`:
- etapa 1: yolo26s-cls, 224 px, 60 épocas, paciencia 15, semilla 42, `--externo`;
- etapa 2: desde la etapa 1, AdamW lr 1e-4, freeze 6, replay 2, `--externo`, 30 épocas, paciencia 10, semilla 42.

`juliaca` y `publico` no cambiaron desde el 25-10 18:51 (ningún archivo más nuevo que el modelo de `e2_ext_full`);
`auditar.py` sin errores. Por la receta, en la etapa 2 el externo se suma completo: en train quedan 5.684 externas
frente a 145 fotos de juliaca y 250 de replay.

| | e1_ext | **e1_ext2** | e2_ext_full (actual) | **e2_ext2_full** |
|---|---|---|---|---|
| externo/train | 179 | 5.684 | 179 | 5.684 |
| épocas (mejor\*) | 60 (52) | 60 (59) | 17 (7) | 11 (5) |
| minutos | 15.3 | 20.1 | 1.0 | 2.3 |
| F1 macro publico/val | 0.9153 | 0.9071 | 0.8849 | 0.8837 |
| F1 macro juliaca/val (28 fotos) | 0.4274 | 0.3912 | 0.8186 | 0.8057 |
| top-1 juliaca/val | 0.5000 | 0.5000 | 0.7500 (21/28) | 0.7500 (21/28) |
| **F1 macro val_camara (293 fotos)** | 0.3510 | **0.4147** | **0.5183** | **0.4301** |
| top-1 val_camara | 0.3481 | 0.3652 | 0.5529 | 0.4300 |

\* Mejor época según `reportes/experimentos.csv` (primera con el máximo top-1). En `e2_ext2_full`, Ultralytics guardó
`best.pt` en la época 1 (criterio (top1+top5)/2; log: "Best results observed at epoch 1").

**val_camara, F1 por clase (aciertos/n)**, que es donde va el peso principal:

| clase | n | e1_ext | e1_ext2 | e2_ext_full | e2_ext2_full |
|---|---|---|---|---|---|
| generales | 122 | 0.293 (22) | 0.165 (11) | **0.691 (75)** | 0.494 (44) |
| metales | 13 | 0.065 (1) | 0.095 (2) | **0.298 (7)** | 0.050 (1) |
| organico | 20 | 0.552 (8) | 0.889 (16) | 0.769 (15) | 0.800 (14) |
| papel_carton | 26 | 0.412 (20) | 0.358 (24) | 0.475 (14) | 0.396 (20) |
| peligrosos | 22 | 0.319 (15) | 0.438 (14) | 0.429 (12) | 0.413 (13) |
| plasticos | 73 | 0.371 (26) | 0.424 (28) | 0.467 (28) | 0.390 (23) |
| vidrio | 17 | 0.444 (10) | 0.533 (12) | 0.500 (11) | 0.468 (11) |

Confusiones principales en val_camara (real → predicho):
- e2_ext_full: plasticos → metales 12, generales → papel_carton 12, plasticos → vidrio 11, generales → plasticos 11,
  generales → peligrosos 10.
- e2_ext2_full: **generales → papel_carton 41**, generales → peligrosos 17, plasticos → vidrio 16,
  generales → plasticos 12, plasticos → metales 10.

En juliaca/val, `e2_ext2_full` y `e2_ext_full` aciertan lo mismo en todas las clases (21/28; metales 4/9,
generales 2/3, peligrosos 2/2). El F1 por clase cambia por los falsos positivos de otras clases, no por aciertos.
En publico/val la etapa 1 baja en plasticos (0.851 → 0.828), vidrio (0.909 → 0.894) y peligrosos (aciertos 260 → 249);
la etapa 2 queda igual (0.8849 vs 0.8837).

Informes: `reportes/etapa1_e1_ext2_20261026_2019_{publico_val,juliaca_val,val_camara}.json`,
`reportes/etapa2_e2_ext2_full_20261026_2021_{publico_val,juliaca_val,val_camara}.json`,
`reportes/etapa2_ext_full_publico_val.json`, `reportes/etapa1_e1_ext_20261025_1657_val_camara.json`,
`reportes/val_camara_por_foto.csv` (7 modelos). Modelos en `modelos/experimentos/` (no se suben).

**Conclusión: EMPEORA.** El peso principal va a val_camara (293 fotos de 29 objetos), porque juliaca/val tiene solo
28 fotos y ahí los dos modelos aciertan exactamente lo mismo (empate). En val_camara, el modelo final `e2_ext2_full`
baja de F1 0.5183 a 0.4301 (top-1 0.5529 → 0.4300), sobre todo en las clases que importaban:
- **generales:** 75 → 44 aciertos; los envoltorios pasan a papel_carton (41 fotos);
- **metales:** 7 → 1;
- **peligrosos:** se queda igual (12 → 13).

Mejora un poco papel_carton (14 → 20 aciertos) y organico se mantiene.
La etapa 1 sola sí mejora en val_camara (e1_ext2 0.4147 frente a e1_ext 0.3510; organico 8 → 16), pero empeora en
publico/val (0.9071 frente a 0.9153; una diferencia de 0.008, mayor que el umbral de 0.005).
Hipótesis, no probada: en la etapa 2 el externo domina el train (5.684 frente a 145 fotos de juliaca; papel_carton
1.209 frente a generales 357), y el mejor modelo sale en la época 1, así que casi no se adapta a juliaca.
Siguiente experimento de un factor, si se quiere: `e2` desde `e1_ext2` **sin** `--externo` en la etapa 2.
El modelo actual sigue siendo `etapa2_ext_full` (no cambia el defecto de demo ni de inferencia). juliaca/test NO se usó.
val_camara ya sirvió para elegir `etapa2_ext_full`, así que sus números tienen sesgo de selección, aunque la
comparación entre los dos modelos sigue siendo válida.

## Pendientes
- ~~tetra/multicapa en papel_carton de publico (revision\tetra\), decisión diferida.~~ CERRADO
  (2026-10-26): los envases tetra/multicapa de cartón se clasifican como `papel_carton` y se
  mantienen en publico. publico no cambia, así que no hay que reentrenar e1_base.
