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
> **Reemplazado el 2026-10-28** por `modelos/etapa2_j1028_s43.pt` (ver "Evaluación final en test", al final).
> `etapa2_ext_full.pt` queda como alternativa documentada en el README.

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

### Etapa 2 desde `e1_ext2` con menos externo (`e2_ext2_taco`, `e2_ext2_sinext`, 2026-10-26)
Los dos parten de `e1_ext2`, con la misma receta de etapa 2 (AdamW lr 1e-4, freeze 6, replay 2, 30 épocas,
paciencia 10, semilla 42). En train quedan las mismas 145 fotos de juliaca y el mismo replay de 250.
- **A `e2_ext2_taco`:** en la etapa 2 solo el externo original (`--externo-fuentes taco`: los `taco_*` de
  `importar_taco.py`, generales 139 + metales 40), igual que `e2_ext_full`. Frente a `e2_ext_full` lo único que
  cambia es el punto de partida (`e1_ext2` en vez de `e1_ext`). La opción `--externo-fuentes` filtra por prefijo
  de archivo (`entrenar.py`, `PREFIJO_FUENTE`), sin mover nada.
- **B `e2_ext2_sinext`:** etapa 2 sin `--externo`.

| | e2_ext_full (actual) | e2_ext2_full | **A e2_ext2_taco** | **B e2_ext2_sinext** |
|---|---|---|---|---|
| parte de | e1_ext | e1_ext2 | e1_ext2 | e1_ext2 |
| externo en etapa 2 | 179 (taco) | 5.684 (todo) | 179 (taco) | 0 |
| épocas corridas | 17 | 11 | 18 | 26 |
| época de `best.pt` (Ultralytics, (top1+top5)/2) | 7 | **1** | 8 | 16 |
| época con el máximo top-1 (`experimentos.csv`) | 7 | 5 | 8 | 16 |
| minutos | 1.0 | 2.3 | 1.1 | 1.5 |
| F1 macro publico/val | **0.8849** | 0.8837 | 0.8691 | 0.8764 |
| F1 macro juliaca/val (28 fotos) | 0.8186 | 0.8057 | 0.8037 | 0.8037 |
| aciertos juliaca/val | 21/28 | 21/28 | 21/28 | 21/28 |
| **F1 macro val_camara (293 fotos)** | **0.5183** | 0.4301 | 0.5031 | 0.4646 |
| top-1 val_camara | **0.5529** (162) | 0.4300 (126) | 0.5324 (156) | 0.5051 (148) |

**val_camara, F1 por clase (aciertos/n):**

| clase | n | e2_ext_full | e2_ext2_full | A taco | B sinext |
|---|---|---|---|---|---|
| generales | 122 | 0.691 (75) | 0.494 (44) | 0.687 (79) | 0.667 (76) |
| metales | 13 | **0.298 (7)** | 0.050 (1) | 0.227 (5) | 0.186 (4) |
| organico | 20 | 0.769 (15) | 0.800 (14) | 0.842 (16) | 0.718 (14) |
| papel_carton | 26 | 0.475 (14) | 0.396 (20) | 0.525 (16) | 0.385 (10) |
| peligrosos | 22 | 0.429 (12) | 0.413 (13) | 0.522 (12) | **0.600 (15)** |
| plasticos | 73 | **0.467 (28)** | 0.390 (23) | 0.300 (15) | 0.304 (17) |
| vidrio | 17 | 0.500 (11) | 0.468 (11) | 0.419 (13) | 0.393 (12) |

Confusiones principales en val_camara (real → predicho):
- e2_ext_full: plasticos → metales 12, generales → papel_carton 12, plasticos → vidrio 11, generales → plasticos 11.
- e2_ext2_full: generales → papel_carton 41, generales → peligrosos 17, plasticos → vidrio 16.
- A taco: **plasticos → vidrio 26**, generales → papel_carton 18, plasticos → generales 15, plasticos → metales 10.
- B sinext: **plasticos → vidrio 25**, generales → papel_carton 16, plasticos → generales 14, papel_carton → generales 10.

En juliaca/val, A y B aciertan lo mismo que e2_ext_full salvo 1 foto de generales (3/3 frente a 2/3) y 1 de metales
(3/9 frente a 4/9; el error extra va a generales). Es el mismo total, 21/28, y no es significativo. En publico/val
A y B quedan por debajo (0.8691 y 0.8764 frente a 0.8849); en A sube la confusión hacia generales
(papel_carton → generales 26, plasticos → generales 23).

Informes: `reportes/etapa2_e2_ext2_{taco_20261026_2041,sinext_20261026_2043}_{publico_val,juliaca_val,val_camara}.json`,
`reportes/experimentos/e2_ext2_{taco,sinext}.json`, `reportes/val_camara_por_foto.csv` (9 modelos).

**Conclusión: ninguno mejora al modelo actual; `e2_ext_full` sigue siendo el mejor en val_camara** (peso principal,
293 fotos frente a las 28 de juliaca/val, donde los cuatro empatan en 21/28).
- **El daño de `e2_ext2_full` venía de meter todo el externo nuevo en la etapa 2:** con el mismo punto de partida,
  quitarlo sube val_camara de 0.4301 a 0.5031 (A) o 0.4646 (B), y generales vuelve de 44 a 76-79 aciertos. Además,
  `best.pt` deja de salir en la época 1 (A: época 8, B: época 16). La hipótesis anterior se confirma.
- **El externo TACO en la etapa 2 ayuda algo:** A supera a B (0.5031 frente a 0.4646), sobre todo en papel_carton
  (16 frente a 10 aciertos) y organico.
- **Partir de `e1_ext2` en vez de `e1_ext` no mejora el modelo final:** A frente a e2_ext_full da 0.5031 frente a
  0.5183 (6 aciertos menos de 293). Es una diferencia pequeña, pero con un patrón claro: plasticos cae de 28 a 15
  aciertos (plasticos → vidrio 26), y no lo compensan las mejoras en peligrosos (F1 0.429 → 0.522), papel_carton
  (14 → 16), organico (15 → 16) y generales (75 → 79). metales baja de 7 a 5.
- B tiene el mejor peligrosos (15/22, F1 0.600) pero el F1 macro más bajo de los tres sin todo el externo.
No cambia el modelo por defecto (decisión del usuario). juliaca/test NO se usó. Recordatorio: val_camara ya se usó
para elegir `etapa2_ext_full`, así que su número es algo optimista para ese modelo.

### Cierre de la línea de datos públicos (decisión del usuario, 2026-10-26)
- **Los datos externos nuevos no mejoran el modelo final.** Ninguna variante con ellos (`e2_ext2_full`,
  `e2_ext2_taco`, `e2_ext2_sinext`) supera a `etapa2_ext_full` en val_camara (0.5183 frente a 0.4301, 0.5031 y
  0.4646). En juliaca/val los cuatro aciertan 21/28.
- **Modelo por defecto: `modelos/etapa2_ext_full.pt`** (`inferencia_deteccion.py`, `inferencia_webcam.py`,
  `demo.bat`), sin cambios.
- **`dataset/externo/train` se queda** con sus 5.684 imágenes (registro en `datos_externos.csv`, fuentes en
  `datos/fuentes.md`), disponible para otros experimentos.
- **Receta recomendada: en la etapa 2, externo solo TACO (`--externo --externo-fuentes taco`).** Meter todo el
  externo en la etapa 2 hace que domine el train y que `best.pt` salga en la época 1 (`e2_ext2_full`).
- **Para reproducir exactamente `etapa2_ext_full`** hay que usar `--externo-fuentes taco` en las **dos** etapas: se
  entrenó el 25-10, cuando `externo/train` solo tenía los `taco_*`, y hoy `--externo` sin filtro toma las 5.684.
  Por eso `e1_ext`, `e2_ext` y `e2_ext_full` de `experimentar.py` ahora llevan `--externo-fuentes taco`:
  `--repetir` reproduce lo que se corrió y no los datos del 26-10. Sus resultados registrados no cambian.

## Reentrenamiento con juliaca del 27-10 (`e2_ext_full_j1027`, 2026-10-27)
**Un solo factor: los datos de juliaca.** Se importó la sesión de la laptop del 27-10 (`importar_sesion.py`, 4 objetos
etiquetados a mano por el usuario): train + generales 8 fotos (servilleta) y + papel_carton 18 (hoja bond, hoja
impresa); val + papel_carton 8 (hoja cuadriculada, 1 objeto). juliaca/train pasa de 145 a 171 fotos y juliaca/val
de 28 a 36. Receta recomendada, la misma que `e2_ext_full`: desde `e1_ext`, AdamW lr 1e-4, freeze 6, replay 2,
`--externo --externo-fuentes taco` (179), 30 épocas, paciencia 10, semilla 42. Como el replay es 2× juliaca por clase,
también crece (papel_carton 6 → 42, generales 62 → 78): train 652 imágenes. `auditar.py` sin errores.

Como juliaca/val cambió, `etapa2_ext_full` se reevaluó en el val nuevo (`reportes/etapa2_ext_full_juliaca_val.json`).
publico/val y val_camara no cambiaron (al reevaluarlo salen los mismos informes, byte a byte).

| | etapa2_ext_full (actual) | **e2_ext_full_j1027** |
|---|---|---|
| juliaca/train | 145 | 171 |
| épocas corridas (época de `best.pt`) | 17 (7) | 21 (11) |
| minutos | 1.0 | 1.5 |
| F1 macro publico/val (1.866 fotos) | **0.8849** | 0.8807 |
| aciertos publico/val | **1651** | 1643 |
| F1 macro juliaca/val **nuevo** (36 fotos) | 0.6428 | **0.8421** |
| aciertos juliaca/val nuevo | 21/36 | **30/36** |
| aciertos en las 28 fotos de juliaca/val anteriores | 21/28 | 22/28 |
| **F1 macro val_camara (293 fotos)** | **0.5183** | 0.4647 |
| top-1 val_camara | **0.5529** (162) | 0.4949 (145) |

**F1 por clase (aciertos/n):**

| clase | publico/val A | publico/val B | juliaca/val A | juliaca/val B | val_camara A | val_camara B |
|---|---|---|---|---|---|---|
| generales | 0.867 (195/208) | 0.850 (187) | 0.364 (2/3) | 0.750 (3/3) | **0.691 (75/122)** | 0.609 (63) |
| metales | 0.900 (266/289) | 0.890 (263) | 0.571 (4/9) | 0.615 (4/9) | 0.298 (7/13) | 0.245 (6) |
| organico | 0.964 (251/254) | 0.971 (250) | 1.000 (1/1) | 1.000 (1/1) | 0.769 (15/20) | 0.789 (15) |
| papel_carton | 0.901 (245/286) | 0.899 (257) | **0.000 (0/8)** | **1.000 (8/8)** | 0.475 (14/26) | 0.400 (13) |
| peligrosos | 0.854 (239/288) | 0.844 (235) | 1.000 (2/2) | 0.800 (2/2) | 0.429 (12/22) | 0.393 (11) |
| plasticos | 0.828 (223/275) | 0.831 (226) | 0.750 (6/6) | 0.857 (6/6) | 0.467 (28/73) | 0.458 (30) |
| sin_residuo | – (0) | – | 0.857 (3/4) | 0.857 (3/4) | – (0) | – |
| vidrio | 0.882 (232/266) | 0.881 (225) | 0.600 (3/3) | 0.857 (3/3) | 0.500 (11/17) | 0.359 (7) |

A = etapa2_ext_full, B = e2_ext_full_j1027.

Confusiones principales (real → predicho):
- juliaca/val: A papel_carton → vidrio 4, metales → generales 4, papel_carton → plasticos 2, papel_carton → generales 2;
  B metales → generales 2, y 1 cada una: sin_residuo → plasticos, metales → vidrio/plasticos/peligrosos.
- val_camara: A plasticos → metales 12, generales → papel_carton 12, plasticos → vidrio 11, generales → plasticos 11,
  generales → peligrosos 10; B **generales → papel_carton 17**, generales → peligrosos 14, generales → plasticos 13,
  plasticos → generales 12, generales → metales 12, plasticos → vidrio 11.

Informes: `reportes/etapa2_e2_ext_full_j1027_20261027_1957_{publico_val,juliaca_val,val_camara}.json`,
`reportes/etapa2_ext_full_juliaca_val.json`, `reportes/experimentos/e2_ext_full_j1027.json`,
`reportes/val_camara_por_foto.csv` (10 modelos: los 9 anteriores + éste). Modelo en `modelos/experimentos/` (no se sube).

**Conclusión: juliaca/val mejora mucho, pero val_camara empeora; no hay ganador claro y el modelo por defecto no cambia
(decide el usuario).**
- **La subida en juliaca/val (21 → 30 de 36) viene casi entera de un solo objeto:** la hoja cuadriculada de val
  (papel_carton 0/8 → 8/8). En las 28 fotos de antes, B acierta 22 frente a 21. Ese objeto se fotografió en la misma
  sesión, persona, fondo y cámara que las hojas de train (obj002, obj003), así que el 8/8 es probablemente optimista:
  las fotos se parecen en todo menos en el objeto. Una sola hoja no basta para decir que papel_carton mejoró.
- **En val_camara (peso principal: 293 fotos de 29 objetos) B baja de 0.5183 a 0.4647** (162 → 145 aciertos):
  - **generales: 75 → 63.** Los errores generales → papel_carton suben de 12 a 17 (envoltorios), y también suben los
    que van a peligrosos (10 → 14) y a metales (9 → 12). Añadir 18 fotos de hojas de papel y 8 de una servilleta no
    ayuda con los envoltorios multicapa;
  - **papel_carton: 14 → 13 (F1 0.475 → 0.400).** Las hojas lisas no se parecen a los papel_carton de val_camara;
  - **vidrio: 11 → 7**, metales 7 → 6, peligrosos 12 → 11; plasticos sube un poco (28 → 30) y organico se queda en 15.
- **publico/val baja un poco** (0.8849 → 0.8807, 8 aciertos menos), dentro de lo esperable.
- Hipótesis, no probada: con replay 2, papel_carton pasa de 9 a 63 imágenes en train (juliaca + replay) y todas las de
  juliaca son hojas blancas de una sola sesión. Eso inclina las decisiones dudosas hacia papel_carton y generales sin
  aportar variedad. Parte de la diferencia en val_camara puede ser también ruido de la semilla, porque es un solo run.
- `reportes/experimentos.csv` mezcla experimentos evaluados en juliaca/val de 28 fotos (los anteriores) y de 36 (éste):
  su F1_val no se compara entre ellos. La comparación válida es la de esta tabla.

juliaca/test NO se usó. val_camara ya sirvió para elegir `etapa2_ext_full`, así que su número es algo optimista para
ese modelo.

## Selección curada de datos públicos en la etapa 2 (`e2_ext_full_sel100`, 2026-10-27)
**Pregunta:** ¿una selección pequeña y curada de publico/train y externo/train, parecida al uso real, ayuda en la etapa 2,
a diferencia de meter todo el externo (`e2_ext2_full`)?

**Selección (fase A, `scripts/seleccionar_curados.py`):** solo de imágenes ya aprobadas para train (publico/train y
externo/train: TACO extra, RealWaste, TrashNet), nunca val ni test. Fuera los `taco_*` (ya en la receta) y la muestra de
replay (semilla 42, replay 2), para no duplicar. Filtro automático: detector YOLOE con prompts por tipo (caja, corrugado,
periódico, revista, Tetra Pak...; envoltura, sachet, tecnopor, servilleta...), prompts negativos (ropa, zapatos, vasos
de papel, pañales...), como mucho 2 objetos, caja principal >= 6 % de la imagen, y en publico fuera el fondo liso de
estudio y las texturas a pantalla completa; en TACO extra el tipo sale de su categoría. Anti-leakage: ninguna candidata a
dHash <= 4 de juliaca (todos los splits), val_camara o publico val/test (`reportes/seleccion_curada/dedup.csv`: 0 casos;
solo casi duplicados entre candidatas). Para elegir **no** se miraron val_camara ni juliaca/val.
494 candidatas (papel_carton 300, generales 194); el usuario descartó 164 a mano (`reportes/seleccion_curada.txt`):
aprobadas papel_carton 228, generales 102.

**Lista final (`seleccionar_curados.py --final 100`, `reportes/seleccion_curada/seleccion_final.csv`):** el mismo número
en las dos clases, con tope 100 (decisión del usuario). Round-robin por tipo y, dentro de cada tipo, por fuente; en
generales entran primero todas las envolturas, sachets y servilletas aprobadas (28) y el tecnopor completa (72).
- papel_carton 100: TACO extra 35, RealWaste 32, publico 33; caja 12, periódico 11, papel de color 11, hoja 11,
  Tetra Pak 11, bolsa de papel 12, corrugado 12, revista 11, cartón de huevos 6, tubo de papel higiénico 3.
- generales 100: tecnopor (pieza, vaso, envase) 72, servilleta 16, snack 8, golosina 2, sachet 2.
`entrenar.py --seleccion <csv>` las copia a `_etapa2/train/<clase>/seleccion` sin mover nada (solo acepta rutas de
publico/train o externo/train de su clase; 0 saltadas por duplicadas). Se comprobó que el replay es idéntico (302).

**Experimento de un factor:** misma receta que `e2_ext_full_j1027` (desde `e1_ext`, AdamW lr 1e-4, freeze 6, replay 2,
`--externo --externo-fuentes taco`, 30 épocas, paciencia 10, semilla 42, juliaca del 27-10) **+ la selección**. Train:
852 imágenes (652 + 200). `auditar.py` sin errores.

| | etapa2_ext_full (actual) | e2_ext_full_j1027 | **e2_ext_full_sel100** |
|---|---|---|---|
| juliaca/train | 145 | 171 | 171 |
| selección curada | – | – | 100 + 100 |
| épocas corridas (época de `best.pt`) | 17 (7) | 21 (11) | 21 (11) |
| minutos | 1.0 | 1.5 | 1.6 |
| F1 macro publico/val | 0.8849 (1651) | 0.8807 (1643) | **0.8888 (1659)** |
| F1 macro juliaca/val (36 fotos) | 0.6428 (21/36) | **0.8421 (30/36)** | 0.6597 (27/36) |
| **F1 macro val_camara (293 fotos)** | **0.5183** (162) | 0.4647 (145) | 0.4150 (126) |
| top-1 val_camara | **0.5529** | 0.4949 | 0.4300 |

**F1 por clase (aciertos/n):**

| clase | publico/val A | B | C | juliaca/val A | B | C | val_camara A | B | C |
|---|---|---|---|---|---|---|---|---|---|
| generales | 0.867 (195/208) | 0.850 (187) | 0.866 (190) | 0.364 (2/3) | 0.750 (3) | 0.444 (2) | **0.691 (75/122)** | 0.609 (63) | 0.500 (50) |
| metales | 0.900 (266/289) | 0.890 (263) | 0.893 (268) | 0.571 (4/9) | 0.615 (4) | 0.462 (3) | 0.298 (7/13) | 0.245 (6) | 0.222 (5) |
| organico | 0.964 (251/254) | 0.971 (250) | 0.963 (245) | 1.000 (1/1) | 1.000 (1) | 0.000 (0) | 0.769 (15/20) | 0.789 (15) | 0.625 (10) |
| papel_carton | 0.901 (245/286) | 0.899 (257) | **0.914 (267)** | 0.000 (0/8) | 1.000 (8) | 1.000 (8) | 0.475 (14/26) | 0.400 (13) | 0.430 (**20**) |
| peligrosos | 0.854 (239/288) | 0.844 (235) | 0.855 (238) | 1.000 (2/2) | 0.800 (2) | 0.800 (2) | 0.429 (12/22) | 0.393 (11) | 0.386 (11) |
| plasticos | 0.828 (223/275) | 0.831 (226) | 0.847 (221) | 0.750 (6/6) | 0.857 (6) | 0.857 (6) | 0.467 (28/73) | 0.458 (30) | 0.386 (22) |
| sin_residuo | – | – | – | 0.857 (3/4) | 0.857 (3) | 0.857 (3) | – | – | – |
| vidrio | 0.882 (232/266) | 0.881 (225) | 0.885 (230) | 0.600 (3/3) | 0.857 (3) | 0.857 (3) | 0.500 (11/17) | 0.359 (7) | 0.356 (8) |

A = etapa2_ext_full, B = e2_ext_full_j1027, C = e2_ext_full_sel100.

Confusiones principales en val_camara (real → predicho):
- A: plasticos → metales 12, generales → papel_carton 12, plasticos → vidrio 11, generales → plasticos 11.
- B: generales → papel_carton 17, generales → peligrosos 14, generales → plasticos 13, plasticos → generales 12.
- C: **generales → papel_carton 37**, plasticos → generales 17, plasticos → vidrio 13, generales → peligrosos 13.
  C predice papel_carton 67 veces (A 33, B 39) y papel_carton → generales baja a 2 (A 8, B 6).

val_camara por objeto (aciertos / predichas papel_carton), A | B | C:
- sachets de Pizza Hut: ají 10/0 | 10/0 | **4/5**; orégano 4/0 | 2/1 | 1/6;
- envolturas: casino 18/4 | 15/3 | 12/9; tentación 8/5 | 5/7 | 3/9; chocman 10/1 | 7/3 | 7/4; morochas, triángulo y
  sublime casi sin cambio;
- papel_carton: caja cereal 6 | 6 | 7 de 8; caja bon o bon 6 | 6 | 8 de 10; **tetra bonle 2 | 1 | 5 de 8**.

Informes: `reportes/etapa2_e2_ext_full_sel100_20261027_2026_{publico_val,juliaca_val,val_camara}.json`,
`reportes/experimentos/e2_ext_full_sel100.json`, `reportes/val_camara_por_foto.csv` (11 modelos),
`reportes/seleccion_curada/` (candidatas.csv, dedup.csv, indice.csv, seleccion_final.csv; los mosaicos no se suben).

**Conclusión: la selección curada NO ayuda al modelo final; empeora val_camara (0.4647 → 0.4150 frente a B, el mismo
modelo sin selección).** El modelo por defecto no cambia.
- **papel_carton mejora donde se buscaba:** val_camara 13 → 20 aciertos (Tetra Pak 1 → 5 de 8, cajas impresas 12 → 15
  de 18), y publico/val papel_carton 257 → 267. publico/val sube en total (0.8807 → 0.8888).
- **Pero generales se hunde:** 63 → 50 aciertos en val_camara, porque generales → papel_carton pasa de 17 a 37. Caen sobre
  todo los sachets y envolturas impresas (sachet de ají 10/10 → 4/10). El modelo aprendió "superficie impresa = papel"
  y ahora la aplica también a los multicapa. plasticos también baja (30 → 22) y organico (15 → 10).
- **juliaca/val:** 30 → 27 de 36 frente a B (pierde 1 foto de organico, la única de la clase, 1 de metales y 1 de
  generales); papel_carton se queda en 8/8.
- Hipótesis, no probada: el desequilibrio está en el contenido, no en el número. papel_carton recibió 100 imágenes muy
  variadas de superficies impresas (cajas, revistas, Tetra Pak); generales, aunque también 100, recibió 72 de tecnopor
  (blanco y liso) y solo 12 envolturas o sachets impresos, porque publico/externo casi no tienen. Así, lo impreso queda
  del lado de papel_carton. Para generales harían falta envolturas y sachets reales (fotos propias), no más tecnopor.
- Es un solo run (semilla 42); la diferencia en val_camara (19 aciertos de 293 frente a B) es mayor que la que se vio
  entre variantes parecidas, pero no se repitió con otra semilla.

juliaca/test NO se usó. El modelo por defecto sigue siendo `etapa2_ext_full`. val_camara se tomó con celular (fotos de
WhatsApp, no con la cámara de la laptop) y ya sirvió para elegir `etapa2_ext_full`, así que su número es optimista para A.

## Cierre definitivo de la línea de datos públicos (decisión del usuario, 2026-10-27)
Tres intentos de mejorar el modelo final con datos públicos o con más replay; **ninguno supera a `etapa2_ext_full` en
val_camara** (F1 macro 0.5183, 162/293):

| intento | experimento | F1 val_camara | aciertos |
|---|---|---|---|
| externo completo en la etapa 2 (5.684 imágenes) | `e2_ext2_full` | 0.4301 | 126 |
| replay con la sesión del 27-10 (receta recomendada, juliaca 145 → 171) | `e2_ext_full_j1027` | 0.4647 | 145 |
| selección curada (100 papel_carton + 100 generales) | `e2_ext_full_sel100` | 0.4150 | 126 |

(Las variantes del 26-10 desde `e1_ext2`, `e2_ext2_taco` 0.5031 y `e2_ext2_sinext` 0.4646, tampoco lo superan.)

- **La selección curada mejora papel_carton pero hunde generales.** papel_carton sube de 13 a 20 aciertos (Tetra Pak y
  cajas impresas), pero generales → papel_carton pasa de 17 a 37 y generales baja de 63 a 50: los sachets y envolturas
  impresas se van a papel_carton. Las fuentes públicas casi no tienen sachets ni envolturas impresas (en la selección,
  12 de 100 de generales; el resto, tecnopor), así que lo impreso queda del lado de papel_carton.
- **Modelo por defecto: `modelos/etapa2_ext_full.pt`, sin cambios.** `dataset/externo/train`, la selección curada
  (`reportes/seleccion_curada/`) y `entrenar.py --seleccion` se quedan disponibles, pero no se usan en la receta.
- No se harán más experimentos con datos públicos. juliaca/test sigue sin usarse.

## Reentrenamiento con juliaca del 28-10, Plaza Vea (`e2_ext_full_j1028`, 2026-10-28)
**Un solo factor: los datos de juliaca.** Se importó la sesión del 28-10 (fotos de celular en Plaza Vea, 67 objetos,
379 fotos; objeto, clase y split decididos a mano por el usuario, `importar_sesion.py --permitir-test`). juliaca pasa a
train 443 fotos / 80 objetos (antes 171/32), val 83/18 (antes 36/9) y test 98/18 (antes 38/8; test no se usó).
`auditar.py` sin errores. Receta recomendada, la misma que `e2_ext_full`: desde `e1_ext`, AdamW lr 1e-4, freeze 6,
replay 2, `--externo --externo-fuentes taco`, 30 épocas, paciencia 10, semilla 42.

**Replay:** sigue en la proporción de la receta, 2× juliaca/train por clase, y ninguna clase llega al tope de publico/train.
sin_residuo no tiene replay porque publico no tiene esa clase. El externo TACO es fijo (179), así que su peso baja: 27 % del train
con juliaca del 27-10 (179/652), 12 % ahora (179/1.468).

| clase | juliaca | replay publico | externo taco | total |
|---|---|---|---|---|
| generales | 119 | 238 | 139 | 496 |
| metales | 72 | 144 | 40 | 256 |
| organico | 5 | 10 | 0 | 15 |
| papel_carton | 52 | 104 | 0 | 156 |
| peligrosos | 16 | 32 | 0 | 48 |
| plasticos | 102 | 204 | 0 | 306 |
| sin_residuo | 20 | 0 | 0 | 20 |
| vidrio | 57 | 114 | 0 | 171 |
| **total** | **443** | **846** | **179** | **1.468** |

Como juliaca/val cambió, `etapa2_ext_full` se reevaluó en el val nuevo (`reportes/etapa2_ext_full_juliaca_val.json`;
el informe de 36 fotos queda en el historial de git, commit 61034a3). publico/val y val_camara no cambiaron.

| | etapa2_ext_full (actual) | **e2_ext_full_j1028** |
|---|---|---|
| juliaca/train | 145 | 443 |
| épocas corridas (época de `best.pt`) | 17 (7) | 17 (7) |
| minutos | 1.0 | 1.8 |
| F1 macro publico/val (1.866 fotos) | **0.8849** (1651) | 0.8710 (1625) |
| F1 macro juliaca/val **nuevo** (83 fotos, 18 objetos) | 0.5566 (40/83) | **0.6619 (62/83)** |
| aciertos en las 36 fotos de juliaca/val anteriores | 21/36 | **28/36** |
| aciertos en las 47 fotos de val de Plaza Vea (9 objetos) | 19/47 | **34/47** |
| **F1 macro val_camara (293 fotos)** | 0.5183 | **0.5301** |
| top-1 val_camara | **0.5529 (162)** | 0.5324 (156) |

**F1 por clase (aciertos/n):**

| clase | publico/val A | B | juliaca/val A | B | val_camara A | B |
|---|---|---|---|---|---|---|
| generales | 0.867 (195/208) | 0.836 (193) | 0.438 (7/16) | 0.686 (12) | **0.691 (75/122)** | 0.616 (65) |
| metales | 0.900 (266/289) | 0.866 (261) | 0.480 (6/13) | 0.522 (6) | 0.298 (7/13) | 0.327 (8) |
| organico | 0.964 (251/254) | 0.978 (249) | 1.000 (1/1) | 0.000 (0) | 0.769 (15/20) | 0.722 (13) |
| papel_carton | 0.901 (245/286) | 0.903 (260) | 0.316 (3/13) | **0.889 (12)** | 0.475 (14/26) | 0.447 (17) |
| peligrosos | 0.854 (239/288) | 0.840 (220) | 0.500 (2/2) | 1.000 (2) | 0.429 (12/22) | **0.558** (12) |
| plasticos | 0.828 (223/275) | 0.792 (215) | 0.577 (15/27) | **0.842 (24)** | 0.467 (28/73) | 0.463 (28) |
| sin_residuo | – | – | 0.857 (3/4) | 0.857 (3) | – | – |
| vidrio | 0.882 (232/266) | 0.883 (227) | 0.286 (3/7) | 0.500 (3) | 0.500 (11/17) | 0.578 (13) |

A = etapa2_ext_full, B = e2_ext_full_j1028.

**Confusiones buscadas (real → predicho, A | B):**

| confusión | publico/val | juliaca/val | val_camara |
|---|---|---|---|
| generales → papel_carton | 5 \| 5 | 3 \| 1 | **12 \| 26** |
| papel_carton → generales | 9 \| 7 | 4 \| 1 | 8 \| 8 |
| plasticos → vidrio | 16 \| 10 | **6 \| 1** | 11 \| 10 |
| vidrio → plasticos | 11 \| 15 | 3 \| 4 | 2 \| 3 |
| metales → generales | 3 \| 5 | 4 \| 4 | 1 \| 5 |
| generales → metales | 0 \| 2 | 4 \| 3 | 9 \| 7 |
| plasticos → peligrosos | 5 \| 5 | 4 \| 0 | 4 \| 3 |
| generales → peligrosos | 3 \| 1 | 0 \| 0 | 10 \| 5 |
| peligrosos → generales / plasticos / metales | 11/13/14 \| 18/16/22 | 0 \| 0 | 1/1/5 \| 0/1/6 |

Por objeto (aciertos A → B / n):
- **Empaques flexibles frente a papel_carton.** En juliaca/val mejora: avena Santa Catalina 1 → 3/3, doypack
  ketchup 4 → 5/6, Tetra Pak Vio 3 → 4/5 y la hoja cuadriculada 0 → 8/8. En val_camara empeora: casino 18 → 13/23
  (→ papel_carton 4 → 7), tentación 8 → 4/16 (5 → 9) y sachet de ají 10 → 7/10 (0 → 3). B predice papel_carton 50
  veces, frente a 33 de A. Las cajas mejoran (cereal 6 → 8/8, bon o bon 6 → 7/10), pero el Tetra Pak bonle sigue en 2/8
  y ahora se va a generales (5).
- **Botellas de plástico frente a vidrio.** La salsa rosada pasa de 0 a 6/8 (A predecía vidrio 6) y la bolsa de refresco de
  val_camara de 8 a 12/23. Pero la botella de vidrio esmerilado (val) sigue en 0/4, ahora con las 4 a plasticos; el
  frasco de almendras se queda en 3/6 y en publico/val vidrio → plasticos sube de 11 a 15.
- **Latas frente a envolturas cilíndricas.** Sin mejora: la lata Maniak se queda en 4/9 (→ generales 4 en los dos), el tubo
  de Pringles pasa de 0 a 2/4 (→ metales 3 → 2), las latas Hey Fit se van ahora a generales (val_camara metales →
  generales 1 → 5) y la lata Monster de val se queda en 2/4.
- **Envases negros frente a peligrosos.** No se puede medir directamente, porque en los sets de validación no hay
  ningún envase negro (el único, el shampoo The Barberia, está en train). Lo que se ve: plasticos → peligrosos baja en
  juliaca/val (4 → 0: Gloria Pro, morada y gris, 3 → 5/6) y generales → peligrosos baja en val_camara (10 → 5). Pero en
  publico/val peligrosos pierde 19 aciertos (239 → 220) y se va más a metales, generales y plasticos.

Informes: `reportes/etapa2_e2_ext_full_j1028_20261028_1643_{publico_val,juliaca_val,val_camara}.json`,
`reportes/etapa2_ext_full_juliaca_val.json` (val nuevo), `reportes/experimentos/e2_ext_full_j1028.json`,
`reportes/val_camara_por_foto.csv`. Modelo en `modelos/experimentos/` (no se sube). El desglose por objeto de juliaca/val
se calculó prediciendo foto a foto con los dos modelos; sus totales coinciden con los informes (40 y 62 de 83).

**Conclusión: no hay ganador claro y el modelo por defecto no cambia (lo decide el usuario).**
- juliaca/val mejora mucho (F1 0.5566 → 0.6619; 40 → 62 de 83), pero es optimista. 34 de los 62 aciertos son de las
  47 fotos de val de Plaza Vea, tomadas el mismo día, en la misma tienda, con el mismo celular y por las mismas personas
  que los 52 objetos de train. En las 36 fotos de val de antes la mejora es menor (21 → 28) y sale entera de la hoja
  cuadriculada (0 → 8/8; sesión del 27-10, con el mismo sesgo). En los otros 8 objetos de antes B pierde 1 acierto
  (organico, la única foto de la clase).
- val_camara, el set más independiente, queda mezclado: F1 macro sube (0.5183 → 0.5301; peligrosos, vidrio, metales),
  pero los aciertos bajan (162 → 156), porque generales cae de 75 a 65. **La confusión principal que se buscaba
  corregir, generales → papel_carton, se duplica (12 → 26)**, sobre todo en las envolturas impresas (casino, tentación).
  Los doypacks y bolsas de alimentos de Plaza Vea no se parecen lo suficiente a las envolturas pequeñas de golosina de
  val_camara.
- publico/val baja (0.8849 → 0.8710, 26 aciertos menos; peligrosos 239 → 220).
- Es un solo run (semilla 42). La diferencia en val_camara (6 aciertos) está dentro de lo que se ha visto entre
  variantes parecidas.

juliaca/test NO se usó. val_camara ya sirvió para elegir `etapa2_ext_full`, así que su número es algo optimista para A.

### Variación entre semillas de `e2_ext_full_j1028` (42, 43, 44; 2026-10-28)
La misma receta con `--seed 43` y `--seed 44` (`e2_ext_full_j1028_s43`, `e2_ext_full_j1028_s44`). La semilla cambia
el entrenamiento y también la muestra de replay de publico (`armar_etapa2` baraja con esa semilla); los conteos por clase
son los mismos (1.468 imágenes). Épocas (época de `best.pt`): 42 → 17 (7), 43 → 19 (9), 44 → 22 (12).

**Contexto (usuario):** val_camara se tomó con el celular del usuario, igual que Plaza Vea, y la app móvil será la demo
principal. **Regla de decisión fijada antes de ver los resultados:** si la media de F1 macro en val_camara de las 3
semillas supera a `etapa2_ext_full`, se propone como modelo por defecto el de la mejor semilla; si no, se queda
`etapa2_ext_full`.

| | etapa2_ext_full | s42 | s43 | s44 | **media (mín–máx)** |
|---|---|---|---|---|---|
| F1 macro val_camara (293) | 0.5183 | 0.5301 | **0.5610** | 0.5401 | **0.5437 (0.5301–0.5610)** |
| aciertos val_camara | 162 | 156 | **166** | 160 | 160.7 (156–166) |
| F1 macro juliaca/val (83) | 0.5566 | 0.6619 | 0.6160 | **0.7626** | 0.6802 (0.6160–0.7626) |
| aciertos juliaca/val | 40 | 62 | 61 | **63** | 62.0 (61–63) |
| F1 macro publico/val (1.866) | **0.8849** | 0.8710 | 0.8739 | 0.8735 | 0.8728 (0.8710–0.8739) |
| aciertos publico/val | **1651** | 1625 | 1631 | 1629 | 1628.3 (1625–1631) |

**F1 por clase en val_camara** (etapa2_ext_full frente a la media de las 3 semillas; aciertos s42/s43/s44):

| clase | n | etapa2_ext_full | media semillas (mín–máx) | aciertos |
|---|---|---|---|---|
| generales | 122 | **0.691** (75) | 0.624 (0.610–0.645) | 65 / 71 / 64 |
| metales | 13 | 0.298 (7) | **0.351** (0.327–0.364) | 8 / 8 / 8 |
| organico | 20 | 0.769 (15) | **0.788** (0.722–0.842) | 13 / 14 / 16 |
| papel_carton | 26 | **0.475** (14) | 0.456 (0.442–0.479) | 17 / 17 / 17 |
| peligrosos | 22 | 0.429 (12) | **0.493** (0.440–0.558) | 12 / 12 / 11 |
| plasticos | 73 | 0.467 (28) | **0.490** (0.463–0.512) | 28 / 31 / 33 |
| vidrio | 17 | 0.500 (11) | **0.604** (0.550–0.684) | 13 / 13 / 11 |

generales → papel_carton en val_camara: etapa2_ext_full 12; semillas 26 / 27 / 23.

**Resultado según la regla: la media en val_camara (0.5437) supera a `etapa2_ext_full` (0.5183); incluso la peor
semilla (0.5301) la supera.** La mejor semilla en val_camara es la **43** (`modelos/experimentos/
etapa2_e2_ext_full_j1028_s43_20261028_1701.pt`: F1 0.5610, 166/293, la única con más aciertos que `etapa2_ext_full`).
Se propone como modelo por defecto; **no se cambia hasta que el usuario lo apruebe.**
- Ventaja estable en las 3 semillas: metales, peligrosos, plasticos y vidrio suben en val_camara, y juliaca/val
  sube mucho (aunque es optimista, ver arriba).
- **Lo que empeora de forma estable: generales** (F1 0.691 → 0.61–0.65), porque generales → papel_carton se duplica
  (12 → 23–27) en las 3 semillas. No es ruido: las envolturas impresas pequeñas se van a papel_carton. publico/val
  baja en las 3 (−20 a −26 aciertos).
- Elegir la semilla 43 por val_camara hace que su 0.5610 sea optimista (se eligió entre 3 mirando ese set). El número
  sin ese sesgo es la media, 0.5437. val_camara también fue el set con el que se eligió `etapa2_ext_full`.
- En juliaca/val la semilla 43 es la peor de las tres (0.6160), aunque sigue por encima de `etapa2_ext_full` (0.5566).

Informes: `reportes/etapa2_e2_ext_full_j1028_s43_20261028_1701_*.json`,
`reportes/etapa2_e2_ext_full_j1028_s44_20261028_1704_*.json`, `reportes/experimentos/e2_ext_full_j1028_s4{3,4}.json`,
`reportes/val_camara_por_foto.csv`. Modelos en `modelos/experimentos/` (no se suben). juliaca/test NO se usó.

## Evaluación final en test (2026-10-28)
**Modelo final (decisión del usuario, por validación, antes de mirar test): `modelos/etapa2_j1028_s43.pt`**, copia
idéntica (mismo SHA-1) de `modelos/experimentos/etapa2_e2_ext_full_j1028_s43_20261028_1701.pt`. Es el modelo por
defecto de `inferencia_deteccion.py`, `inferencia_webcam.py` y `demo.bat`. `etapa2_ext_full.pt` queda como alternativa.

`python scripts/evaluar.py --pesos modelos/etapa2_j1028_s43.pt modelos/etapa2_ext_full.pt --datos juliaca --split test`,
**una sola ejecución** (registro en `reportes/registro_test.csv`: 2026-10-28 17:08 y 17:09). `etapa2_ext_full` se evaluó
solo como referencia. **La elección ya estaba hecha y no cambia por este resultado.**

juliaca/test: 98 fotos de 18 objetos. 10 objetos (60 fotos) son de Plaza Vea (28-10) y 8 objetos (38 fotos) son de
antes. Ningún objeto de test está en train ni en val: la partición es manual y por objeto.

| | **etapa2_j1028_s43 (final)** | etapa2_ext_full (referencia) |
|---|---|---|
| F1 macro (8 clases) | **0.4452** | 0.4972 |
| top-1 | 0.5408 (53/98) | 0.5510 (54/98) |
| top-5 | 0.9796 | 0.9694 |
| aciertos en los 10 objetos de Plaza Vea (60 fotos) | 32 | 33 |
| aciertos en los 8 objetos anteriores (38 fotos) | 21 | 21 |

**F1 por clase (aciertos/n):**

| clase | n | etapa2_j1028_s43 | etapa2_ext_full |
|---|---|---|---|
| generales | 23 | **0.704 (19)** | 0.533 (12) |
| metales | 24 | 0.683 (14) | **0.744 (16)** |
| organico | 6 | 0.286 (1) | **0.500 (2)** |
| papel_carton | 6 | 0.000 (0) | **0.400 (3)** |
| peligrosos | 4 | **0.308 (2)** | 0.000 (0) |
| plasticos | 21 | 0.455 (10) | **0.600 (12)** |
| sin_residuo | 7 | 0.727 (4) | **0.800 (6)** |
| vidrio | 7 | 0.400 (3) | 0.400 (3) |

**Matriz de confusión de etapa2_j1028_s43** (filas = real, columnas = predicho):

| | gen | met | org | pap | pel | pla | sin | vid |
|---|---|---|---|---|---|---|---|---|
| generales | **19** | 0 | 0 | 2 | 0 | 1 | 0 | 1 |
| metales | 1 | **14** | 0 | 1 | 4 | 4 | 0 | 0 |
| organico | 1 | 0 | **1** | 0 | 0 | 3 | 0 | 1 |
| papel_carton | 6 | 0 | 0 | **0** | 0 | 0 | 0 | 0 |
| peligrosos | 2 | 0 | 0 | 0 | **2** | 0 | 0 | 0 |
| plasticos | 2 | 3 | 0 | 0 | 3 | **10** | 0 | 3 |
| sin_residuo | 0 | 0 | 0 | 2 | 0 | 1 | **4** | 0 |
| vidrio | 0 | 0 | 0 | 0 | 0 | 4 | 0 | **3** |

**Matriz de confusión de etapa2_ext_full** (referencia):

| | gen | met | org | pap | pel | pla | sin | vid |
|---|---|---|---|---|---|---|---|---|
| generales | **12** | 1 | 0 | 4 | 1 | 3 | 0 | 2 |
| metales | 2 | **16** | 0 | 1 | 5 | 0 | 0 | 0 |
| organico | 0 | 0 | **2** | 0 | 0 | 1 | 1 | 2 |
| papel_carton | 2 | 0 | 0 | **3** | 0 | 1 | 0 | 0 |
| peligrosos | 3 | 0 | 0 | 0 | **0** | 0 | 1 | 0 |
| plasticos | 3 | 0 | 0 | 0 | 5 | **12** | 0 | 1 |
| sin_residuo | 0 | 0 | 0 | 1 | 0 | 0 | **6** | 0 |
| vidrio | 0 | 2 | 0 | 0 | 0 | 2 | 0 | **3** |

**Aciertos por objeto** (`reportes/juliaca_test_por_objeto.csv`; los totales coinciden con los informes, 53 y 54):

| clase | objeto | n | s43 | errores s43 | ext_full | errores ext_full |
|---|---|---|---|---|---|---|
| generales | obj007 pasta dental Oral-B | 4 | 1 | pla 1, pap 1, vid 1 | 2 | pla 1, vid 1 |
| generales | obj077 doypack mostaza B&D (PV) | 6 | **6** | – | 4 | met 1, pap 1 |
| generales | obj081 bolsa Shin Ramyun (PV) | 7 | **6** | pap 1 | 3 | pap 3, pel 1 |
| generales | obj092 bolsa frutos secos Bell's (PV) | 6 | **6** | – | 3 | pla 2, vid 1 |
| metales | obj026 lata de atún | 16 | 9 | pel 4, pla 2, pap 1 | 8 | pel 5, gen 2, pap 1 |
| metales | obj074 lata de café Kirma (PV) | 8 | 5 | pla 2, gen 1 | **8** | – |
| organico | obj104 ajo (PV) | 6 | 1 | pla 3, vid 1, gen 1 | 2 | vid 2, pla 1, sin 1 |
| papel_carton | obj059 caja de Zucaritas (PV) | 6 | **0** | gen 6 | 3 | gen 2, pla 1 |
| peligrosos | obj030 pila Toshiba | 4 | 2 | gen 2 | 0 | gen 3, sin 1 |
| plasticos | obj016 botella agua Cielo | 2 | 2 | – | 2 | – |
| plasticos | obj018 botella alcohol en gel | 2 | 1 | pel 1 | 2 | – |
| plasticos | obj068 pote aceitunas negras (PV) | 5 | **5** | – | 3 | gen 1, pel 1 |
| plasticos | obj085 pote orégano Bell's (PV) | 6 | 2 | pel 1, vid 1, met 1, gen 1 | 1 | pel 2, gen 2, vid 1 |
| plasticos | obj108 pote de gel Ego, negro (PV) | 6 | **0** | met 2, vid 2, gen 1, pel 1 | 4 | pel 2 |
| sin_residuo | esc05 mesa oscura horizontal | 3 | 2 | pla 1 | 3 | – |
| sin_residuo | esc07 manta negra | 4 | 2 | pap 2 | 3 | pap 1 |
| vidrio | obj020 vaso rosado | 3 | 2 | pla 1 | 1 | pla 1, met 1 |
| vidrio | obj065 frasco encurtido Selection (PV) | 4 | 1 | pla 3 | 2 | met 1, pla 1 |

(PV = Plaza Vea, 28-10.) El desglose por objeto se calculó prediciendo foto a foto con los dos mismos modelos, en la
misma evaluación; no es un uso nuevo de test.

**Lectura:**
- **El modelo final queda por debajo de la referencia en test** (F1 macro 0.4452 frente a 0.4972; 53 frente a 54
  aciertos). La diferencia en aciertos es de 1 foto. La diferencia en F1 macro se debe sobre todo a clases con 4–7
  fotos y 1–2 objetos (papel_carton 0/6, organico 1/6, sin_residuo 4/7), así que el número es muy inestable.
- **Lo que se ve en validación se repite en test:**
  - las bolsas y doypacks impresos de generales mejoran mucho: 18/19 frente a 10/19 en los tres de Plaza Vea;
  - el sesgo hacia lo impreso se invierte: la caja de Zucaritas se va entera a generales (0/6; la referencia 3/6). Con
    un solo objeto de papel_carton en test, el F1 de esa clase es 0;
  - **envase negro**: el pote de gel Ego (negro con tapa naranja), que no tenía equivalente en val, falla 6/6
    (metales, vidrio, generales, peligrosos); la referencia acierta 4/6.
- Es un set pequeño (98 fotos, 18 objetos): un solo objeto mueve el F1 de su clase entre 0 y 1.

**juliaca/test ya está usado: no se vuelve a usar para elegir modelos, épocas ni hiperparámetros** (reglas 5 y 9 de
CLAUDE.md). Cualquier modelo futuro se compara solo en validación. Si se quiere otra medida final honesta, hace falta un
test nuevo con objetos que ningún modelo haya visto.

Informes: `reportes/etapa2_j1028_s43_juliaca_test.json`, `reportes/etapa2_ext_full_juliaca_test.json`,
`reportes/juliaca_test_por_objeto.csv`, `reportes/registro_test.csv`.

## Pendientes
- **Siguiente paso: fotos propias de juliaca** (`capturar` → `etiquetar` → `importar`). Prioridad:
  0. **sachets, envolturas impresas, cajas impresas y Tetra Pak**, para separar lo impreso multicapa (generales) de lo
     impreso de cartón (papel_carton), que es lo que los datos públicos no resolvieron: sachets de salsa y condimentos,
     envolturas de golosina y snack (cerradas y abiertas), cajas de cereal, galletas o medicinas, y cajitas de leche y
     jugo. Varios objetos de cada tipo, con distintos fondos y sesiones, y algunos en val (por objeto);
  1. **plasticos y vidrio**: la confusión plasticos → vidrio es la principal en val_camara (11 con `e2_ext_full`;
     25-26 al partir de `e1_ext2`). Hacen falta plásticos transparentes y vidrios lado a lado: botellas PET
     transparentes, vasos y tapers frente a frascos y botellas de vidrio;
  2. **metales**: F1 0.298 en val_camara (7/13); en juliaca/val, metales → generales 4;
  3. **peligrosos**: F1 0.429 en val_camara (12/22); ninguna fuente externa con licencia clara lo refuerza
     (solo 80 externas);
  4. **generales**: generales → papel_carton 12 y → plasticos 11 en val_camara (envoltorios multicapa).
  Para referencia, juliaca/train tiene hoy (27-10) plasticos 28, vidrio 19, metales 26, peligrosos 16, generales 39
  y papel_carton 21 fotos (solo hojas de papel de una sesión: faltan cartón, cajas y papel de otros fondos);
  organico (2) sigue con muy pocas. organico y papel_carton no tienen fotos en test (aviso de `auditar.py`).
  Al reentrenar se compara con `etapa2_ext_full` solo en validación (regla 9).
- ~~tetra/multicapa en papel_carton de publico (revision\tetra\), decisión diferida.~~ CERRADO
  (2026-10-26): los envases tetra/multicapa de cartón se clasifican como `papel_carton` y se
  mantienen en publico. publico no cambia, así que no hay que reentrenar e1_base.
