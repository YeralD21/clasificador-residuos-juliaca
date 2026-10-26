# Clasificador de residuos — Juliaca (NTP 900.058:2019)

Clasificación de un residuo por imagen en el tacho que le corresponde según el **código de colores ampliado de la NTP 900.058:2019** (instituciones, empresas y residuos no municipales). Se usa **YOLO26-cls de Ultralytics (PyTorch)** preentrenado, con fine-tuning sobre datos públicos y luego con fotos propias tomadas en Juliaca.

- **Entrenamiento:** PC con RTX 3060 12 GB
- **Uso en vivo:** laptop con RTX 3050 4 GB + webcam Logitech MX Brio

## 1. Qué clasifica el modelo: el MATERIAL, no el color

El color de la norma es el **color del tacho**, no el color del objeto. El modelo reconoce el material y el programa muestra el tacho:

| Clase (carpeta) | Tacho | Ejemplos | Ojo con |
|---|---|---|---|
| `papel_carton` | 🔵 Azul | cajas, hojas, periódico, cartón de huevos | papel con grasa o comida → `generales` |
| `metales` | 🟡 Amarillo | latas de bebida, conservas, tapas metálicas | latas de pintura/aerosol → `peligrosos` |
| `plasticos` | ⚪ Blanco | botellas PET de **cualquier color**, envases, bolsas simples | bolsas metalizadas de snack → `generales` |
| `vidrio` | ⚫ Gris | botellas y frascos verdes, marrones, transparentes | focos y fluorescentes → `peligrosos` |
| `organico` | 🟤 Marrón | cáscaras, restos de comida, hojas | |
| `peligrosos` | 🔴 Rojo | pilas, baterías, focos, jeringas, envases de químicos | |
| `generales` | ⚫ Negro | bolsas de chisitos (multicapa), servilletas sucias, tecnopor sucio, envolturas mixtas | |
| `sin_residuo` | — | fondo vacío, solo la mano | evita que el modelo "adivine" cuando no hay nada |

Para que el modelo **no aprenda "blanco = plástico"**: toma fotos de varios colores en cada clase (botellas verdes, azules y transparentes; latas de distintas marcas). Además, el entrenamiento varía el tono de las imágenes (`hsv_h`).

## 2. Plan: dos etapas de fine-tuning

```
yolo26s-cls.pt (preentrenado en ImageNet)
      │  ETAPA 1: aprende las 8 clases con datos públicos (miles de fotos)
      ▼
modelos/etapa1_*.pt
      │  ETAPA 2: se ajusta a Juliaca con tus fotos (+ un poco de datos públicos para no olvidar)
      ▼
modelos/etapa2_*.pt
```

Los entrenamientos y experimentos se evalúan **solo en validación** (etapa 1: `publico/val`, etapa 2: `juliaca/val`); ahí se elige el modelo de cada etapa (ver `EXPERIMENTOS.md`). `dataset/juliaca/test` **no** se evalúa automáticamente: se usa **una sola vez**, al final, con los dos modelos elegidos (`scripts/evaluar.py --split test`), para medir cuánto mejoró con datos locales.

- **Etapa 1:** learning rate automático, hasta 60 épocas.
- **Etapa 2:** learning rate bajo (1e-4), primeras 6 capas congeladas, hasta 30 épocas, y *replay* de 1 foto pública por cada foto tuya.

## 3. Estructura

```
dataset/
├── crudo/<fuente>/<clase_origen>/        descargas públicas tal cual (Roboflow "Folder Structure", Kaggle)
├── publico/{train,val,test}/<clase>/     GENERADO por scripts/preparar_publico.py
├── juliaca/                              TUS FOTOS — partición MANUAL
│   ├── train/<clase>/<objeto>/*.jpg
│   ├── val/<clase>/<objeto>/*.jpg
│   └── test/<clase>/<objeto>/*.jpg       ← evaluación final, nunca se entrena con esto
├── externo/train/<clase>/                datos externos reales (regla 1), SOLO train; ver §7.1
├── _etapa1/                              GENERADO por entrenar.py --etapa 1 --externo (no editar)
└── _etapa2/                              GENERADO por entrenar.py --etapa 2 (no editar)
scripts/        código
modelos/        best.pt de cada etapa + .json con versiones
reportes/       métricas en JSON de cada evaluación
runs/           salidas de Ultralytics (gráficas, results.csv)
class_mapping.csv   clase pública → clase NTP (la columna aprobado la llenas TÚ)
```

### Partición manual sin leakage: una carpeta por objeto

Cada **objeto físico** tiene su propia subcarpeta, y **todas sus fotos van en un solo split**:

```
dataset/juliaca/train/plasticos/obj001_botella_inca_kola/   (5 fotos, distintos ángulos y fondos)
dataset/juliaca/train/plasticos/obj002_botella_san_luis/
dataset/juliaca/val/plasticos/obj003_envase_yogurt/
dataset/juliaca/test/plasticos/obj004_botella_cielo_verde/
```

Si la misma botella aparece en train y en test, el modelo la **memoriza** y la nota sale inflada. `scripts/auditar.py` lo detecta: marca error si un nombre de objeto aparece en dos splits o si hay fotos idénticas entre splits, y avisa si dos fotos son casi iguales. Por eso **cada nombre de objeto debe ser único** (el número `objNNN` no se repite entre splits).

**Reparto sugerido por clase** (contando objetos, no fotos): ~60 % train, ~20 % val, ~20 % test. Por ejemplo, con 15 objetos de 4–5 fotos cada uno: 9 / 3 / 3 objetos.

Puedes copiar fotos del celular a mano en esas carpetas, o usar la Brio con `scripts/capturar_fotos.py --split train`, que crea las carpetas `objNNN` solas (tecla N = nuevo objeto).

## 4. Instalación (mismas versiones en PC y laptop)

1. Actualiza el driver NVIDIA: debe soportar **CUDA 13.0 o más** (verifícalo con `nvidia-smi`, campo "CUDA Version"). No necesitas el CUDA Toolkit: PyTorch trae su runtime.
2. Usa la **misma versión de Python** en los dos equipos (recomendado 3.13).
3. En el PRIMER equipo donde instales (PC o laptop) — ya hecho en la laptop, con torch 2.14.1+cu130:
   ```bash
   py -3.13 -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
   pip install -r requirements.txt
   pip freeze > requirements-lock.txt
   python scripts/verificar_entorno.py --guardar entorno_referencia.json
   git add requirements-lock.txt entorno_referencia.json && git commit -m "Fijar versiones" && git push
   ```
4. En el OTRO equipo: `git pull`, crea el venv con `py -3.13 -m venv .venv`, actívalo y
   ```bash
   pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130
   python scripts/verificar_entorno.py --comparar entorno_referencia.json
   ```

## 5. Flujo completo

```bash
# Datos públicos → dataset/crudo/<fuente>/<clase>/   (ver sección 7)
# Revisar class_mapping.csv y poner aprobado=SI a mano
python scripts/preparar_publico.py --max-por-clase 1500
# Tus fotos → dataset/juliaca/<split>/<clase>/<objeto>/
python scripts/auditar.py                                  # debe terminar "Sin errores"
python scripts/entrenar.py --etapa 1
python scripts/entrenar.py --etapa 2 --pesos modelos/etapa1_AAAAMMDD_HHMM.pt
python scripts/entrenar.py --etapa 2 --pesos yolo26s-cls.pt --replay 0   # opcional: solo Juliaca, para comparar
python scripts/evaluar.py --pesos <modelo_etapa1_elegido>.pt <modelo_etapa2_elegido>.pt   # validación (juliaca/val)
# Evaluación FINAL, UNA sola vez, con los modelos ya elegidos por val (queda en reportes/registro_test.csv):
python scripts/evaluar.py --pesos <modelo_etapa1_elegido>.pt <modelo_etapa2_elegido>.pt --split test
```

Cada época imprime una línea:
`[EPOCA  12/60]  41.3s | loss_train=... loss_val=... | top1_val=... top5_val=... | lr=... | GPU_pico=...GB`

Uso en vivo (laptop): en la PC haz `git add modelos/ reportes/ && git commit -m "Modelo etapa 2" && git push`; en la laptop `git pull` y ejecuta
```bash
python scripts/inferencia_webcam.py --modelo modelos/etapa2_AAAAMMDD_HHMM.pt
```
Con `S` guardas un recorte en `capturas_revisar/`. Tú decides su clase y lo mueves a `dataset/juliaca/...` para reentrenar.

## 6. Usar el modelo en la laptop

> **Modelo provisional:** `modelos/etapa2_avance.pt` es e2_replay2, elegido solo por validación (ver `EXPERIMENTOS.md`, "Avance etapa 2"). `dataset/juliaca/test` aún **no** se ha evaluado.

```bash
git pull
.venv\Scripts\activate
python scripts/verificar_entorno.py --comparar entorno_referencia.json      # debe decir "Todo coincide."
python scripts/inferencia_webcam.py --modelo modelos/etapa2_avance.pt --camara 0 --imgsz 224
```

- Si `.venv` no existe en la laptop, créalo como en la sección 4, paso 4: `py -3.13 -m venv .venv`, actívalo y `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130`.
- `--camara` es el índice de OpenCV de la MX Brio (por defecto `0`; si se abre otra cámara, prueba `1`). `--imgsz 224` es el tamaño con el que se entrenó. Opcionales: `--umbral` (por defecto `0.6`; por debajo muestra "NO SEGURO"), `--suavizado` (por defecto `8` frames promediados) y `--recuadro` (lado del recuadro como fracción del alto del frame; por defecto `0.75`, admite p. ej. `0.9` o `1.0`). Con `1.0` el recuadro es el cuadrado central de alto completo, que es exactamente lo que vería el clasificador si le pasaras el frame entero (Ultralytics redimensiona y recorta al centro).
- Ventana: se puede maximizar o redimensionar y el video ocupa toda la ventana. Antes quedaba una franja gris porque `cv2.imshow` sin `namedWindow` crea una ventana `WINDOW_AUTOSIZE`, que dibuja el frame a su tamaño nativo; ahora se crea con `WINDOW_NORMAL | WINDOW_FREERATIO` (`scripts/comun.py`, `crear_ventana`).
- Teclas: **Q** sale (o cerrar la ventana); **F** alterna pantalla completa; **S** guarda el recorte actual en `capturas_revisar/` (úsalo cuando el modelo se equivoque). Esas fotos no se clasifican solas: tú decides su clase y las pasas a `dataset/juliaca/<split>/<clase>/<objeto>/`.

### 6.1 Modo detección + clasificación (dos etapas, todo el frame)

`scripts/inferencia_deteccion.py` busca objetos en **toda** la imagen (varios a la vez, priorizando los cercanos) y clasifica cada uno con tu modelo:

1. **Detector de vocabulario abierto** (no se entrena): YOLOE con prompts de texto en inglés. Busca los de residuo (`--prompts`) y también los **negativos** (`--ignorar`: persona, cara, mano, ropa, muebles, pared...). Solo localiza las cajas; la clase la decide siempre el clasificador.
2. **Filtros**, en este orden (con `--bench` se imprime cuántas cajas descarta cada uno):
   - `ignorada`: cajas de los prompts negativos; no se clasifican ni se dibujan.
   - `duplicada`: dos prompts de residuo sobre el mismo objeto (p. ej. `bottle` y `cup`, IoU > 0.7); queda la de mayor confianza.
   - `roi`: el centro de la caja debe caer en la zona central `--roi` (por defecto `1.0` = todo el frame; si es menor se dibuja la zona).
   - `area`: área entre `--min-area` y `--max-area`.
   - `ropa_mueble`: se descarta la caja de residuo que coincide casi entera (IoU ≥ `--iou-ignorar`) con una caja de ropa o mueble (clothing, jacket, sofa, chair, shelf, bed, furniture...) de confianza igual o mayor, es decir, cuando el detector cree que es la prenda o el mueble. Persona, cara y mano **no** anulan cajas: un objeto en la mano sí se muestra. Pared, puerta, ventana y cortina tampoco, porque cubren todo el fondo y anularían a un objeto en primer plano (pasaba en `dataset/crudo/val`: una botella cercana contra una caja `wall` y unos vasos detectados como `curtain`).
   - `lejana`: **foco en lo cercano**, solo quedan las cajas con área ≥ `--foco-relativo` × la caja de residuo más grande del frame.
   - `borrosa`: **nitidez**, varianza del Laplaciano del recorte (a 128×128) ≥ `--nitidez-rel` × la del recorte más nítido del frame. Desactívalo con `--no-nitidez`.
   - `exceso`: se ordenan por área (mayor primero) y se quedan `--max-objetos`.
3. Cada caja se recorta **cuadrada con ~10 % de margen** y la clasifica `--modelo` a `imgsz 224`.
4. **Persistencia**: cada caja se sigue entre frames por IoU (≥ `--iou`). Se muestra solo si apareció en al menos `--persistencia` de los últimos `--historial` frames (3 de 5), y su clase es el **promedio de las probabilidades** del clasificador en esos frames. Se dibuja con el color del tacho y "clase (conf)"; si la confianza promedio es menor a `--umbral`, gris con "NO SEGURO". Se ignoran las cajas cuyo promedio da `sin_residuo`.

Teclas: **Q** sale, **F** pantalla completa, **S** guarda los recortes mostrados en `capturas_revisar/` (nombre `pred-<clase>_...jpg`, o `pred-NOSEGURO_...`; la clase la decides tú). La ventana se maximiza sin franja gris, igual que en `inferencia_webcam.py`.

```bash
python scripts/inferencia_deteccion.py --camara 0                                     # cámara de la laptop / MX Brio
python scripts/inferencia_deteccion.py --camara 0 --modelo modelos/etapa2_avance.pt   # con el modelo de etapa 2
python scripts/inferencia_deteccion.py --fuente video.mp4                             # probar con un video
python scripts/inferencia_deteccion.py --fuente foto.jpg --bench 100                  # FPS y descartes por filtro, sin ventana
```

| Argumento | Defecto | Qué hace |
|---|---|---|
| `--modelo` | `modelos/etapa1_ganador.pt` | clasificador. Por ahora es el que mejor generaliza en vivo; `modelos/etapa2_avance.pt` sigue disponible |
| `--detector` | `yoloe-26n-seg.pt` | detector; se descarga solo a `modelos/detectores/` (no se sube a git) |
| `--camara` / `--fuente` | `0` / — | índice de webcam, o video/imagen para pruebas |
| `--umbral` | `0.6` | confianza mínima del clasificador; por debajo, "NO SEGURO" |
| `--conf-det` | `0.35` | confianza mínima del detector (también para los negativos) |
| `--max-objetos` | `4` | máximo de cajas mostradas por frame (las de mayor área) |
| `--roi` | `1.0` | zona central (fracción del ancho y del alto) donde debe caer el centro de la caja; `1.0` = todo el frame, sin dibujar zona |
| `--min-area` / `--max-area` | `0.01` / `0.6` | área mínima y máxima de la caja (fracción del frame) |
| `--ignorar` | person, face, hand, clothing, jacket, furniture, sofa, chair, wall, door, window, shelf, curtain, bed | prompts negativos (`--ignorar ""` = ninguno) |
| `--iou-ignorar` | `0.6` | IoU con una caja de ropa/mueble a partir del cual se descarta la de residuo |
| `--foco-relativo` | `0.25` | área mínima relativa a la caja de residuo más grande (`0` = desactivado) |
| `--nitidez` / `--no-nitidez` | activado | filtro de recortes borrosos |
| `--nitidez-rel` | `0.3` | nitidez mínima relativa al recorte más nítido del frame |
| `--persistencia` / `--historial` | `3` / `5` | la caja debe aparecer en 3 de los últimos 5 frames; las probabilidades se promedian en esos 5 |
| `--iou` | `0.3` | IoU mínimo para considerar que dos cajas de frames seguidos son el mismo objeto |
| `--prompts` | bottle, can, plastic bag, snack wrapper, battery, phone, charger, cable, cup, glass jar, fruit peel, food waste, tube | qué residuos busca el detector. Se quitaron los genéricos `box`, `paper` y `container` (marcaban mesa, hojas y fondo); si los necesitas, pásalos con `--prompts "...,box,paper,container"` |
| `--imgsz-det` | `640` | tamaño del detector; bajarlo (p. ej. `480`, `416`) acelera |
Detectores que soporta Ultralytics 8.4.184 (`ultralytics/utils/downloads.py`):
- YOLOE con texto: `yoloe-26{n,s,m,l,x}-seg.pt`, `yoloe-11{s,m,l}-seg.pt`, `yoloe-v8{s,m,l}-seg.pt` (el codificador de texto `mobileclip2_b.ts` se descarga la primera vez a la raíz del repo; está en `.gitignore`).
- YOLOE prompt-free (vocabulario interno, ignora `--prompts`): los mismos con sufijo `-seg-pf.pt`.
- YOLO-World: `yolov8{s,m,l,x}-world.pt` y `yolov8{s,m,l,x}-worldv2.pt` (necesitan además el paquete `clip` de OpenAI).
- Recomendado para la RTX 3050 4 GB: **`yoloe-26n-seg.pt`** (el más liviano y el único tamaño `n`).

FPS medidos en la laptop (RTX 3050 Laptop 4 GB, `--bench`, sin ventana ni captura de cámara, imágenes de `dataset/crudo/val`): con `yoloe-26n-seg` el detector tarda ~22–26 ms. En total da ~42–45 FPS con 0 objetos, ~27 FPS con 1 objeto y ~23 FPS con 5 objetos. Con `yoloe-26s-seg` y 2 objetos da ~26 FPS. En vivo será algo menos por la captura y la ventana. Si baja de ~8 FPS: usa `--imgsz-det 480` o `416` y/o `--max-objetos 3`.

Con los filtros y la persistencia (valores por defecto actuales, misma laptop, `--bench 200`, imágenes de `dataset/crudo/val`): 1 caja mostrada ~22.5–23.8 ms (~42–44 FPS); 3 cajas mostradas ~22.6–24.1 ms (~41–44 FPS, el tope de `--max-objetos`); 0 cajas ~13.8–15 ms (~66–72 FPS; en las dos primeras corridas de la sesión, con la GPU recién activada, el detector tardó 36–39 ms y el total dio ~26–28 FPS). Una imagen con una sola caja de área 0.68 queda en 0 cajas por `--max-area 0.6`.

Detección global (valores por defecto actuales: `--roi 1.0`, `--ignorar`, foco y nitidez, `--max-objetos 4`, `etapa1_ganador.pt`; misma laptop, `--bench 200`). En `clf` entran también los filtros, la nitidez y la persistencia:

| Imagen (`dataset/crudo/val`) | Cajas detector → mostradas | Descartes (último frame) | Total |
|---|---|---|---|
| `Blanco/dl_blanco_00098` | 31 → 4 | area 10, borrosa 7, exceso 10 | 27.1 ms (36.9 FPS) |
| `Gris/dl_gris_00001` | 14 → 4 | lejana 2, borrosa 3, exceso 5 | 25.1 ms (39.9 FPS) |
| `Negro/dl_negro_00138` | 10 → 4 | lejana 3, borrosa 2, exceso 1 | 27.0 ms (37.0 FPS) |
| `Amarillo/dl_amarillo_00002` | 8 → 3 | duplicada 5 | 25.6 ms (39.0 FPS) |
| `Gris/dl_gris_00227` (con persona) | 11 → 4 | ignorada 1, duplicada 1, borrosa 1, exceso 4 | 24.7 ms (40.5 FPS) |
| `Amarillo/dl_amarillo_00018` (latas en la mano) | 9 → 4 | ignorada 3, duplicada 1, borrosa 1 | 26.2 ms (38.2 FPS) |
| `Azul/dl_azul_00016` (envase en la mano) | 4 → 1 | ignorada 3 | 21.4 ms (46.8 FPS) |
| `Azul/dl_azul_00098` (solo personas) | 7 → 0 | ignorada 7 | 15.3 ms (65.6 FPS) |

Con `--modelo modelos/etapa2_avance.pt` en `Gris/dl_gris_00001`: 25.6 ms (39.1 FPS), los mismos descartes. En las 2100 imágenes de `dataset/crudo/val` (una pasada por imagen, sin persistencia) el detector dio 2038 cajas; los filtros dejaron 1078 y descartaron: ignorada 543, duplicada 107, roi 0, area 78, ropa_mueble 0, lejana 44, borrosa 73 y exceso 115. `ropa_mueble` no se activó en val (son fotos de producto); la regla se comprobó con cajas sintéticas: una bolsa que coincide con una chaqueta más segura se descarta, y una botella pequeña sobre la chaqueta, en la mano o delante de una pared se muestra.

## 7. Datos públicos

| Fuente (carpeta en `crudo/`) | Contenido | Nota |
|---|---|---|
| `garbage12` | Kaggle *Garbage Classification (12 classes)*; mejor la versión **ENHANCED** de huberthamelin (etiquetas corregidas a mano) | ya está en `class_mapping.csv` |
| `roboflow_<nombre>` | Roboflow Universe, exportado como **Folder Structure** | al correr `preparar_publico.py` sus clases se agregan al CSV con `aprobado=NO` |

Antes de aprobar `garbage12/plastic`, abre la carpeta y **saca a mano** las bolsas de snack metalizadas (van a `generales`). Revisa la licencia de cada dataset y cítalo en tu informe.

### 7.1 Datos externos (regla 1 de CLAUDE.md)

Van solo a `dataset/externo/train/` (nunca a val/test ni a `dataset/juliaca`), cada imagen queda registrada en `datos_externos.csv` y antes de usarse se eliminan los casi duplicados (dHash <= 4) contra todos los splits de `publico` y `juliaca` (`reportes/externo_dedup.csv`). Los recortes no se suben a git (`dataset/**`).

`auditar.py` vuelve a comprobarlo antes de cada entrenamiento (registro, formatos, nada en val/test, dHash <= 4 = ERROR). Se usan solo con `--externo` (apagado por defecto) en `entrenar.py`/`experimentar.py`: en la etapa 1 se arma `dataset/_etapa1` (publico/train + externo; val = publico/val) y en la etapa 2 se suman completos a `_etapa2/train` aparte del replay. El `.json` del modelo guarda `parametros.externo` = n imágenes externas por clase.

| Fuente | Script | Categorías → destino | Licencia |
|---|---|---|---|
| TACO, `data/annotations.json` (commit `70479ad`) | `python scripts/importar_taco.py` | Crisp packet y Other plastic wrapper → `train/generales`; Aluminium foil → `train/metales`. Revisión manual por recorte en `reportes/externo_taco_revision.csv`; los descartados se mueven a `_descartados/<categoria>/` | Anotaciones CC BY 4.0. Imágenes: CC BY 4.0 (licencia vacía, por defecto según tacodataset.org) u ODbL 1.0. Se excluyen las marcadas "CC" sin especificar. |

**Atribución:**
- Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection*. arXiv:2003.06975. http://tacodataset.org — anotaciones bajo CC BY 4.0.
- Imágenes con licencia ODbL: © OpenLitterMap & Contributors (https://openlittermap.com), Open Database License 1.0. Los recortes derivados de ellas se comparten, si se publican, bajo la misma licencia (share-alike).

## 8. Reglas de integridad

- No se generan ni sintetizan imágenes. Solo fotos reales.
- La clase de cada foto la decides tú (carpetas). Ningún script mueve fotos entre clases o splits de `dataset/juliaca`.
- `dataset/juliaca/test` no se usa para entrenar ni para elegir hiperparámetros.
- Las métricas del informe se copian de `reportes/*.json` y `runs/*/results.csv`, tal como salieron.

Las fotos (`dataset/`) no se suben a git (ver `.gitignore`); para pasarlas entre equipos usa un USB o Google Drive. Los modelos finales de `modelos/` (con su `.json` de versiones) y `reportes/` sí se suben.

## 9. Respaldo de fotos (fuera de git)

`dataset/` no se sube a git (`.gitignore` ignora `dataset/**` salvo las carpetas y los `.gitkeep`). Si se pierde el disco o se clona el repo en otro equipo, las fotos no vienen con él. Las carpetas de abajo son la copia de seguridad: no se editan a mano y no las modifica ningún script.

| Conjunto | Copia de seguridad | Destino en el repo |
|---|---|---|
| `val_camara` (Grupo 3) | `D:\Proyecto-Residuos-ML\Paquetes de FOTOS\Grupo 3\val_camara_paquete-clasificadas` | `dataset/val_camara/<clase>/<objeto>/` |

En la misma carpeta del Grupo 3 están también los `.zip` del paquete (`val_camara_paquete-CLASIFICADAS.zip` y `val_camara_paquete-IMAGENES-CRUDAS.zip`).
