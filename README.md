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

### 6.1 Modo detección + clasificación (dos etapas, sin recuadro fijo)

`scripts/inferencia_deteccion.py` busca objetos en **toda** la imagen y clasifica cada uno con tu modelo:

1. **Detector de vocabulario abierto** (no se entrena): YOLOE con prompts de texto (`--prompts`, en inglés, separados por coma). Solo localiza las cajas; su etiqueta no se usa.
2. **Filtros** para que no detecte de más: solo cuentan las cajas con confianza ≥ `--conf-det`, cuyo **centro** cae dentro de la zona central `--roi` (dibujada como un rectángulo fino) y cuya área está entre `--min-area` y `--max-area`; como máximo `--max-objetos` por frame (las de mayor confianza).
3. Cada caja se recorta **cuadrada con ~10 % de margen** y la clasifica `--modelo` a `imgsz 224`.
4. **Persistencia**: cada caja se sigue entre frames por IoU (≥ `--iou`). Se muestra solo si apareció en al menos `--persistencia` de los últimos `--historial` frames (3 de 5), y su clase es el **promedio de las probabilidades** del clasificador en esos frames. Se dibuja con el color del tacho y "clase (conf)"; si la confianza promedio es menor a `--umbral`, gris con "NO SEGURO". Se ignoran las cajas cuyo promedio da `sin_residuo`.

Teclas: **Q** sale, **F** pantalla completa, **S** guarda los recortes mostrados en `capturas_revisar/`. La ventana se maximiza sin franja gris, igual que en `inferencia_webcam.py`.

```bash
python scripts/inferencia_deteccion.py --camara 1                       # MX Brio en la laptop
python scripts/inferencia_deteccion.py --fuente video.mp4               # probar con un video
python scripts/inferencia_deteccion.py --fuente foto.jpg --bench 100    # medir FPS sin ventana
```

| Argumento | Defecto | Qué hace |
|---|---|---|
| `--modelo` | `modelos/etapa2_avance.pt` | clasificador (tu modelo) |
| `--detector` | `yoloe-26n-seg.pt` | detector; se descarga solo a `modelos/detectores/` (no se sube a git) |
| `--camara` / `--fuente` | `0` / — | índice de webcam, o video/imagen para pruebas |
| `--umbral` | `0.6` | confianza mínima del clasificador; por debajo, "NO SEGURO" |
| `--conf-det` | `0.35` | confianza mínima del detector |
| `--max-objetos` | `3` | máximo de cajas clasificadas y mostradas por frame (las de mayor confianza) |
| `--roi` | `0.8` | zona central (fracción del ancho y del alto) donde debe caer el centro de la caja; `1.0` = todo el frame |
| `--min-area` | `0.01` | área mínima de la caja (fracción del frame) |
| `--max-area` | `0.6` | área máxima de la caja (fracción del frame); descarta cajas que abarcan mesa/fondo |
| `--persistencia` / `--historial` | `3` / `5` | la caja debe aparecer en 3 de los últimos 5 frames; las probabilidades se promedian en esos 5 |
| `--iou` | `0.3` | IoU mínimo para considerar que dos cajas de frames seguidos son el mismo objeto |
| `--prompts` | bottle, can, plastic bag, snack wrapper, battery, phone, charger, cable, cup, glass jar, fruit peel, food waste, tube | qué busca el detector. Se quitaron los genéricos `box`, `paper` y `container` (marcaban mesa, hojas y fondo); si los necesitas, pásalos con `--prompts "...,box,paper,container"` |
| `--imgsz-det` | `640` | tamaño del detector; bajarlo (p. ej. `480`, `416`) acelera |

Detectores que soporta Ultralytics 8.4.184 (`ultralytics/utils/downloads.py`):
- YOLOE con texto: `yoloe-26{n,s,m,l,x}-seg.pt`, `yoloe-11{s,m,l}-seg.pt`, `yoloe-v8{s,m,l}-seg.pt` (el codificador de texto `mobileclip2_b.ts` se descarga la primera vez a la raíz del repo; está en `.gitignore`).
- YOLOE prompt-free (vocabulario interno, ignora `--prompts`): los mismos con sufijo `-seg-pf.pt`.
- YOLO-World: `yolov8{s,m,l,x}-world.pt` y `yolov8{s,m,l,x}-worldv2.pt` (necesitan además el paquete `clip` de OpenAI).
- Recomendado para la RTX 3050 4 GB: **`yoloe-26n-seg.pt`** (el más liviano y el único tamaño `n`).

FPS medidos en la laptop (RTX 3050 Laptop 4 GB, `--bench`, sin ventana ni captura de cámara, imágenes de `dataset/crudo/val`): con `yoloe-26n-seg` el detector tarda ~22–26 ms. En total da ~42–45 FPS con 0 objetos, ~27 FPS con 1 objeto y ~23 FPS con 5 objetos. Con `yoloe-26s-seg` y 2 objetos da ~26 FPS. En vivo será algo menos por la captura y la ventana. Si baja de ~8 FPS: usa `--imgsz-det 480` o `416` y/o `--max-objetos 3`.

Con los filtros y la persistencia (valores por defecto actuales, misma laptop, `--bench 200`, imágenes de `dataset/crudo/val`): 1 caja mostrada ~22.5–23.8 ms (~42–44 FPS); 3 cajas mostradas ~22.6–24.1 ms (~41–44 FPS, el tope de `--max-objetos`); 0 cajas ~13.8–15 ms (~66–72 FPS; en las dos primeras corridas de la sesión, con la GPU recién activada, el detector tardó 36–39 ms y el total dio ~26–28 FPS). Una imagen con una sola caja de área 0.68 queda en 0 cajas por `--max-area 0.6`.

Teclas: **Q** sale; **S** guarda **cada recorte** del frame en `capturas_revisar/` (nombre `pred-<clase>_...jpg`, o `pred-NOSEGURO_...`). Igual que en el otro modo, la clase la decides tú.

## 7. Datos públicos

| Fuente (carpeta en `crudo/`) | Contenido | Nota |
|---|---|---|
| `garbage12` | Kaggle *Garbage Classification (12 classes)*; mejor la versión **ENHANCED** de huberthamelin (etiquetas corregidas a mano) | ya está en `class_mapping.csv` |
| `roboflow_<nombre>` | Roboflow Universe, exportado como **Folder Structure** | al correr `preparar_publico.py` sus clases se agregan al CSV con `aprobado=NO` |

Antes de aprobar `garbage12/plastic`, abre la carpeta y **saca a mano** las bolsas de snack metalizadas (van a `generales`). Revisa la licencia de cada dataset y cítalo en tu informe.

## 8. Reglas de integridad

- No se generan ni sintetizan imágenes. Solo fotos reales.
- La clase de cada foto la decides tú (carpetas). Ningún script mueve fotos entre clases o splits de `dataset/juliaca`.
- `dataset/juliaca/test` no se usa para entrenar ni para elegir hiperparámetros.
- Las métricas del informe se copian de `reportes/*.json` y `runs/*/results.csv`, tal como salieron.

Las fotos (`dataset/`) no se suben a git (ver `.gitignore`); para pasarlas entre equipos usa un USB o Google Drive. Los modelos finales de `modelos/` (con su `.json` de versiones) y `reportes/` sí se suben.
