# Clasificador de residuos — Juliaca (NTP 900.058:2019)

Clasificación de un residuo por imagen en el tacho que le corresponde según el **código de colores ampliado de la NTP 900.058:2019** (instituciones, empresas y residuos no municipales). Se usa **YOLO26-cls de Ultralytics (PyTorch)** preentrenado, con fine-tuning sobre datos públicos y luego con fotos propias tomadas en Juliaca.

- **Entrenamiento:** PC con RTX 3060 12 GB
- **Uso en vivo:** laptop con RTX 3050 4 GB + webcam Logitech MX Brio

## Uso rápido

Lanzadores en la raíz del repo. Se abren con **doble clic**, o escribiendo su nombre en la terminal desde la raíz (`demo` en cmd, `.\demo` en PowerShell). Funcionan desde cualquier carpeta (hacen `cd` al repo con `%~dp0`) y activan `.venv` solos; si `.venv` no existe, lo dicen (créalo como en la sección 4).

| Lanzador | Qué hace | Script |
|---|---|---|
| `demo.bat` | Demo en vivo: detección + clasificación con los valores por defecto (cámara `auto`, detector `yoloe-26s-seg.pt`, modelo `modelos/etapa2_ext_full.pt`). Q sale, F pantalla completa, S guarda recortes. | `scripts/inferencia_deteccion.py` (sección 6.1) |
| `capturar.bat` | Sesión de fotos: **ESPACIO o clic** = foto JPG calidad 95, sin modificar y a la resolución completa de la cámara; **N** = nuevo objeto (`objNNN` automático); **Q** = salir. Muestra el objeto actual, cuántas fotos lleva y un destello al capturar. Las teclas van a la **ventana de la cámara** (haz clic en ella primero; en la terminal no hacen nada). Guarda en `<repo>\fotos_laptop\<AAAA-MM-DD>\objNNN\` (otra carpeta: `capturar --salida E:\fotos`), **fuera de `dataset\` y de git** (`.gitignore`): la clase y el split los decides tú después. `etiquetar` e `importar` buscan ahí y, si no hay sesiones, en la ubicación anterior (`..\fotos_laptop`, carpeta hermana del repo); ambos aceptan `--salida`/`--raiz`. | `scripts/capturar_sesion.py` |
| `etiquetar.bat` | Etiquetas **a mano** cada objeto de la última sesión de `capturar.bat`: ves un mosaico de sus fotos, eliges la **clase** (1 papel_carton, 2 metales, 3 plasticos, 4 vidrio, 5 organico, 6 peligrosos, 7 generales, 0 sin_residuo) y el **split** (T train, V val). S salta el objeto (vuelve a salir la próxima vez), D lo descarta, Z deshace, Q sale. Guarda en `fotos_laptop\<fecha>\etiquetas.csv` (objeto, clase, split, n_fotos, nota) tras cada decisión y retoma donde te quedaste. | `scripts/etiquetar_sesion.py` |
| `importar.bat` | Con doble clic **solo muestra** qué haría (`--dry-run`): a qué split/clase va cada objeto y la tabla de fotos/objetos antes y después. `importar --importar` copia de verdad a `dataset\juliaca\<split>\<clase>\objNNN_<fecha>-objMMM\` (siguiente número libre de juliaca, archivos `20261026_obj003_01.jpg`) y luego corre `auditar.py`. | `scripts/importar_sesion.py` |
| `evaluar.bat` | Evalúa todos los `modelos\*.pt` en `dataset\val_camara` (nunca en `juliaca\test`) y deja una pausa para leer la tabla. | `scripts/evaluar.py --datos val_camara` |

**Flujo de fotos nuevas:** `capturar` → `etiquetar` → `importar` (revisar) → `importar --importar`. La clase y el split los decides tú por objeto en `etiquetar` (reglas 2 y 3 de CLAUDE.md: la partición de juliaca es manual y por objeto); `importar` no decide nada, solo copia lo que dice `etiquetas.csv`. Garantías de `importar_sesion.py`:
- cada objeto va **entero a un solo split**, en su propia carpeta; solo `train` y `val` (`test` se arma aparte, y el CSV con `test` se rechaza);
- valida todo el CSV antes de escribir (clase, split, objeto repetido o sin fotos) y, si hay un error, no copia nada;
- nunca sobrescribe; un objeto ya importado (misma fecha y objeto) se salta, así que repetirlo no duplica;
- no toca las fotos originales de `fotos_laptop`;
- imagen: el frame completo escalado a lado mayor **1600 px**, JPG 95, igual que las fotos que ya hay en juliaca (1200×1600 y 1600×1200, del celular). `--recorte roi` aplica antes el cuadrado central de `capturar_fotos.py` (75 % del lado menor); ninguna foto actual de juliaca tiene ese recorte. `--lado-max 0` no redimensiona.

Los argumentos pasan al script: `demo --debug`, `demo --camara 1`, `capturar --camara 1`, `etiquetar --fecha 2026-10-26`, `importar --importar --fecha 2026-10-26`. La cámara `auto` imprime cuál eligió (ver sección 6). El `capturar_fotos.py` de antes (guarda directo en `dataset/juliaca/<split>/<clase>/` con recorte central) sigue disponible.

## 1. Qué clasifica el modelo: el MATERIAL, no el color

El color de la norma es el **color del tacho**, no el color del objeto. El modelo reconoce el material y el programa muestra el tacho:

| Clase (carpeta) | Tacho | Ejemplos | Ojo con |
|---|---|---|---|
| `papel_carton` | 🔵 Azul | cajas, hojas, periódico, cartón de huevos, envases Tetra Pak | papel con grasa o comida → `generales` |
| `metales` | 🟡 Amarillo | latas de bebida, conservas, tapas metálicas | latas de pintura/aerosol → `peligrosos` |
| `plasticos` | ⚪ Blanco | botellas PET de **cualquier color**, envases, bolsas simples | bolsas metalizadas de snack → `generales` |
| `vidrio` | ⚫ Gris | botellas y frascos verdes, marrones, transparentes | focos y fluorescentes → `peligrosos` |
| `organico` | 🟤 Marrón | cáscaras, restos de comida, hojas | |
| `peligrosos` | 🔴 Rojo | pilas, baterías, focos, jeringas, envases de químicos | |
| `generales` | ⚫ Negro | bolsas de chisitos (multicapa), servilletas sucias, tecnopor sucio, envolturas mixtas | |
| `sin_residuo` | — | fondo vacío, solo la mano | evita que el modelo "adivine" cuando no hay nada |

Para que el modelo **no aprenda "blanco = plástico"**: toma fotos de varios colores en cada clase (botellas verdes, azules y transparentes; latas de distintas marcas). Además, el entrenamiento varía el tono de las imágenes (`hsv_h`).

**Casos dudosos (reglas de etiquetado ya decididas):**
- **Regla general:** se clasifica el **material principal del objeto entero**; la tapa no cuenta (una botella PET con tapa metálica sigue siendo `plasticos`).
- **Envases Tetra Pak / multicapa de cartón** (cajitas de leche o jugo, p. ej. `obj128_caja_tetra_bonle`) → `papel_carton` (aprovechable), porque son ~75 % fibra de cartón (el resto, polietileno y aluminio). Se asume **envase vacío y enjuagado**; con restos de leche o jugo, el material sigue siendo el mismo pero en la práctica no se recicla. Decisión del 2026-10-26, también en `EXPERIMENTOS.md` ("Pendientes", cerrado).
- **Empaques flexibles** (bolsas de galleta, snack o fideos, sachets, doypacks, envolturas de golosinas), **metalizados o no** → `generales`: no se separan sus capas.
- **Envases rígidos de plástico** (botellas, frascos, potes, tapers) de **cualquier color** → `plasticos`.
- **Vasos y potes de cartón con recubrimiento para comida** (Ajinomen, vaso de café, tubo de Pringles) → `generales`.
- **Cartón limpio** → `papel_carton`; **cartón con grasa** o restos de comida → `generales`.
- **Frutas y tubérculos con cera** → `organico`.
- **Cinta plástica de señalización** aunque diga "PELIGRO" (`obj129_cinta_peligro_obras`) → `plasticos`: se clasifica el material, no el texto.

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
python scripts/inferencia_webcam.py --modelo modelos/etapa2_AAAAMMDD_HHMM.pt   # sin --modelo usa el modelo actual
```
Con `S` guardas un recorte en `capturas_revisar/`. Tú decides su clase y lo mueves a `dataset/juliaca/...` para reentrenar.

## 6. Usar el modelo en la laptop

> **Modelo actual:** `modelos/etapa2_ext_full.pt` (e2_ext_full), es el `--modelo` por defecto de `inferencia_webcam.py` e `inferencia_deteccion.py`. Se eligió por el máximo F1 macro en `val_camara` (0.5183), así que ese número es optimista (ver `EXPERIMENTOS.md`, "Modelo actual"). La medida final será `dataset/juliaca/test`, una sola vez; aún **no** se ha evaluado.

```bash
git pull
.venv\Scripts\activate
python scripts/verificar_entorno.py --comparar entorno_referencia.json      # debe decir "Todo coincide."
python scripts/inferencia_webcam.py --imgsz 224                             # cámara auto, --modelo modelos/etapa2_ext_full.pt
```

- Si `.venv` no existe en la laptop, créalo como en la sección 4, paso 4: `py -3.13 -m venv .venv`, actívalo y `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu130`.
- `--camara` es `auto` por defecto (en los 3 scripts de cámara): prueba los índices 0-5, y si Windows ve una MX Brio elige la que acepta mayor resolución (la Brio llega a 4K); si no hay Brio, la integrada (el primer índice que funciona). Imprime cuál eligió. OpenCV no da el nombre de cada índice en Windows, por eso el nombre solo dice si la Brio está y la resolución dice cuál es. `--camara N` fuerza el índice N. `--imgsz 224` es el tamaño con el que se entrenó. Opcionales: `--umbral` (por defecto `0.6`; por debajo muestra "NO SEGURO"), `--suavizado` (por defecto `8` frames promediados) y `--recuadro` (lado del recuadro como fracción del alto del frame; por defecto `0.75`, admite p. ej. `0.9` o `1.0`). Con `1.0` el recuadro es el cuadrado central de alto completo, que es exactamente lo que vería el clasificador si le pasaras el frame entero (Ultralytics redimensiona y recorta al centro).
- Ventana: se puede maximizar o redimensionar y el video ocupa toda la ventana. Antes quedaba una franja gris porque `cv2.imshow` sin `namedWindow` crea una ventana `WINDOW_AUTOSIZE`, que dibuja el frame a su tamaño nativo; ahora se crea con `WINDOW_NORMAL | WINDOW_FREERATIO` (`scripts/comun.py`, `crear_ventana`).
- Teclas: **Q** sale (o cerrar la ventana); **F** alterna pantalla completa; **S** guarda el recorte actual en `capturas_revisar/` (úsalo cuando el modelo se equivoque). Esas fotos no se clasifican solas: tú decides su clase y las pasas a `dataset/juliaca/<split>/<clase>/<objeto>/`.

### 6.1 Modo detección + clasificación (dos etapas, todo el frame)

`scripts/inferencia_deteccion.py` busca objetos en **toda** la imagen (varios a la vez, priorizando los cercanos) y clasifica cada uno con tu modelo:

1. **Detector de vocabulario abierto** (no se entrena): YOLOE con prompts de texto en inglés. Busca los de residuo (`--prompts`) y también los **negativos** (`--ignorar`: persona, cara, mano, ropa, muebles, pared...). Solo localiza las cajas; la clase la decide siempre el clasificador.
2. **Filtros**, en este orden (con `--bench` se imprime cuántas cajas descarta cada uno):
   - `ignorada`: cajas de los prompts negativos; no se clasifican ni se dibujan.
   - `duplicada`: dos prompts de residuo sobre el mismo objeto (p. ej. `bottle` y `cup`, IoU > 0.7); queda la de mayor confianza.
   - `roi`: el centro de la caja debe caer en la zona central `--roi` (por defecto `1.0` = todo el frame; si es menor se dibuja la zona).
   - `area`: área entre `--min-area` y `--max-area` (por defecto 0.01 y 0.85).
   - `ropa_mueble`: se descarta la caja de residuo que coincide casi entera (IoU ≥ `--iou-ignorar`) con una caja de ropa o mueble (clothing, jacket, sofa, chair, shelf, bed, furniture...) de confianza igual o mayor, es decir, cuando el detector cree que es la prenda o el mueble. Persona, cara y mano **no** anulan cajas: un objeto en la mano sí se muestra. Pared, puerta, ventana y cortina tampoco, porque cubren todo el fondo y anularían a un objeto en primer plano (pasaba en `dataset/crudo/val`: una botella cercana contra una caja `wall` y unos vasos detectados como `curtain`).
   - `lejana`: **foco en lo cercano**, solo quedan las cajas con área ≥ `--foco-relativo` × la caja de residuo más grande del frame.
   - `borrosa`: **nitidez**, varianza del Laplaciano del recorte (a 128×128) ≥ `--nitidez-rel` × la del recorte más nítido del frame. Desactívalo con `--no-nitidez`.
   - `exceso`: se ordenan por área (mayor primero) y se quedan `--max-objetos`.
3. Cada caja se recorta **cuadrada con ~10 % de margen, dentro del frame y sin relleno** (si no cabe, se recorta a los bordes) y la clasifica `--modelo` a `imgsz 224` en fp32, con el mismo preprocesado que `evaluar.py` (ver "Diagnóstico" abajo).
4. **Persistencia**: cada caja se sigue entre frames por IoU (≥ `--iou`). Se muestra solo si apareció en al menos `--persistencia` de los últimos `--historial` frames (3 de 5), y su clase es el **promedio de las probabilidades** del clasificador en esos frames. Se ignoran las cajas cuyo promedio da `sin_residuo`.

**Pantalla:**
- Cada objeto: recuadro del color de su tacho (NTP 900.058: azul papel_carton, amarillo metales, blanco plasticos, gris vidrio, marrón organico, rojo peligrosos, negro generales) con la etiqueta `AZUL - Papel y carton 87%`. Si la confianza promedio es menor a `--umbral`: morado con `NO SEGURO 45%` (morado porque el gris es el tacho de vidrio).
- **Barra superior**: el tacho del objeto principal (el de mayor área) con fondo de su color y texto grande, p. ej. `TACHO AZUL - Papel y carton (87%)`; `NO SEGURO (45%)` si su confianza es baja; `Sin residuo` si no hay objetos. Si los objetos van a tachos distintos, debajo aparece una fila de **chips** de colores, uno por tacho con su mayor confianza.
- Los textos van sin tildes (`carton`, `Organicos`) porque la fuente de OpenCV solo dibuja ASCII.
- **Respaldo (vista completa)**: si los filtros no dejan ninguna caja (el detector no ve el objeto, p. ej. una caja que llena el encuadre), se clasifica el **frame completo** por el mismo camino que `evaluar.py`, promediando los últimos `--historial` frames. Si sale `sin_residuo` → `Sin residuo`; otra clase con confianza ≥ `--umbral` → `TACHO ... (87%) (vista completa)`; si no → `NO SEGURO (45%) (vista completa)`. Abajo a la izquierda aparece `vista completa`. Con **S** se guarda el frame entero (`pred-<clase>_completo_...jpg`). `--sin-respaldo` lo desactiva.
- `--debug`: dibuja además, en gris fino, las cajas descartadas con el motivo y el prompt: `ignorada [wall]`, `duplicada [cup]`, `area [cardboard box]`, `lejana`, `borrosa`, `exceso`, `persistencia 1/3` o `sin_residuo 80%`.

Teclas: **Q** sale, **F** pantalla completa, **S** guarda los recortes mostrados en `capturas_revisar/` (nombre `pred-<clase>_...jpg`, o `pred-NOSEGURO_...`; la clase la decides tú). La ventana se maximiza sin franja gris, igual que en `inferencia_webcam.py`.

```bash
python scripts/inferencia_deteccion.py                                                 # cámara auto (MX Brio si está), modelo actual
python scripts/inferencia_deteccion.py --modelo modelos/etapa1_ganador.pt  # con el modelo de etapa 1
python scripts/inferencia_deteccion.py --fuente video.mp4                             # probar con un video
python scripts/inferencia_deteccion.py --fuente foto.jpg --bench 100                  # FPS y descartes por filtro, sin ventana
python scripts/inferencia_deteccion.py --debug                           # además, cajas descartadas en gris con su motivo
```

| Argumento | Defecto | Qué hace |
|---|---|---|
| `--modelo` | `modelos/etapa2_ext_full.pt` | clasificador (modelo actual, ver `EXPERIMENTOS.md`); `modelos/etapa1_ganador.pt` y `modelos/etapa2_v2.pt` siguen disponibles |
| `--detector` | `yoloe-26s-seg.pt` | detector (decisión del usuario tras la comparación de abajo; antes `yoloe-26n-seg.pt`); se descarga solo a `modelos/detectores/` (no se sube a git) |
| `--camara` / `--fuente` | `auto` / — | índice de webcam (auto: MX Brio si está, si no la integrada), o video/imagen para pruebas |
| `--prompts` | bottle, can, plastic bag, snack wrapper, battery, phone, charger, cable, cup, glass jar, fruit peel, food waste, tube, cardboard box, juice box, candy wrapper, chocolate bar, sachet, printed paper document, sheet of paper, newspaper, notebook, paper bag, egg carton, used tissue | qué residuos busca el detector. Sin los genéricos `box`, `paper` y `container` (marcaban mesa, hojas y fondo); si los necesitas, pásalos con `--prompts "...,box,paper,container"` |
| `--max-area` | `0.85` | área máxima de la caja (fracción del frame); antes `0.6` |
| `--debug` | apagado | cajas descartadas en gris con su motivo |
| `--sin-respaldo` | apagado | no clasifica el frame completo cuando no hay cajas |
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
| `--imgsz-det` | `640` | tamaño del detector; bajarlo (p. ej. `480`, `416`) acelera |
Detectores que soporta Ultralytics 8.4.184 (`ultralytics/utils/downloads.py`):
- YOLOE con texto: `yoloe-26{n,s,m,l,x}-seg.pt`, `yoloe-11{s,m,l}-seg.pt`, `yoloe-v8{s,m,l}-seg.pt` (el codificador de texto `mobileclip2_b.ts` se descarga la primera vez a la raíz del repo; está en `.gitignore`).
- YOLOE prompt-free (vocabulario interno, ignora `--prompts`): los mismos con sufijo `-seg-pf.pt`.
- YOLO-World: `yolov8{s,m,l,x}-world.pt` y `yolov8{s,m,l,x}-worldv2.pt` (necesitan además el paquete `clip` de OpenAI).
- Detector por defecto: **`yoloe-26s-seg.pt`** (detecta más objetos en val_camara, ver "Comparación de detectores"). Si en la RTX 3050 4 GB va lento, `--detector yoloe-26n-seg.pt` es el más liviano.

FPS medidos en la laptop (RTX 3050 Laptop 4 GB, `--bench`, sin ventana ni captura de cámara, imágenes de `dataset/crudo/val`): con `yoloe-26n-seg` el detector tarda ~22–26 ms. En total da ~42–45 FPS con 0 objetos, ~27 FPS con 1 objeto y ~23 FPS con 5 objetos. Con `yoloe-26s-seg` y 2 objetos da ~26 FPS. En vivo será algo menos por la captura y la ventana. Si baja de ~8 FPS: usa `--imgsz-det 480` o `416` y/o `--max-objetos 3`.

Con los filtros y la persistencia (valores por defecto actuales, misma laptop, `--bench 200`, imágenes de `dataset/crudo/val`): 1 caja mostrada ~22.5–23.8 ms (~42–44 FPS); 3 cajas mostradas ~22.6–24.1 ms (~41–44 FPS, el tope de `--max-objetos`); 0 cajas ~13.8–15 ms (~66–72 FPS; en las dos primeras corridas de la sesión, con la GPU recién activada, el detector tardó 36–39 ms y el total dio ~26–28 FPS). Una imagen con una sola caja de área 0.68 quedaba en 0 cajas por `--max-area 0.6` (ahora el defecto es 0.85). Esas medidas son de antes de pasar el clasificador a fp32 y de los prompts nuevos; en la PC (RTX 3060, `--bench 200`, 1 caja) ahora da det 19.5 ms + clf 10.2 ms = 29.7 ms (33.7 FPS). Falta medirlo en la laptop.

**Diagnóstico** (`python scripts/diagnostico_inferencia.py`): pasa cada foto de `dataset/val_camara` por `evaluar.py` y por la función `clasificar()` de este script (la imagen completa como si fuera una caja, leída con `cv2.imread`), y guarda 10 recortes reales del detector en `revision/recortes_debug/` (no se sube a git). Con `etapa2_ext_full.pt`:

| Camino | top-1 | F1 macro |
|---|---|---|
| `evaluar.py` (ruta) | 0.5529 | 0.5183 |
| antes: recorte con relleno gris + fp16 | 0.4437 | 0.3189 |
| sin relleno, fp16 | 0.5495 | 0.5113 |
| ahora: sin relleno, fp32 | 0.5529 | 0.5183 (0 de 293 fotos distintas) |

El orden de canales y la lectura no tenían problema: `cv2.imread` y el lector de Ultralytics dan los mismos píxeles en las 293 fotos, y Ultralytics recibe BGR en ambos caminos (Resize del lado corto + CenterCrop, BGR→RGB, /255). La diferencia era el relleno gris de `recorte_cuadrado` (cambiaba 142 de 293 predicciones) y fp16 (2 de 293).

**Comparación de detectores** (`python scripts/comparar_detectores.py`, 293 fotos de `dataset/val_camara`, filtros por defecto, una pasada por foto, clasificador `etapa2_ext_full.pt`). val_camara no tiene cajas anotadas, así que "la caja cubre el objeto" se aproxima: `caja` = queda al menos una caja tras los filtros; `centro` = la caja principal contiene el centro de la foto; `clase_ok` = su recorte se clasifica bien; `final_ok` = clase final de un frame (caja, o frame completo si no hay caja). Detalle en `reportes/comparar_detectores.csv`; hojas de contacto con la caja dibujada en `revision/comparar_detectores/` (no se suben).

| Detector | caja | centro | clase_ok | final_ok |
|---|---|---|---|---|
| `yoloe-26n-seg` | 97 (33.1 %) | 91 (31.1 %) | 48 (16.4 %) | 159 (54.3 %) |
| `yoloe-26s-seg` (defecto) | 152 (51.9 %) | 147 (50.2 %) | 84 (28.7 %) | 163 (55.6 %) |

En la primera hoja de contacto, 26n además encuadra a veces solo una parte del envoltorio (y aun así cuenta en `centro`). FPS en la PC (RTX 3060, `--bench 200`): 26n 30.0 FPS con 1 caja / 29.9 sin cajas (respaldo); 26s 29.4 / 29.2. **Falta medir en la laptop (RTX 3050)**, donde la diferencia será mayor. Defecto elegido por el usuario: `yoloe-26s-seg.pt`.

Detección global (`--roi 1.0`, `--ignorar`, foco y nitidez, `--max-objetos 4`, con `--modelo modelos/etapa1_ganador.pt`, el defecto cuando se midió; misma laptop, `--bench 200`). En `clf` entran también los filtros, la nitidez y la persistencia:

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

Fuentes añadidas el 2026-10-26 con `python scripts/importar_publicos.py` (`--dry-run` primero). Cada una fue aprobada por el usuario. Mapeo completo, licencias y citas en **`datos/fuentes.md`**:

| Fuente (prioridad en el tope) | Licencia | Usa | Descarta |
|---|---|---|---|
| 1. RealWaste (UCI) | CC BY 4.0 | 7 clases; `Metal` con **revisión manual** (aerosoles → `peligrosos`) | Textile, Miscellaneous Trash |
| 2. TACO extra (prefijo `tacox_`) | CC BY 4.0 / ODbL | 41 categorías, recortes ≥ 96 px | Cigarette, Unlabeled litter, Meal carton... |
| 3. TrashNet (GitHub) | MIT | glass, paper, cardboard, plastic, metal | trash |

**No aprobados:** Garbage-12 (las imágenes no tienen licencia) y **Household Waste** (Kaggle, alistairking). Household se descartó tras revisar una muestra: su parte `real_world` son fotos de stock, gráficos con texto, ilustraciones e infografías, con duplicados entre clases (detalle en `datos/fuentes.md`).

**Revisión de RealWaste `Metal`:** el script genera mosaicos numerados de 48 en `reportes/revision_realwaste_metal/` (con `indice.csv`) y la plantilla `reportes/revision_realwaste_metal.txt`, que nunca sobrescribe. En la plantilla anotas los números de los aerosoles, más otras correcciones como `número clase` o `número descartar`, y cambias `terminado: no` por `terminado: si`. Hasta entonces, la importación real se niega. Si las candidatas de `Metal` cambian y el índice ya no corresponde, el script se detiene en vez de aplicar mal los números.

**Selección curada para la etapa 2** (`python scripts/seleccionar_curados.py`): propone candidatas de papel_carton y generales parecidas al uso real, tomadas solo de `publico/train` y `externo/train`. Genera mosaicos numerados en `reportes/seleccion_curada/` y la plantilla `reportes/seleccion_curada.txt`, donde anotas los descartes. Después, `--final N` escribe `seleccion_final.csv` y `entrenar.py --seleccion <csv>` la suma al train de la etapa 2 sin mover archivos. Resultado del experimento en `EXPERIMENTOS.md` (`e2_ext_full_sel100`: no mejora).

Tope: **1.200 imágenes nuevas por clase**, llenado por prioridad (muestra con semilla 42 dentro de cada fuente y clase). Todo va solo a `externo/train`. Deduplicación con dHash ≤ 4 contra publico (train/val/test), juliaca y el externo previo, y dentro y entre las fuentes; si dos se parecen, se queda la de mayor prioridad. Informes: `reportes/externo_publicos_dedup.csv` y `reportes/externo_publicos_conteos.csv`. Mosaicos de revisión (24 por fuente y clase) en `reportes/mosaicos_externos/`, que no se suben a git. El script solo reemplaza sus propios archivos y filas (prefijos `realwaste_`, `tacox_`, `trashnet_`); no toca lo de `importar_taco.py`. Sin la revisión de RealWaste Metal terminada, la importación real se niega.

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

`python scripts/auditar.py` verifica que ninguna foto de `val_camara` esté (idéntica o casi igual, dHash <= 4) en juliaca, publico o externo; si está en algún `train` es ERROR, porque `val_camara` nunca se entrena. Informe en `reportes/val_camara_dedup.csv`; los falsos positivos revisados a ojo van en `reportes/val_camara_falsos_positivos.csv`.

En la misma carpeta del Grupo 3 están también los `.zip` del paquete (`val_camara_paquete-CLASIFICADAS.zip` y `val_camara_paquete-IMAGENES-CRUDAS.zip`).
