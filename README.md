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
modelos/etapa2_*.pt  ──► se evalúan AMBOS en dataset/juliaca/test → cuánto mejoró con datos locales
```

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

1. Actualiza el driver NVIDIA. No necesitas el CUDA Toolkit: PyTorch trae su runtime.
2. Usa la **misma versión de Python** en los dos equipos (recomendado 3.13).
3. En el PRIMER equipo donde instales (PC o laptop):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   # comando de https://pytorch.org/get-started/locally/ (Stable · Windows · Pip · CUDA 12.x), p. ej.:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12X
   pip install -r requirements.txt
   pip freeze > requirements-lock.txt
   python scripts/verificar_entorno.py --guardar entorno_referencia.json
   git add requirements-lock.txt entorno_referencia.json && git commit -m "Fijar versiones" && git push
   ```
4. En el OTRO equipo: `git pull`, crea el venv y
   ```bash
   pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu12X
   python scripts/verificar_entorno.py --comparar entorno_referencia.json
   ```

## 5. Flujo completo

```bash
# Datos públicos → dataset/crudo/<fuente>/<clase>/   (ver sección 6)
# Revisar class_mapping.csv y poner aprobado=SI a mano
python scripts/preparar_publico.py --max-por-clase 1500
# Tus fotos → dataset/juliaca/<split>/<clase>/<objeto>/
python scripts/auditar.py                                  # debe terminar "Sin errores"
python scripts/entrenar.py --etapa 1
python scripts/entrenar.py --etapa 2 --pesos modelos/etapa1_AAAAMMDD_HHMM.pt
python scripts/entrenar.py --etapa 2 --pesos yolo26s-cls.pt --replay 0   # opcional: solo Juliaca, para comparar
python scripts/evaluar.py --pesos modelos/etapa1_*.pt modelos/etapa2_*.pt
```

Cada época imprime una línea:
`[EPOCA  12/60]  41.3s | loss_train=... loss_val=... | top1_val=... top5_val=... | lr=... | GPU_pico=...GB`

Uso en vivo (laptop): en la PC haz `git add modelos/ reportes/ && git commit -m "Modelo etapa 2" && git push`; en la laptop `git pull` y ejecuta
```bash
python scripts/inferencia_webcam.py --modelo modelos/etapa2_AAAAMMDD_HHMM.pt
```
Con `S` guardas un recorte en `capturas_revisar/`. Tú decides su clase y lo mueves a `dataset/juliaca/...` para reentrenar.

## 6. Datos públicos

| Fuente (carpeta en `crudo/`) | Contenido | Nota |
|---|---|---|
| `garbage12` | Kaggle *Garbage Classification (12 classes)*; mejor la versión **ENHANCED** de huberthamelin (etiquetas corregidas a mano) | ya está en `class_mapping.csv` |
| `roboflow_<nombre>` | Roboflow Universe, exportado como **Folder Structure** | al correr `preparar_publico.py` sus clases se agregan al CSV con `aprobado=NO` |

Antes de aprobar `garbage12/plastic`, abre la carpeta y **saca a mano** las bolsas de snack metalizadas (van a `generales`). Revisa la licencia de cada dataset y cítalo en tu informe.

## 7. Reglas de integridad

- No se generan ni sintetizan imágenes. Solo fotos reales.
- La clase de cada foto la decides tú (carpetas). Ningún script mueve fotos entre clases o splits de `dataset/juliaca`.
- `dataset/juliaca/test` no se usa para entrenar ni para elegir hiperparámetros.
- Las métricas del informe se copian de `reportes/*.json` y `runs/*/results.csv`, tal como salieron.

Las fotos (`dataset/`) no se suben a git (ver `.gitignore`); para pasarlas entre equipos usa un USB o Google Drive. Los modelos finales de `modelos/` (con su `.json` de versiones) y `reportes/` sí se suben.
