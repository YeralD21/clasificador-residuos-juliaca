# Fuentes de datos públicos y externos

Para el informe: de dónde sale cada imagen que no es una foto propia de Juliaca, con su licencia, cita y mapeo
de clases. El registro por archivo (regla 1 de CLAUDE.md) está en `datos_externos.csv`; los conteos reales, en
`reportes/externo_publicos_conteos.csv` y `reportes/externo_dedup.csv`.

## Cómo se usan

| Conjunto | Carpeta | Splits | Para qué |
|---|---|---|---|
| Fotos propias de Juliaca | `dataset/juliaca/<split>/<clase>/<objeto>/` | train / val / test, manual por objeto | etapa 2; `val` elige modelos; `test` solo una vez al final |
| Público base | `dataset/publico/` (desde `dataset/crudo/companero`) | train / val / test (80/10/10, `preparar_publico.py`) | etapa 1; `publico/val` elige modelos de la etapa 1 |
| Externos | `dataset/externo/train/<clase>/` | **solo train** (regla 1c) | se suman a train con `--externo`; nunca a val ni test |

Todo lo externo pasa antes por un dHash ≤ 4 contra **todos** los splits de `publico` y `juliaca` (regla 1d), y
lo que se parece se elimina.

## Público base

### companero (`dataset/crudo/companero`)
- **Qué es:** dataset recibido de un compañero, organizado por color de tacho (Amarillo, Azul, Blanco, Gris,
  Marrón, Negro, Rojo). En su mayoría son fotos de stock y de búsquedas web (varios objetos por foto, fondos de catálogo,
  marcas de agua).
- **Licencia:** no documentada. Para el informe hay que pedir al compañero el origen de las imágenes.
- **Mapeo** (`class_mapping.csv`, aprobado por el usuario): Amarillo → metales, Azul → papel_carton,
  Blanco → plasticos, Gris → vidrio, Marrón → organico, Negro → generales, Rojo → peligrosos.
  Exclusiones manuales en `excluir_imagenes.txt`.

## Externos (solo `dataset/externo/train`)

Prioridad dentro del tope de **1.200 imágenes nuevas por clase** (decisión del usuario, 2026-10-26):
RealWaste > TACO extra > TrashNet. Dentro de cada fuente y clase se toma una muestra aleatoria con semilla 42
(`scripts/importar_publicos.py`). Lo que no figura en un mapeo se descarta; no se fuerza a una clase.

### TACO, primera importación (`scripts/importar_taco.py`)
- **Licencia:** anotaciones CC BY 4.0; imágenes CC BY 4.0 (las que no declaran licencia, por defecto según
  tacodataset.org) u ODbL 1.0 © OpenLitterMap & Contributors. Se excluyen las marcadas "CC" sin especificar.
- **Cita:** Proença, P. F., & Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection*.
  arXiv:2003.06975. http://tacodataset.org
- **Mapeo:** Crisp packet y Other plastic wrapper → generales (con revisión manual por recorte,
  `reportes/externo_taco_revision.csv`); Aluminium foil → metales. Archivos `taco_*.jpg`.

### TACO extra (`scripts/importar_publicos.py`, prefijo `tacox_`, prioridad 2)
- **Licencia y cita:** las mismas que arriba. Commit de anotaciones `70479ad`. Recortes cuadrados con 10 % de margen,
  mínimo 96 px (igual que la primera importación).
- **Mapeo:**
  - papel_carton: Drink carton (Tetra Pak, regla de etiquetado del README §1), Corrugated carton, Other carton,
    Egg carton, Toilet tube, Paper bag, Normal paper, Magazine paper, Wrapping paper
  - plasticos: Clear plastic bottle, Other plastic bottle, Disposable plastic cup, Plastic lid, Plastic bottle cap,
    Single-use carrier bag, Plastic film, Disposable food container, Plastic utensils, Plastic straw, Garbage bag,
    Spread tub, Tupperware, Polypropylene bag, Six pack rings
  - metales: Drink can, Food Can, Metal bottle cap, Metal lid, Pop tab, Scrap metal
  - vidrio: Glass bottle, Glass jar, Glass cup, Broken glass
  - generales: Tissues, Styrofoam piece, Foam cup, Foam food container
  - peligrosos: Aerosol, Battery
  - organico: Food waste
  - **descartadas:** Cigarette, Unlabeled litter, Other plastic, Meal carton y Pizza box (cartón con grasa),
    Paper cup, Shoe, Rope & strings, Plastic glooves, Squeezable tube, Aluminium blister pack, Carded blister pack,
    Other plastic cup, Other plastic container, Paper straw (además de las de la primera importación)
- **Parecido al uso real:** medio. Residuo real con fondos variados, pero tirado en el suelo y en recortes chicos.

### RealWaste (prefijo `realwaste_`, prioridad 1)
- **Origen:** UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/908/realwaste. 4.752 imágenes
  de 524×524, 9 clases, tomadas en la recepción de un relleno sanitario.
- **Licencia:** CC BY 4.0.
- **Cita:** Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5SS4G
- **Mapeo:** Cardboard, Paper → papel_carton; Glass → vidrio; Metal → metales; Plastic → plasticos;
  Food Organics, Vegetation → organico. **Descartadas:** Textile Trash, Miscellaneous Trash (decisión del usuario).
- **Revisión manual de Metal:** RealWaste mete los aerosoles (desodorantes, insecticidas, pintura) en `Metal`,
  y nuestra regla los manda a `peligrosos`. El usuario revisa las 672 imágenes de Metal que quedan tras deduplicar,
  en mosaicos numerados (`reportes/revision_realwaste_metal/`, con su `indice.csv`), y anota en
  `reportes/revision_realwaste_metal.txt`: los aerosoles pasan a peligrosos, y otras correcciones van con su clase o
  como "descartar". En `datos_externos.csv` figuran con `categoria_origen` = `Metal (revisión del usuario: <clase>)`.
- **Parecido al uso real:** el más alto de las fuentes externas. Residuo real (aplastado, sucio), un objeto por foto,
  sobre piso de concreto.

### TrashNet (prefijo `trashnet_`, prioridad 3)
- **Origen:** https://github.com/garythung/trashnet, `data/dataset-resized.zip`. 2.527 imágenes de 512×384,
  6 clases; un objeto sobre cartulina blanca (estudio).
- **Licencia:** MIT (licencia del repositorio; el dataset no tiene una propia y los autores piden citarlo).
- **Cita:** los autores piden citar el repositorio: Thung, G., & Yang, M. (2016). *TrashNet* [repositorio].
  https://github.com/garythung/trashnet (informe asociado: *Classification of Trash for Recyclability Status*,
  CS229, Stanford University).
- **Mapeo:** glass → vidrio; paper, cardboard → papel_carton; plastic → plasticos; metal → metales.
  **Descartada:** trash (mezcla).
- **Prioridad:** la última dentro del tope (fotos de estudio, lo menos parecido al uso real).

## Resultado de la importación (2026-10-26)

`python scripts/importar_publicos.py`, con la revisión de RealWaste Metal aplicada (68 aerosoles → peligrosos,
30 descartadas). Números copiados de `reportes/externo_publicos_conteos.csv`. Candidatas = imágenes del mapeo;
tras dedup = tras quitar las de dHash ≤ 4 contra publico, juliaca, val_camara, el externo previo y entre fuentes;
elegidas = tras el tope de 1.200 nuevas por clase.

| clase | fuente | candidatas | tras dedup | elegidas |
|---|---|---|---|---|
| generales | taco_extra | 125 | 125 | 125 |
| metales | realwaste | 722 | 574 | 574 |
| metales | taco_extra | 304 | 304 | 304 |
| metales | trashnet | 410 | 315 | 315 |
| organico | realwaste | 847 | 832 | 832 |
| organico | taco_extra | 5 | 4 | 4 |
| papel_carton | realwaste | 961 | 927 | 927 |
| papel_carton | taco_extra | 251 | 251 | 251 |
| papel_carton | trashnet | 997 | 791 | 22 |
| peligrosos | realwaste | 68 | 68 | 68 |
| peligrosos | taco_extra | 12 | 12 | 12 |
| plasticos | realwaste | 921 | 886 | 886 |
| plasticos | taco_extra | 1109 | 1103 | 314 |
| plasticos | trashnet | 482 | 395 | 0 |
| vidrio | realwaste | 420 | 402 | 402 |
| vidrio | taco_extra | 95 | 94 | 94 |
| vidrio | trashnet | 501 | 375 | 375 |
| **total** | | **8.230** | **7.458** | **5.505** |

RealWaste Metal: 790 originales → 672 tras dedup → 574 metales + 68 peligrosos + 30 descartadas por el usuario. La
clase se asigna después de deduplicar, así que los 68 aerosoles aparecen como candidatas de peligrosos y no de metales.

Eliminadas por dedup (742): casi iguales a publico 598, a val_camara 4, a juliaca 3; duplicados entre o dentro de
fuentes 137 (RealWaste 94, TrashNet 39, TACO 4). Las 598 de publico son sobre todo TrashNet: `companero` ya contenía
buena parte de TrashNet. Las de juliaca y val_camara eran falsos positivos a la vista (fotos distintas de objetos
distintos), pero se eliminaron por la regla.

`dataset/externo/train` final, contando lo de `importar_taco.py` (`python scripts/auditar.py`, 0 ERRORES):
generales 264, metales 1.233, organico 836, papel_carton 1.200, peligrosos 80, plasticos 1.200, vidrio 871,
sin_residuo 0.

## Evaluados y no aprobados
- **Recyclable and Household Waste Classification**, Kaggle `alistairking`
  (https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification; MIT según Kaggle).
  Se aprobó al principio solo su parte `real_world`, que según la descripción muestra "tacho, suelo, desorden". Se
  **descartó** el 2026-10-26, tras revisar una muestra de 2 imágenes por cada una de las 30 categorías y 6 pares
  de duplicados (decisión del usuario). Lo que hay en realidad:
  - fotos de stock y de producto sobre fondo blanco, el mismo dominio que `companero`;
  - gráficos con texto, anuncios y miniaturas ("5.0 TRILLION…", "27 Extraordinary Fruit & Vegetable Peel Uses",
    "CUSTOM STYROFOAM CUPS");
  - ilustraciones vectoriales e infografías (plantilla de vaso de papel, tabla de tamaños de latas) y al menos un
    posible render 3D, lo que choca con la regla 1 (solo fotos reales);
  - 1.205 duplicados internos (misma imagen con otro recorte o rótulo), algunos en dos categorías a la vez
    (steel_food_cans / aluminum_food_cans 78, cardboard_packaging / cardboard_boxes 53).
  El zip descargado a mano quedó en `dataset/externo/_fuentes/household/` y no se usa.
- **Garbage Classification (12 classes)**, Kaggle `mostafaabla`: licencia "Database: ODbL, Contents: © Original
  Authors". Las imágenes son de búsquedas web y no tienen licencia propia, así que no cumple la regla 1b. Además es del mismo estilo que
  `companero`.
- **TrashBox** (Kumsetty et al., 2022): licencia no declarada; descarga solo para suscriptores de IEEE DataPort.
- **The Garbage Dataset** (arXiv 2602.10500): la CC BY 4.0 es la del artículo; no se publica licencia ni enlace
  del dataset.
