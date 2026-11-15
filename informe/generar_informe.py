"""Arma informe/INFORME_FINAL.docx con python-docx (portada, índice, títulos, tablas, figuras numeradas).

Requiere python-docx (no está en requirements-lock.txt; se usa un entorno aparte):
    py -3.13 -m venv <env> && <env>\\Scripts\\pip install python-docx pywin32
    <env>\\Scripts\\python.exe informe\\generar_informe.py            # .docx
    <env>\\Scripts\\python.exe informe\\generar_informe.py --pdf      # además actualiza índices y exporta PDF con Word

Antes: .venv\\Scripts\\python.exe informe\\generar_figuras.py (figuras e informe/datos/figuras_meta.json).
Todas las métricas se copian de reportes/*.json, reportes/*.csv y EXPERIMENTOS.md (regla 6).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

RAIZ = Path(__file__).resolve().parents[1]
INF = RAIZ / "informe"
FIG = INF / "figuras"
META = json.load(open(INF / "datos" / "figuras_meta.json", encoding="utf-8"))
REP = RAIZ / "reportes"
SALIDA = INF / "INFORME_FINAL.docx"

AUTOR = "Fabrizio Yerald Alfonso Sanchez Saravia"
CURSO = "Machine Learning"
UNIV = "Universidad Peruana Unión (UPeU) – Filial Juliaca"
FECHA = "14 de noviembre de 2026"
TITULO = "Clasificador de residuos sólidos por material según la NTP 900.058:2019"
SUBTITULO = "Fine-tuning en dos etapas de YOLO26-cls con datos públicos y fotos propias de Juliaca"

AZUL = RGBColor(0x1C, 0x4E, 0x8A)
GRIS = RGBColor(0x52, 0x51, 0x4E)
FUENTE = "Calibri"
CLASES = ["generales", "metales", "organico", "papel_carton", "peligrosos", "plasticos", "sin_residuo", "vidrio"]
TACHO = {"generales": "Negro", "metales": "Amarillo", "organico": "Marrón", "papel_carton": "Azul",
         "peligrosos": "Rojo", "plasticos": "Blanco", "sin_residuo": "—", "vidrio": "Gris"}


def rj(nombre):
    return json.load(open(REP / nombre, encoding="utf-8"))


# ------------------------------------------------------------------ utilidades XML
def campo(parrafo, instr, texto="1"):
    """Inserta un campo de Word (SEQ, TOC, PAGE...)."""
    r = parrafo.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = texto
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2); r._r.append(t); r._r.append(f3)
    return r


def sombrear(celda, hex_color):
    tcPr = celda._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def bordes_tabla(tabla, color="BFBFBF"):
    tblPr = tabla._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), color)
        b.append(e)
    tblPr.append(b)


def repetir_encabezado(fila):
    trPr = fila._tr.get_or_add_trPr()
    e = OxmlElement("w:tblHeader"); e.set(qn("w:val"), "true")
    trPr.append(e)


def no_partir(fila):
    trPr = fila._tr.get_or_add_trPr()
    e = OxmlElement("w:cantSplit"); e.set(qn("w:val"), "true")
    trPr.append(e)


# ------------------------------------------------------------------ documento
class Informe:
    def __init__(self):
        self.d = Document()
        self._estilos()
        s = self.d.sections[0]
        s.page_width, s.page_height = Cm(21), Cm(29.7)
        s.left_margin = s.right_margin = Cm(2.5)
        s.top_margin, s.bottom_margin = Cm(2.5), Cm(2.2)
        self.ancho = Cm(16)

    def _estilos(self):
        st = self.d.styles
        n = st["Normal"]
        n.font.name = FUENTE; n.font.size = Pt(11)
        n.element.rPr.rFonts.set(qn("w:eastAsia"), FUENTE)
        n.paragraph_format.space_after = Pt(6)
        n.paragraph_format.line_spacing = 1.15
        for nivel, tam, antes in ((1, 16, 18), (2, 13, 14), (3, 11.5, 10)):
            h = st[f"Heading {nivel}"]
            h.font.name = FUENTE; h.font.size = Pt(tam); h.font.bold = True
            h.font.color.rgb = AZUL if nivel < 3 else RGBColor(0x26, 0x26, 0x26)
            rf = h.element.rPr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts"); h.element.rPr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(a), FUENTE)
            for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                if rf.get(qn(a)) is not None:
                    del rf.attrib[qn(a)]
            h.paragraph_format.space_before = Pt(antes); h.paragraph_format.space_after = Pt(6)
            h.paragraph_format.keep_with_next = True
        c = st["Caption"]
        c.font.name = FUENTE; c.font.size = Pt(9); c.font.italic = False; c.font.bold = False
        c.font.color.rgb = GRIS
        c.paragraph_format.space_before = Pt(3); c.paragraph_format.space_after = Pt(12)
        from docx.enum.style import WD_STYLE_TYPE
        pd_ = st.add_style("Pie detalle", WD_STYLE_TYPE.PARAGRAPH)
        pd_.base_style = st["Normal"]
        pd_.font.size = Pt(8.5); pd_.font.color.rgb = GRIS
        pd_.paragraph_format.space_after = Pt(12); pd_.paragraph_format.line_spacing = 1.0
        for lst in ("List Bullet", "List Number"):
            st[lst].font.name = FUENTE; st[lst].font.size = Pt(11)
            st[lst].paragraph_format.space_after = Pt(3)

    # -- bloques
    def h(self, texto, nivel=1):
        return self.d.add_heading(texto, nivel)

    def p(self, texto="", negrita_inicial=None, cursiva=False, alinear=WD_ALIGN_PARAGRAPH.JUSTIFY, estilo=None):
        par = self.d.add_paragraph(style=estilo)
        par.alignment = alinear
        if negrita_inicial:
            par.add_run(negrita_inicial).bold = True
        self._runs(par, texto, cursiva)
        return par

    def _runs(self, par, texto, cursiva=False):
        """Admite **negrita** y `código` en línea."""
        import re
        for trozo in re.split(r"(\*\*[^*]+\*\*|`[^`]+`|(?<![\w*])\*[^*\s][^*]*\*(?![\w*]))", texto):
            if not trozo:
                continue
            if trozo.startswith("**"):
                r = par.add_run(trozo[2:-2]); r.bold = True
            elif trozo.startswith("*") and trozo.endswith("*") and len(trozo) > 2:
                r = par.add_run(trozo[1:-1]); r.italic = True
            elif trozo.startswith("`"):
                r = par.add_run(trozo[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
            else:
                r = par.add_run(trozo)
            if cursiva:
                r.italic = True

    def viñetas(self, items, numerada=False):
        for it in items:
            par = self.d.add_paragraph(style="List Number" if numerada else "List Bullet")
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT
            self._runs(par, it)

    @staticmethod
    def _partir(texto):
        """Título (primera oración, va al índice) y detalle (resto)."""
        import re
        sin_cod = re.sub(r"`[^`]*`", lambda m: "x" * len(m.group()), texto)
        m = re.search(r"\.(\*\*)?\s", sin_cod)
        if not m:
            return texto, ""
        corte = m.end()
        return texto[:corte].rstrip(), texto[corte:].strip()

    def pie(self, tipo, texto):
        titulo, detalle = self._partir(texto)
        centro = WD_ALIGN_PARAGRAPH.CENTER if tipo == "Figura" else WD_ALIGN_PARAGRAPH.LEFT
        par = self.d.add_paragraph(style="Caption")
        par.alignment = centro
        r = par.add_run(f"{tipo} "); r.bold = True; r.font.color.rgb = AZUL
        c = campo(par, "SEQ " + tipo + r" \* ARABIC", "1"); c.bold = True; c.font.color.rgb = AZUL
        r = par.add_run(". "); r.bold = True; r.font.color.rgb = AZUL
        self._runs(par, titulo)
        ultimo = par
        if detalle:
            par.paragraph_format.space_after = Pt(0)
            par.paragraph_format.keep_with_next = True
            dp = self.d.add_paragraph(style="Pie detalle")
            dp.alignment = centro
            self._runs(dp, detalle)
            ultimo = dp
        return ultimo

    def figura(self, archivo, pie, ancho_cm=16.0):
        par = self.d.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.keep_with_next = True
        par.paragraph_format.space_after = Pt(0)
        par.add_run().add_picture(str(FIG / archivo), width=Cm(ancho_cm))
        self.pie("Figura", pie)

    def tabla(self, encabezado, filas, pie, anchos=None, tam=9, alinear_num=True, resaltar=None):
        ult = self.pie("Tabla", pie)
        ult.paragraph_format.keep_with_next = True
        for q in self.d.paragraphs[-2:]:
            q.paragraph_format.keep_with_next = True
        t = self.d.add_table(rows=1, cols=len(encabezado))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        bordes_tabla(t)
        anchos = anchos or [16 / len(encabezado)] * len(encabezado)
        for i, txt in enumerate(encabezado):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(txt); r.bold = True; r.font.size = Pt(tam); r.font.color.rgb = RGBColor(255, 255, 255)
            sombrear(c, "1C4E8A")
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        repetir_encabezado(t.rows[0])
        for k, fila in enumerate(filas):
            cells = t.add_row().cells
            no_partir(t.rows[-1])
            for i, v in enumerate(fila):
                txt = str(v)
                par = cells[i].paragraphs[0]
                negrita = txt.startswith("**") and txt.endswith("**")
                if negrita:
                    txt = txt[2:-2]
                r = par.add_run(txt); r.font.size = Pt(tam); r.bold = negrita or None
                es_num = alinear_num and i > 0 and txt.replace(".", "").replace(",", "").replace("−", "").replace("-", "").replace("/", "").replace(" ", "").replace("(", "").replace(")", "").isdigit()
                par.alignment = WD_ALIGN_PARAGRAPH.RIGHT if es_num else WD_ALIGN_PARAGRAPH.LEFT
                par.paragraph_format.space_after = Pt(0)
                if (resaltar is not None and k in resaltar) :
                    sombrear(cells[i], "E8F0FB")
                elif k % 2 == 1:
                    sombrear(cells[i], "F5F5F3")
        for fila in t.rows:
            for i, w in enumerate(anchos):
                fila.cells[i].width = Cm(w)
        self.d.add_paragraph().paragraph_format.space_after = Pt(4)
        return t

    def salto(self):
        self.d.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# ------------------------------------------------------------------ datos de apoyo
def conteo_juliaca():
    out = {}
    for sp in ("train", "val", "test"):
        for c in CLASES:
            base = RAIZ / "dataset" / "juliaca" / sp / c
            objs = [o for o in base.glob("*") if o.is_dir()] if base.exists() else []
            n = sum(1 for o in objs for p in o.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
            out[(sp, c)] = (n, len(objs))
    return out


def f1_clase(js):
    return {r["clase"]: r for r in rj(js)["por_clase"]}


# ------------------------------------------------------------------ contenido
def portada(I: Informe):
    d = I.d
    for _ in range(3):
        d.add_paragraph()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(UNIV.upper()); r.bold = True; r.font.size = Pt(14); r.font.color.rgb = AZUL
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Curso: {CURSO}"); r.font.size = Pt(12); r.font.color.rgb = GRIS
    for _ in range(4):
        d.add_paragraph()
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("INFORME FINAL"); r.bold = True; r.font.size = Pt(13); r.font.color.rgb = GRIS
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(TITULO); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = AZUL
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(SUBTITULO); r.font.size = Pt(13); r.font.color.rgb = GRIS
    # línea
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pPr = p._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); bt = OxmlElement("w:bottom")
    bt.set(qn("w:val"), "single"); bt.set(qn("w:sz"), "12"); bt.set(qn("w:space"), "1"); bt.set(qn("w:color"), "1C4E8A")
    bd.append(bt); pPr.append(bd)
    for _ in range(6):
        d.add_paragraph()
    for etiqueta, valor in (("Autor", AUTOR), ("Curso", CURSO), ("Institución", UNIV), ("Lugar", "Juliaca, Puno, Perú"),
                            ("Fecha", FECHA)):
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(f"{etiqueta}: "); r.bold = True; r.font.size = Pt(12)
        r = p.add_run(valor); r.font.size = Pt(12)
    I.salto()


def indices(I: Informe):
    I.p("Índice", alinear=WD_ALIGN_PARAGRAPH.LEFT).runs[0].font.size = Pt(16)
    I.d.paragraphs[-1].runs[0].bold = True
    I.d.paragraphs[-1].runs[0].font.color.rgb = AZUL
    campo(I.d.add_paragraph(), 'TOC \\o "1-3" \\h \\z \\u', "Actualice el índice (clic derecho → Actualizar campo).")
    I.salto()
    for titulo, tipo in (("Índice de figuras", "Figura"), ("Índice de tablas", "Tabla")):
        par = I.p(titulo, alinear=WD_ALIGN_PARAGRAPH.LEFT)
        par.runs[0].font.size = Pt(14); par.runs[0].bold = True; par.runs[0].font.color.rgb = AZUL
        campo(I.d.add_paragraph(), f'TOC \\h \\z \\c "{tipo}"', "Actualice el índice.")
    I.salto()


def cap1(I: Informe):
    I.h("1. Introducción")
    I.p("En Juliaca, como en la mayoría de ciudades del Perú, los residuos se mezclan en origen y la segregación "
        "depende de que cada persona sepa a qué tacho va cada objeto. La **NTP 900.058:2019** fija el código de "
        "colores de los dispositivos de almacenamiento de residuos sólidos. Este proyecto construye un clasificador "
        "de imagen que, a partir de la foto de **un solo residuo**, predice su **material** y, con él, el color del "
        "tacho que le corresponde según el código ampliado de la norma.")
    I.h("1.1 Objetivo", 2)
    I.p("Entrenar y evaluar de forma honesta un modelo de clasificación de imágenes (8 clases) que funcione con fotos "
        "reales tomadas en Juliaca con celular y con la cámara de una laptop, y usarlo en vivo con una cámara web. "
        "Los objetivos específicos fueron:")
    I.viñetas([
        "ajustar un modelo preentrenado (YOLO26s-cls, Ultralytics) en dos etapas: primero con datos públicos y luego "
        "con fotos propias de Juliaca;",
        "comparar las variantes **solo en validación** (F1 macro) y medir el modelo elegido **una sola vez** en un "
        "conjunto de prueba separado por objeto;",
        "documentar el origen, la licencia y la deduplicación de cada imagen externa, sin generar ni sintetizar imágenes.",
    ])
    I.h("1.2 Clases y tachos", 2)
    I.p("El modelo clasifica el **material**; el color es el del tacho al que va ese material, no el color del "
        "objeto. El orden de la tabla es el de los índices del modelo (orden alfabético de carpetas).")
    ej = {"generales": "bolsas de snack multicapa, servilletas usadas, envolturas mixtas",
          "metales": "latas de bebida y conserva, tapas metálicas",
          "organico": "cáscaras, restos de comida, frutas y tubérculos",
          "papel_carton": "cajas, hojas, periódico, Tetra Pak",
          "peligrosos": "pilas, baterías, cables, focos, envases de químicos",
          "plasticos": "botellas PET de cualquier color, envases rígidos, bolsas simples",
          "sin_residuo": "fondo vacío o solo la mano (evita que el modelo “adivine”)",
          "vidrio": "botellas, frascos y vasos de vidrio"}
    I.tabla(["Índice", "Clase", "Tacho NTP 900.058", "Ejemplos"],
            [[i, c, TACHO[c], ej[c]] for i, c in enumerate(CLASES)],
            "Clases del modelo y color de tacho según la NTP 900.058:2019.", anchos=[1.4, 2.8, 3.0, 8.8])
    I.p("Reglas de etiquetado para casos dudosos (decididas antes de entrenar): se clasifica el material principal "
        "del objeto entero (la tapa no cuenta); los envases Tetra Pak van a papel_carton; los empaques flexibles "
        "impresos o metalizados (snacks, sachets, doypacks) van a generales; los envases rígidos de plástico de "
        "cualquier color van a plasticos; el cartón con grasa va a generales; y se clasifica el material, no el "
        "texto impreso (una cinta plástica que dice “PELIGRO” es plasticos).")


def cap2(I: Informe):
    I.h("2. Datos")
    I.p("Se usan tres grupos de datos con papeles distintos. Ninguna imagen se generó ni se sintetizó: el "
        "aumento de datos (giros, color, RandAugment, borrado aleatorio, escala de grises) solo transforma fotos "
        "reales durante el entrenamiento.")
    cj = conteo_juliaca()
    tot = {sp: (sum(cj[(sp, c)][0] for c in CLASES), sum(cj[(sp, c)][1] for c in CLASES)) for sp in ("train", "val", "test")}
    I.tabla(["Conjunto", "Carpeta", "Imágenes", "Uso"],
            [["Fotos propias de Juliaca", "dataset/juliaca/{train,val,test}",
              f"{tot['train'][0]} / {tot['val'][0]} / {tot['test'][0]}",
              "etapa 2; val elige modelos; test solo una vez al final"],
             ["val_camara (fotos propias)", "dataset/val_camara", "293 (29 objetos)",
              "validación independiente; nunca se entrena con ella"],
             ["Público base", "dataset/publico/{train,val,test}", "8.341 / 1.866 / 2.013",
              "etapa 1; publico/val elige modelos de la etapa 1"],
             ["Externos", "dataset/externo/train", "solo train",
              "se suman al train por fuente; nunca a val ni a test"]],
            "Conjuntos de datos y su papel.", anchos=[3.6, 4.4, 3.2, 4.8], alinear_num=False)
    I.h("2.1 Fotos propias de Juliaca", 2)
    I.p(f"Son **{sum(v[0] for v in tot.values())} fotos de {sum(v[1] for v in tot.values())} objetos** físicos, "
        "tomadas por el autor entre el 24 y el 28 de octubre de 2026: fotos de celular de los días 24 y 25, una "
        "sesión con la cámara de la laptop el 27 (`capturar.bat`) y una sesión con celular en Plaza Vea el 28 "
        "(379 fotos, 67 objetos). La clase y el split de **cada objeto** los decidió el autor a mano: todas las fotos "
        "de un objeto están en una sola carpeta `<split>/<clase>/<objeto>/`, así que ningún objeto aparece en dos "
        "splits (sin fuga por objeto). `auditar.py` comprueba además que no haya fotos idénticas ni casi iguales "
        "(dHash ≤ 4) entre splits.")
    I.tabla(["Clase", "train fotos (obj.)", "val fotos (obj.)", "test fotos (obj.)"],
            [[c] + [f"{cj[(sp, c)][0]} ({cj[(sp, c)][1]})" for sp in ("train", "val", "test")] for c in CLASES]
            + [["**Total**"] + [f"**{tot[sp][0]} ({tot[sp][1]})**" for sp in ("train", "val", "test")]],
            "Fotos y objetos de dataset/juliaca por clase y split (conteo del disco al 2026-11-14).",
            anchos=[4, 4, 4, 4])
    I.p("**val_camara** son 293 fotos de 29 objetos nuevos (7 clases, sin sin_residuo), tomadas con celular el "
        "2026-10-26. Se verificó con dHash que ninguna esté en juliaca, publico ni externo (1 coincidencia, revisada "
        "a ojo: falso positivo). Como es el conjunto más independiente y más grande de validación, es el que se usó "
        "para elegir el modelo final; por eso su número es optimista para el modelo elegido.")
    I.h("2.2 Datos públicos y externos", 2)
    I.p("El público base (`dataset/crudo/companero`) es un dataset recibido de un compañero, organizado por color de "
        "tacho; en su mayoría son fotos de stock y de búsquedas web, y su licencia no está documentada. Se mapeó a "
        "las 7 clases de material (no tiene sin_residuo) y se partió 80/10/10 quitando casi duplicados.")
    I.p("Los datos externos solo entran a `dataset/externo/train` y cumplen cuatro condiciones (regla 1 del proyecto): "
        "aprobación explícita del autor por fuente; licencia que permita uso académico, registrada imagen por imagen "
        "en `datos_externos.csv`; nunca en val ni test; y deduplicación previa con dHash ≤ 4 contra **todos** los "
        "splits de publico y juliaca (y, desde noviembre, también contra val_camara y las 1.455 imágenes del benchmark "
        "del docente, con sus 8 giros/espejos y con ORB + RANSAC). Los informes de cada deduplicación están en "
        "`reportes/externo_*dedup*.csv`.")
    I.tabla(["Fuente (prefijo)", "Clases destino", "Imágenes", "Licencia", "¿En el modelo final?"],
            [["Público base (companero_)", "7 clases", "8.341 train", "no documentada", "Sí (etapa 1 y replay)"],
             ["TACO (taco_)", "generales 139, metales 40", "179", "CC BY 4.0 / ODbL 1.0", "Sí (etapas 1 y 2)"],
             ["Food Waste Detection v1 (fwd_)", "organico", "192", "CC BY 4.0", "Sí (etapa 1)"],
             ["NexTech14 (nx_)", "organico 85, peligrosos 65", "150", "CC BY-NC 4.0", "Sí (etapa 1)"],
             ["RealWaste (realwaste_)", "7 clases", "3.689", "CC BY 4.0", "No"],
             ["TACO extra (tacox_)", "7 clases", "1.104", "CC BY 4.0 / ODbL 1.0", "No"],
             ["TrashNet (trashnet_)", "metales, papel, vidrio", "712", "MIT", "No (fuente del benchmark)"],
             ["Freiburg Groceries (frg_)", "5 clases", "687", "sin licencia formal*", "No"],
             ["Grocery Store (gro_)", "organico, papel_carton", "300", "MIT", "No"],
             ["BDWaste (bdw_)", "organico, peligrosos", "293", "CC BY 4.0", "No"],
             ["MaskNet / E-Waste v9 / RecyBat24", "peligrosos", "150 / 150 / 100", "MIT / CC BY 4.0 / CC BY 4.0", "No"],
             ["Garbage Class. v4 / Smart medical v3", "organico / peligrosos", "500 / 500", "CC BY 4.0", "No (familia del benchmark)"]],
            "Fuentes externas en dataset/externo/train (datos_externos.csv y reportes de importación). "
            "*Freiburg: condición de uso publicada por los autores para entrenamiento y pruebas.",
            anchos=[4.3, 3.6, 2.2, 3.2, 2.7], alinear_num=False, resaltar={0, 1, 2, 3})


def cap3(I: Informe):
    I.h("3. Metodología")
    I.h("3.1 Modelo y entrenamiento en dos etapas", 2)
    I.p("Se usa **YOLO26s-cls** (Ultralytics 8.4.184, PyTorch 2.14.1, CUDA 13.0) preentrenado en ImageNet, con "
        "entrada de 224 × 224 px. El preprocesado de Ultralytics redimensiona el lado corto y recorta el cuadrado "
        "central. El entrenamiento se hizo en una PC con RTX 3060 12 GB (batch 64) y la inferencia en vivo en una "
        "laptop con RTX 3050 4 GB y cámara Logitech MX Brio.")
    I.tabla(["Parámetro", "Etapa 1 (e1_nf)", "Etapa 2 (e2_nf_s43)"],
            [["Pesos iniciales", "yolo26s-cls.pt (ImageNet)", "salida de e1_nf"],
             ["Datos de train", "publico/train + externo taco, fwd, nx", "juliaca/train + replay publico ×2 + externo taco"],
             ["Validación (early stopping)", "publico/val", "juliaca/val"],
             ["Épocas (máx / corridas / mejor)", "60 / 54 / 42", "30 / 25 / 15"],
             ["Optimizador, lr0", "AdamW, 0.000833 (auto)", "AdamW, 0.0001"],
             ["Capas congeladas", "0", "6"],
             ["Paciencia", "15", "10"],
             ["Semilla", "42", "43"],
             ["Aumento de datos", "RandAugment, hsv_h 0.03, flips, erasing 0.3, gris p = 0.3", "igual"]],
            "Receta del modelo final (modelos/etapa2_nf_s43.json y reportes/experimentos/e1_nf.json).",
            anchos=[4.4, 5.8, 5.8], alinear_num=False)
    I.p("El *replay* de la etapa 2 mezcla, por cada clase, el doble de imágenes públicas que fotos de Juliaca, para "
        "que el modelo se adapte a las fotos locales sin olvidar lo aprendido en la etapa 1. La escala de grises "
        "aleatoria (p = 0.3) se añadió para que el modelo no dependa del color del objeto.")
    I.h("3.2 Protocolo de evaluación sin fuga", 2)
    I.viñetas([
        "**train** ajusta los pesos; **val** (publico/val en la etapa 1; juliaca/val y val_camara en la etapa 2) "
        "compara experimentos y elige el modelo; **test** (`dataset/juliaca/test`) mide el modelo ya elegido una sola vez.",
        "Métrica de selección: **F1 macro** en validación, que pesa igual a las clases con pocas fotos. Desde el 28-10, "
        "cada receta se entrena con 3 semillas (42, 43, 44) y gana la de mayor **media** en val_camara; si gana, se "
        "propone su mejor semilla. La regla se fijó antes de ver los resultados.",
        "Un factor a la vez: cada experimento cambia una sola cosa respecto a la referencia.",
        "Cada uso de juliaca/test queda en `reportes/registro_test.csv`. Se usó el 2026-10-28 (modelo de entonces) y el "
        "2026-11-11 (modelo final, solo para reportar). Las métricas de este informe se copian de `reportes/*.json`.",
    ])


def cap4(I: Informe):
    I.h("4. Experimentos")
    I.h("4.1 Etapa 1 (datos públicos)", 2)
    I.tabla(["Experimento", "Cambio", "F1 macro publico/val", "top-1", "Minutos"],
            [["e1_m224", "yolo26m-cls", "0.9139", "0.9126", "16.9"],
             ["**e1_base**", "referencia (yolo26s, 224 px)", "**0.9129**", "0.9116", "9.5"],
             ["e1_s320", "320 px", "0.9113", "0.9110", "15.8"],
             ["e1_drop02", "dropout 0.2", "0.9086", "0.9073", "9.6"],
             ["e1_ext", "+ externo TACO", "0.9153", "0.9143", "15.3"],
             ["e1_nf", "+ taco, fwd, nx, gris 0.3", "0.9052", "0.9046", "9.1"]],
            "Etapa 1 en publico/val (EXPERIMENTOS.md, reportes/experimentos.csv). Diferencias < 0.005 se tratan como empate "
            "y se prefiere el modelo más simple.", anchos=[2.8, 5.4, 3.2, 2.2, 2.4])
    I.h("4.2 Etapa 2 (fotos de Juliaca)", 2)
    I.p("Con los primeros datos (28 fotos en juliaca/val) se ajustaron los hiperparámetros de la etapa 2: el *replay* "
        "de 2 fotos públicas por foto propia dio el mayor F1 macro (0.8139), por encima de lr 3e-4 (0.7479), "
        "freeze 9 (0.7228), la referencia (0.6982) y sin replay (0.6262). Entrenar solo con Juliaca desde ImageNet "
        "dio 0.7827. Después, el peso de la decisión pasó a val_camara, que es más grande e independiente.")
    I.figura("fig_recetas.png", "F1 macro en val_camara de cada receta de la etapa 2 con 3 semillas (reportes/"
             "etapa2_*_val_camara.json). nf (resaltada) es la única con media mayor que j1028, el modelo por defecto "
             "anterior. Las barras muestran la dispersión entre semillas.", 15.5)
    I.tabla(["Receta", "Qué cambia", "Media F1 val_camara", "Resultado"],
            [["j1028", "juliaca con Plaza Vea (28-10), externo taco", "0.5437", "modelo por defecto 28-10 → 10-11"],
             ["rfA / rfB", "+ Roboflow v4 Organik y Smart medical (familia del benchmark)", "0.5445 / 0.5346", "no gana; atajo en gris"],
             ["**nf**", "**sin fuentes del benchmark; + fwd, nx; gris 0.3**", "**0.5568**", "**gana → etapa2_nf_s43**"],
             ["tienda / gro", "+ Freiburg y Grocery Store / solo Grocery", "0.5394 / 0.5147", "no gana"],
             ["nf_m", "yolo26m-cls con la receta nf", "0.5428", "no gana; 1.9× más pesado"],
             ["pel A / pel B", "+ BDWaste, MaskNet, E-Waste, RecyBat24", "0.5137 / 0.5122", "no gana"],
             ["soup", "promedio de pesos de las 3 semillas nf", "0.5454 (un modelo)", "no gana"]],
            "Resumen de las recetas comparadas en validación (EXPERIMENTOS.md).", anchos=[2.4, 6.6, 3.0, 4.0],
            alinear_num=False, resaltar={2})
    I.p("Líneas cerradas que vale la pena reportar: meter **todo** el externo nuevo (5.684 imágenes) en la etapa 2 "
        "bajó val_camara de 0.5183 a 0.4301, porque el externo dominaba el train y el mejor modelo salía en la época 1; "
        "una selección curada de 100 papel_carton + 100 generales mejoró papel_carton pero hundió generales "
        "(generales → papel_carton 17 → 37), porque lo impreso quedó del lado del cartón.")
    I.figura("fig_curvas.png", "Pérdida de entrenamiento y de validación del modelo final por época "
             "(runs/etapa1_e1_nf/results.csv y runs/etapa2_e2_nf_s43/results.csv). En la etapa 2 la pérdida de "
             "validación se estabiliza desde la época 7; Ultralytics guardó best.pt en la época 15.", 16)


def cap5(I: Informe):
    I.h("5. Resultados")
    I.h("5.1 Modelo final", 2)
    I.p("**Modelo entregado:** `modelos/etapa2_nf_s43.pt` (YOLO26s-cls, 5.5 M parámetros, 11.0 MB, SHA-1 "
        "`0fe1ff9468055b2292ded03b707c895be8c5c17e`), elegido el 2026-11-11 por la regla de la media de "
        "3 semillas en val_camara.")
    vc, jv, pv, te = (rj("etapa2_e2_nf_s43_20261110_2132_val_camara.json"),
                      rj("etapa2_e2_nf_s43_20261110_2132_juliaca_val.json"),
                      rj("etapa2_e2_nf_s43_20261110_2132_publico_val.json"), rj("etapa2_nf_s43_juliaca_test.json"))
    I.tabla(["Conjunto", "Fotos", "Top-1", "Top-5", "F1 macro", "Nota"],
            [["publico/val", "1.866", f"{pv['accuracy_top1']:.4f}", f"{pv['accuracy_top5']:.4f}", f"{pv['f1_macro']:.4f}", "validación etapa 1"],
             ["juliaca/val", "83", f"{jv['accuracy_top1']:.4f}", f"{jv['accuracy_top5']:.4f}", f"{jv['f1_macro']:.4f}", "early stopping"],
             ["val_camara", "293", f"{vc['accuracy_top1']:.4f}", f"{vc['accuracy_top5']:.4f}", f"{vc['f1_macro']:.4f}", "elección (optimista)"],
             ["**juliaca/test**", "**98**", f"**{te['accuracy_top1']:.4f}**", f"**{te['accuracy_top5']:.4f}**", f"**{te['f1_macro']:.4f}**", "**una sola vez**"]],
            "Métricas de etapa2_nf_s43 (reportes/etapa2_e2_nf_s43_20261110_2132_*.json y "
            "reportes/etapa2_nf_s43_juliaca_test.json).", anchos=[3.0, 1.8, 2.0, 2.0, 2.2, 5.0])
    I.p("En juliaca/test el modelo acierta **62 de 98 fotos** (18 objetos). Es la segunda vez que se mide un modelo "
        "elegido en este test (la primera, el 2026-10-28, con etapa2_j1028_s43: F1 0.4452, 53/98), así que la "
        "comparación entre ambos es solo informativa y no sirvió para elegir.")
    I.figura("fig_f1_por_clase.png", "F1 por clase del modelo final en val_camara y en juliaca/test. val_camara no "
             "tiene fotos de sin_residuo; en test, organico (ajo) y papel_carton (caja de Zucaritas) tienen un solo "
             "objeto cada uno y el modelo falla todas sus fotos (F1 = 0).", 15.5)
    I.figura("fig_cm_val_camara.png", "Matriz de confusión de etapa2_nf_s43 en val_camara (169/293 aciertos). "
             "La confusión dominante es generales → papel_carton (32 fotos): envolturas y sachets impresos.", 13.5)
    I.figura("fig_cm_test.png", "Matriz de confusión de etapa2_nf_s43 en juliaca/test (62/98 aciertos), copiada de "
             "reportes/etapa2_nf_s43_juliaca_test.json. La lata de atún se va a peligrosos en 8 de 16 fotos, el ajo "
             "a plasticos (6/6) y la caja de cereal a generales (6/6).", 13.5)
    I.h("5.2 Benchmark NTP 900.058 del docente", 2)
    I.p("El modelo final se evaluó una vez con la réplica del benchmark del docente (1.455 imágenes de 3 datasets "
        "públicos; `evaluacion_docente/correr_todo.py`). Ninguna imagen de esos datasets ni derivada de ellos está "
        "en el entrenamiento de nf (se deduplicó contra el benchmark por nombre, dHash con 8 giros y ORB).")
    I.tabla(["Criterio", "Máx.", "etapa2_j1028_s43", "etapa2_nf_s43"],
            [["Exactitud balanceada (9 grupos)", "8", "5.268", "5.482"],
             ["Macro-F1 estricta", "4", "1.895", "2.521"],
             ["Robustez (gris + sin detección)", "3", "2.723", "3.000"],
             ["Cobertura 7 colores", "3", "3.000", "3.000"],
             ["Seguridad (Rojo + orgánicos)", "2", "0.521", "2.000"],
             ["**Nota /20**", "20", "**13.4**", "**16.0**"]],
            "Rúbrica reproducida del benchmark (evaluacion_docente/INFORME.md, EXPERIMENTOS.md). Con nf la caída en "
            "gris es −0.84 pp.", anchos=[6.4, 1.6, 4.0, 4.0])
    I.h("5.3 Velocidad de inferencia", 2)
    I.p("Medido con `scripts/medir_inferencia.py` (batch 1, 224 px, 100 fotos de val_camara, incluye leer y "
        "redimensionar la foto): **22.5 ms** de mediana en GPU (RTX 3060) y **28.9 ms** en CPU (AMD de 16 hilos). "
        "En la demo con detector YOLOE-26s + clasificador, la PC da unos 30 FPS con un objeto.")


def cap6(I: Informe):
    I.h("6. Imágenes de evidencia")
    I.p("Esta sección muestra muestras reales de los datos usados, separadas por origen, ejemplos de lo que se "
        "descartó y predicciones del modelo final. Las miniaturas son el recorte cuadrado central de cada foto "
        "(lo mismo que ve el clasificador). Las fotos se eligieron con semilla fija (42), repartidas entre objetos, "
        "con `informe/generar_figuras.py`.")
    I.p("**Privacidad.** En las fotos propias no se muestran rostros. Cada foto candidata pasó por dos detectores "
        "(YuNet de OpenCV y YOLOE con los prompts “human face”, “head” y “person”). Se eligieron primero fotos sin "
        "ninguna persona detectada. Cuando no alcanzaban, se usaron fotos con la **silueta de la persona pixelada y "
        "desenfocada** (máscara de segmentación de YOLOE, sin tocar el objeto). En la sesión del 2026-10-27, en la que "
        "la persona aparece en todas las fotos, cada foto se **recortó alrededor del objeto** y solo se difuminó el "
        "rostro que quedaba dentro del recorte. Cada mosaico se revisó a ojo; una foto que seguía dejando ver parte de "
        "un rostro se excluyó a mano. Por eso en algunas miniaturas la mano o el fondo aparecen borrosos: el modelo se "
        "entrenó y evaluó con las fotos originales completas, no con estas versiones. "
        "No se muestran nombres ni notas de etiquetado de otras personas; los objetos se identifican por su número.")
    m = META
    # ---------------- 6.1
    I.h("6.1 Datos propios (locales)", 2)
    I.p("Se muestran solo fotos de train y val; las de `dataset/juliaca/test` no se incluyen en los mosaicos. "
        "Bajo cada miniatura figura el objeto y su split.")
    lap = m["laptop_1027"]["clases"]
    I.figura("fig_propio_laptop_1027.jpg",
             "Sesión con la **cámara web de la laptop** (1280 × 720 px, `capturar.bat`), **Juliaca, interior**, "
             f"**2026-10-27**. Clases presentes en la sesión: generales {lap['generales']['mostradas']} de "
             f"{lap['generales']['fotos']} fotos (1 objeto: servilleta) y papel_carton {lap['papel_carton']['mostradas']} de "
             f"{lap['papel_carton']['fotos']} fotos (3 objetos: hojas). En todas las fotos de esta sesión aparece la "
             "persona que sostiene el objeto, así que cada foto se recortó alrededor del objeto (detectado con "
             "YOLOE, margen de 4 %) y se centró sobre un fondo gris. Si dentro del recorte quedaba parte del rostro, "
             "se difuminó solo esa zona y el objeto quedó nítido. "
             + (f"En {lap['generales']['fotos'] - lap['generales']['mostradas']} fotos de la servilleta el detector no "
                "encontró el objeto y no se muestran." if lap['generales']['fotos'] > lap['generales']['mostradas'] else ""),
             15.5)
    pv = m["plazavea_1028"]["clases"]
    txt = "; ".join(f"{c} {v['mostradas']} de {v['fotos']} fotos ({v['objetos']} obj.)" for c, v in pv.items())
    tot_f = sum(v["fotos"] for v in pv.values()); tot_m = sum(v["mostradas"] for v in pv.values())
    I.figura("fig_propio_plazavea_1028a.jpg",
             "Sesión con **celular** en el supermercado **Plaza Vea, Juliaca**, **2026-10-28** (fotos de 1200 × 1600 y "
             "972 × 1296 px), parte 1: generales, metales, organico y papel_carton.", 15.5)
    I.figura("fig_propio_plazavea_1028b.jpg",
             f"Sesión Plaza Vea, parte 2: plasticos y vidrio. Total de las dos partes: {tot_m} de {tot_f} fotos de train "
             f"y val ({txt}). La sesión no tiene peligrosos ni sin_residuo. organico muestra 2 de 3 fotos porque la "
             "tercera dejaba ver parte de un rostro.", 15.5)
    vc = m["val_camara"]["clases"]
    txt = "; ".join(f"{c} {v['mostradas']}/{v['fotos']} ({v['objetos']} obj.)" for c, v in vc.items())
    I.figura("fig_propio_val_camaraa.jpg",
             "**val_camara**: fotos de **celular** (1200 × 1600, 1600 × 1200 y 972 × 1296 px), **Juliaca**, "
             "**2026-10-26**, 29 objetos que no aparecen en ningún otro conjunto. Parte 1: generales, metales, organico "
             "y papel_carton.", 15.5)
    I.figura("fig_propio_val_camarab.jpg",
             f"val_camara, parte 2: peligrosos, plasticos y vidrio. Mostradas/total por clase: {txt} "
             "(293 fotos en total). val_camara no tiene sin_residuo.", 15.5)
    # ---------------- 6.2
    I.h("6.2 Datos de internet usados en el modelo final", 2)
    I.p("El modelo final (`etapa2_nf_s43`) usa cuatro fuentes externas: el público base y, solo en train, TACO, "
        "Food Waste Detection y NexTech14. Los conteos son los de las imágenes usadas para entrenar.")
    pb = m["publico"]
    I.figura("fig_internet_publico.jpg",
             "**Público base (companero)**: dataset recibido de un compañero, organizado por color de tacho; fotos de "
             "stock y de búsquedas web (se ven marcas de agua y fondos de catálogo). URL de origen: no documentada; "
             "**licencia: no documentada**. 6 imágenes por clase de publico/train ("
             + ", ".join(f"{c} {v['total']}" for c, v in pb.items()) + "; 8.341 en total). "
             "Se usa en la etapa 1 y como replay en la etapa 2.", 15.5)
    I.figura("fig_internet_taco.jpg",
             "**TACO** — Proença y Simões (2020), *TACO: Trash Annotations in Context for Litter Detection*, "
             "arXiv:2003.06975, http://tacodataset.org. **Licencia:** anotaciones CC BY 4.0; imágenes CC BY 4.0 u "
             "ODbL 1.0 (© OpenLitterMap & Contributors). Recortes revisados a mano: generales 6 de 139 (Crisp packet, "
             "Other plastic wrapper) y metales 6 de 40 (Aluminium foil). Usado en las etapas 1 y 2.", 15.5)
    I.figura("fig_internet_fwd.jpg",
             "**Food Waste Detection v1** (Roboflow Universe, kkh), "
             "https://universe.roboflow.com/kkh/food-waste-detection-jghxg-ziq5h. **Licencia:** CC BY 4.0. Recortes "
             "de cajas ≥ 64 px → organico: 12 de 192. Usado en la etapa 1.", 15.5)
    I.figura("fig_internet_nx.jpg",
             "**NexTech14_Waste_Dataset** (Salam et al., Mendeley Data p6hgv27jh4), "
             "https://data.mendeley.com/datasets/p6hgv27jh4. **Licencia:** CC BY-NC 4.0 (uso no comercial, académico). "
             "organico 6 de 85 (food, garden, other-organic) y peligrosos 6 de 65 (battery). Usado en la etapa 1.", 15.5)
    I.h("Fuentes probadas que no están en el modelo final", 3)
    nf = m["no_final"]
    I.figura("fig_internet_no_final.jpg",
             "Fuentes externas importadas a externo/train que se probaron en algún experimento pero **no** forman "
             "parte del modelo final; 6 imágenes por fuente con su clase destino. Totales: "
             + ", ".join(f"{k} {v['total']}" for k, v in nf.items()) + ". Licencias: RealWaste CC BY 4.0 "
             "(doi:10.24432/C5SS4G); TACO extra CC BY 4.0/ODbL; TrashNet MIT; Freiburg sin licencia formal (condición "
             "de uso de los autores); Grocery Store MIT; BDWaste CC BY 4.0; MaskNet MIT; E-Waste v9 CC BY 4.0; "
             "RecyBat24 v1 CC BY 4.0; Garbage Classification v4 y Smart medical v3 CC BY 4.0 (estas dos son de la "
             "familia del benchmark del docente y se retiraron en la receta nf).", 11.0)
    # ---------------- 6.3
    I.h("6.3 Ejemplos de lo descartado", 2)
    dup = m["duplicado"]
    I.figura("fig_duplicado_benchmark.jpg",
             "Duplicado detectado contra el benchmark del docente. Izquierda: candidata del split train de *Smart "
             "medical waste segregation v3* (Roboflow, CC BY 4.0). Derecha: la imagen del test del benchmark. Son el "
             f"mismo original (`IMG_6894`) con otra exportación de Roboflow; distancia dHash = {dup['dhash']}. Se "
             "eliminó por la regla dHash ≤ 4. En esa fuente se quitaron 843 por nombre, 2.087 por dHash y 71 por "
             "ORB + RANSAC (reportes/externo_roboflow_benchmark.csv).", 13.5)
    rz = m["rechazadas"]
    I.figura("fig_rechazadas.jpg",
             f"Fuentes y recortes rechazados, con su motivo. **Household Waste** (Kaggle alistairking; {rz['household']['total']} "
             "imágenes): fotos de stock con marca de agua, gráficos, ilustraciones y 1.205 duplicados internos. "
             f"**Surgical Waste** (Mendeley rnhz3fvbj2; {rz['surgical']['total']} imágenes): fotos de internet con marca de "
             "agua de agencias de stock, cuadros de video y montajes; su CC BY no puede licenciar esas fotos. "
             f"**Freiburg** CHOCOLATE, JUICE y COFFEE ({rz['freiburg']['total']} imágenes): cada clase mezcla envases de "
             "papel, vidrio y plástico, así que no se puede asignar un material por clase. **TACO**: "
             f"{rz['taco_descartados']['total']} recortes descartados en la revisión manual (material ambiguo o "
             "recorte ilegible).", 15.5)
    I.p("Otras fuentes descartadas sin importar nada: Garbage Classification (12 classes) (imágenes de búsquedas web "
        "sin licencia propia), Medical-Waste-4.0 (es el origen de la parte médica del benchmark), MARIDA (imágenes "
        "satelitales), Gloves_cls (un solo video con recortes de 15–20 px), mario78 biomedical (hasta 72 % coincidía "
        "con la familia Smart medical) y Garbage Collection v1 (TrashNet aumentado, origen de garbage.v1i del benchmark).")
    # ---------------- 6.4
    I.h("6.4 Predicciones del modelo", 2)
    pr = m["predicciones"]
    vf = m["verificacion_predicciones"]
    I.p("Predicciones de `etapa2_nf_s43` sobre **validación**: val_camara para 7 clases y juliaca/val para "
        "sin_residuo (val_camara no tiene esa clase). No se muestran fotos de juliaca/test, que solo se usa para la "
        "evaluación final. Las predicciones se recalcularon foto a foto y coinciden con los informes "
        f"({vf['val_camara']['aciertos']}/{vf['val_camara']['n']} en val_camara y "
        f"{vf['juliaca/val']['aciertos']}/{vf['juliaca/val']['n']} en juliaca/val; "
        "`informe/datos/predicciones_validacion.csv`).")
    def lst(cs):
        return "; ".join(f"{c} {pr[c]['aciertos']}/{pr[c]['n']}" for c in cs)
    I.figura("fig_predicciones_a.jpg",
             "Aciertos (borde verde, “ACIERTO”) y errores (borde rojo, “ERROR”) por clase real, con la clase predicha y "
             "la confianza (probabilidad top-1). Por clase: 3 aciertos elegidos al azar y los 3 errores de mayor "
             f"confianza, de objetos distintos cuando es posible. Aciertos totales: {lst(CLASES[:4])}.", 15.5)
    I.figura("fig_predicciones_b.jpg",
             f"Continuación: {lst(CLASES[4:])}. En sin_residuo solo hay 1 error. Los errores más seguros son "
             "sistemáticos: sachets y envolturas impresas → papel_carton, plásticos transparentes → vidrio y el "
             "estuche negro de audífonos → vidrio.", 15.5)


def cap7(I: Informe):
    I.h("7. Discusión y limitaciones")
    I.viñetas([
        "**Lo impreso multicapa frente al cartón.** Es la confusión más estable en todos los experimentos: sachets y "
        "envolturas impresas (generales) se predicen como papel_carton, y en test una caja de cereal se predice como "
        "generales. Los datos públicos casi no tienen sachets ni envolturas impresas; agregar cartón público sin "
        "envolturas inclina la frontera hacia papel_carton.",
        "**Plástico transparente frente a vidrio** y **latas frente a pilas** (la lata de atún de test se va a "
        "peligrosos en 8 de 16 fotos).",
        "**Validación pequeña y con sesgo de selección.** val_camara (293 fotos, 29 objetos) se usó para elegir, así "
        "que su 0.5716 es optimista; la media de las 3 semillas (0.5568) es más honesta. juliaca/val tiene fotos de la "
        "misma sesión, tienda, celular y personas que train, y por eso da números altos.",
        "**Test pequeño.** juliaca/test tiene 98 fotos de 18 objetos; organico y papel_carton tienen un solo objeto "
        "cada uno, así que un objeto mueve el F1 de su clase entre 0 y 1. Además, ya se usó dos veces: un modelo "
        "futuro necesita un test nuevo con objetos no vistos.",
        "**Licencias.** El público base no tiene licencia documentada; Freiburg no tiene licencia formal. NexTech14 "
        "es CC BY-NC (solo uso no comercial). Para publicar el modelo con fines comerciales habría que retirar esas "
        "fuentes.",
        "**Correlación en el público base.** Su partición venía aleatoria por foto; se quitaron casi duplicados, pero "
        "puede quedar correlación. Por eso publico/val (≈ 0.87–0.91) sobrestima el desempeño real.",
    ])


def cap8(I: Informe):
    I.h("8. Conclusiones")
    I.viñetas([
        "Se entrenó un clasificador de 8 clases (YOLO26s-cls) en dos etapas que, con fotos nuevas de Juliaca, alcanza "
        "**F1 macro 0.5025 y top-1 0.6327 (62/98)** en el test separado por objeto, y **16.0/20** en el benchmark del "
        "docente sin usar ninguno de sus datos.",
        "Las fotos propias fueron lo que más ayudó: con la sesión de Plaza Vea, las bolsas y doypacks impresos pasaron "
        "de 10/19 a 18/19 aciertos en test frente al modelo sin ellas. Agregar más datos públicos casi nunca mejoró "
        "val_camara, y meterlos todos en la etapa 2 la empeoró.",
        "Comparar recetas por la media de 3 semillas evitó elegir por ruido: la mejor semilla de *tienda* (0.5886) "
        "superaba a nf, pero su media no.",
        "Siguiente paso: más fotos propias de sachets, envolturas y cajas impresas, plásticos transparentes junto a "
        "vidrios, latas y pilas, en distintas sesiones y fondos, y un test nuevo con objetos no vistos.",
    ])


def cap9(I: Informe):
    I.h("9. Referencias y atribución de datos")
    refs = [
        "INACAL (2019). *NTP 900.058:2019 Gestión de residuos. Código de colores para el almacenamiento de residuos sólidos.* Lima: Instituto Nacional de Calidad.",
        "Jocher, G., Qiu, J. y Chaurasia, A. *Ultralytics YOLO* (versión 8.4.184). https://github.com/ultralytics/ultralytics (AGPL-3.0).",
        "Proença, P. F. y Simões, P. (2020). *TACO: Trash Annotations in Context for Litter Detection.* arXiv:2003.06975. http://tacodataset.org — anotaciones CC BY 4.0; imágenes CC BY 4.0 u ODbL 1.0 © OpenLitterMap & Contributors.",
        "Roboflow Universe, kkh. *Food Waste Detection v1.* https://universe.roboflow.com/kkh/food-waste-detection-jghxg-ziq5h — CC BY 4.0.",
        "Salam et al. *NexTech14_Waste_Dataset.* Mendeley Data, p6hgv27jh4. https://data.mendeley.com/datasets/p6hgv27jh4 — CC BY-NC 4.0.",
        "Single, S., Iranmanesh, S. y Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G — CC BY 4.0.",
        "Thung, G. y Yang, M. (2016). *TrashNet* [repositorio]. https://github.com/garythung/trashnet — MIT.",
        "Jund, P., Abdo, N., Eitel, A. y Burgard, W. (2016). *The Freiburg Groceries Dataset.* arXiv:1611.05799.",
        "Klasson, M., Zhang, C. y Kjellström, H. (2019). *A Hierarchical Grocery Store Image Dataset with Visual and Semantic Labels.* WACV 2019. https://github.com/marcusklasson/GroceryStoreDataset — MIT.",
        "Akter, M. et al. (2023). *BDWaste.* Mendeley Data, doi:10.17632/96g5pgfnfw.1 — CC BY 4.0.",
        "Peyvandi, A. et al. (2022). *MaskNet*, Multimedia Tools and Applications, doi:10.1007/s11042-021-11654-w. https://github.com/Tenebris97/MaskNet — MIT.",
        "Roboflow Universe, student-esos5. *E-Waste Classifications v9* — CC BY 4.0. Roboflow Universe, rupiah-detector. *Garbage Classification v4* — CC BY 4.0. Roboflow Universe, visalatchi-t-bme-student. *Smart medical waste segregation v3* — CC BY 4.0.",
        "*RecyBat24 v1.* Zenodo 13126689 — CC BY 4.0.",
        "Wu, W., Peng, H. y Yu, S. (2023). *YuNet: A Tiny Millisecond-level Face Detector* (opencv_zoo, face_detection_yunet_2023mar), usado solo para anonimizar las figuras.",
    ]
    for r in refs:
        par = I.d.add_paragraph()
        par.paragraph_format.left_indent = Cm(1); par.paragraph_format.first_line_indent = Cm(-1)
        par.paragraph_format.space_after = Pt(4)
        I._runs(par, r)
    I.h("Anexo A. Reproducibilidad", 2)
    I.p("Repositorio: `clasificador-residuos-juliaca` (rama main). Entorno fijado en `requirements-lock.txt` y "
        "`entorno_referencia.json` (Python 3.13.15, torch 2.14.1+cu130, ultralytics 8.4.184). Experimentos: "
        "`scripts/experimentar.py` y `EXPERIMENTOS.md`; métricas en `reportes/`. Este informe: "
        "`informe/generar_figuras.py` (figuras, con el .venv del proyecto) y `informe/generar_informe.py` (documento, "
        "con python-docx). Las fotos (`dataset/`) no se suben a git.")


def pie_pagina(I: Informe):
    sec = I.d.sections[0]
    sec.different_first_page_header_footer = True
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run(f"{AUTOR} · {CURSO} · UPeU Juliaca    —    Página "); r.font.size = Pt(8.5); r.font.color.rgb = GRIS
    c = campo(fp, "PAGE", "1"); c.font.size = Pt(8.5); c.font.color.rgb = GRIS
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("Clasificador de residuos – NTP 900.058:2019"); r.font.size = Pt(8.5); r.font.color.rgb = GRIS


def exportar_pdf(docx: Path):
    import win32com.client as win32
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx))
        for _ in range(2):  # dos pasadas: números de página del índice
            doc.Fields.Update()
            for toc in doc.TablesOfContents:
                toc.Update()
            for tof in doc.TablesOfFigures:
                tof.Update()
        doc.Save()
        doc.ExportAsFixedFormat(str(docx.with_suffix(".pdf")), 17)  # wdExportFormatPDF
        n = doc.ComputeStatistics(2)  # wdStatisticPages
        doc.Close(False)
        print("PDF:", docx.with_suffix(".pdf"), "páginas:", n)
    finally:
        word.Quit()


def main():
    I = Informe()
    portada(I)
    indices(I)
    for cap in (cap1, cap2, cap3, cap4, cap5, cap6, cap7, cap8, cap9):
        cap(I)
    pie_pagina(I)
    I.d.core_properties.author = AUTOR
    I.d.core_properties.title = TITULO
    salida = SALIDA
    if "--salida" in sys.argv:  # p. ej. si INFORME_FINAL.docx está abierto en Word
        salida = Path(sys.argv[sys.argv.index("--salida") + 1]).resolve()
    I.d.save(salida)
    print("DOCX:", salida)
    if "--pdf" in sys.argv:
        exportar_pdf(salida)


if __name__ == "__main__":
    main()
