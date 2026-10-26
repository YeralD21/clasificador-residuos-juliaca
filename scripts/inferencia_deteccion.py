"""Detección + clasificación en dos etapas sobre TODO el frame, pensado para la laptop RTX 3050.

  python scripts/inferencia_deteccion.py --camara 0
  python scripts/inferencia_deteccion.py --camara 0 --modelo modelos/etapa1_ganador.pt --umbral 0.7
  python scripts/inferencia_deteccion.py --fuente video.mp4            # probar sin webcam
  python scripts/inferencia_deteccion.py --fuente foto.jpg --bench 100 # medir FPS sin ventana

Etapa 1: un detector de vocabulario abierto (YOLOE o YOLO-World) busca objetos con prompts de
texto: los de residuo (--prompts) y los negativos (--ignorar: persona, ropa, muebles...). Las cajas
negativas no se clasifican ni se dibujan; además se descarta una caja de residuo que coincide casi
entera (IoU >= --iou-ignorar) con una de ropa o mueble de confianza igual o mayor (el detector cree
que es la prenda/mueble). Persona/cara/mano NO anulan cajas: un residuo en la mano sí se muestra.
Pared/puerta/ventana/cortina tampoco: cubren todo el fondo y anularían a un objeto en primer plano.
Filtros de residuo, en orden: zona central --roi (1.0 = todo el frame), área entre --min-area y
--max-area, ropa/mueble, foco en lo cercano (área >= --foco-relativo x la caja de residuo más
grande), nitidez (varianza del Laplaciano >= --nitidez-rel x la del recorte más nítido) y, ordenadas
por área, hasta --max-objetos.
Etapa 2: cada caja se recorta (cuadrada, +10 % de margen, dentro del frame y sin relleno) y la clasifica
TU modelo (--modelo) con el mismo preprocesado que evaluar.py (comprobado con scripts/diagnostico_inferencia.py).
Para que no parpadee, cada caja se sigue entre frames por IoU: se muestra solo si apareció en al
menos --persistencia de los últimos --historial frames, y su clase es el promedio de las
probabilidades del clasificador en esos frames. La caja se dibuja con el color del tacho
(NTP 900.058) y "AZUL - Papel y carton 87%"; si la confianza promedio es menor a --umbral, en morado con
"NO SEGURO". Las cajas que el modelo ve como sin_residuo se ignoran. La clase la decide siempre el clasificador.
Barra superior: el tacho del objeto de mayor área ("TACHO AZUL - Papel y carton (87%)", fondo del color del
tacho), "Sin residuo" si no hay objetos, y una fila de chips si los objetos van a tachos distintos.
Respaldo: si los filtros no dejan ninguna caja, se clasifica el frame completo (mismo camino que evaluar.py,
promediado en --historial frames): sin_residuo -> "Sin residuo"; otra clase con conf >= --umbral -> su tacho con
"(vista completa)"; si no, NO SEGURO. --sin-respaldo lo desactiva.
--debug dibuja en gris las cajas descartadas con su motivo.
Teclas: Q salir, F pantalla completa, S guardar cada recorte mostrado en capturas_revisar/ (para que TÚ lo clasifiques).
"""
import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO, YOLOE, YOLOWorld
from ultralytics.utils.downloads import attempt_download_asset

from comun import RAIZ, TACHO, abrir_camara, crear_ventana, teclas_ventana
from inferencia_webcam import revisar_versiones

# Sin genéricos (box, paper, container): detectaban mesas, hojas y fondo. Se pueden volver a poner con --prompts.
# "cardboard box" y "juice box" son más específicos que "box"; con --debug se ve si vuelven a agarrar fondo.
PROMPTS = ("bottle,can,plastic bag,snack wrapper,battery,phone,charger,cable,cup,glass jar,"
           "fruit peel,food waste,tube,cardboard box,juice box,candy wrapper,chocolate bar,sachet")
IGNORAR = "person,face,hand,clothing,jacket,furniture,sofa,chair,wall,door,window,shelf,curtain,bed"
# Solo estas negativas anulan una caja de residuo que coincide con ellas. Las demás (persona, pared, cortina...) solo se
# ocultan: un objeto en la mano o en primer plano delante de la pared debe seguir mostrándose.
ROPA_MUEBLE = {"clothing", "jacket", "shirt", "coat", "sweater", "furniture", "sofa", "couch", "chair", "table",
               "desk", "shelf", "bed", "cabinet"}
FILTROS = ("ignorada", "duplicada", "roi", "area", "ropa_mueble", "lejana", "borrosa", "exceso")
DETECTORES = RAIZ / "modelos" / "detectores"   # pesos descargados de Ultralytics (no se suben, *.pt)
GRIS = (130, 130, 130)       # --debug: cajas descartadas (línea fina, sin relleno)
NO_SEGURO = (170, 60, 170)   # morado: no es color de ningún tacho (el gris es vidrio)
SIN_RESIDUO = (60, 60, 60)


def cargar_detector(nombre: str, prompts: list[str]):
    ruta = Path(nombre)
    if not ruta.exists():
        DETECTORES.mkdir(parents=True, exist_ok=True)
        ruta = Path(attempt_download_asset(DETECTORES / ruta.name))
    if "world" in ruta.name:
        det = YOLOWorld(str(ruta))
        det.set_classes(prompts)
    else:
        det = YOLOE(str(ruta))
        if "-pf" in ruta.name:
            print("[AVISO] Detector prompt-free: se ignoran --prompts (usa su vocabulario interno).")
        else:
            det.set_classes(prompts, det.get_text_pe(prompts))
    return det


def recorte_cuadrado(frame, caja, margen=0.10):
    """Cuadrado centrado en la caja, lado = lado mayor * (1 + 2*margen), desplazado para que quede DENTRO del frame.
    Sin relleno: el borde gris no aparece al entrenar y cambiaba la predicción (scripts/diagnostico_inferencia.py).
    Si el lado no cabe, se recorta a los bordes del frame y el clasificador hace Resize + CenterCrop, como evaluar.py."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = caja
    lado = max(x2 - x1, y2 - y1) * (1 + 2 * margen)
    lw, lh = int(round(min(lado, w))), int(round(min(lado, h)))
    a = int(round(min(max((x1 + x2) / 2 - lw / 2, 0), w - lw)))
    b = int(round(min(max((y1 + y2) / 2 - lh / 2, 0), h - lh)))
    return frame[b:b + lh, a:a + lw].copy()  # copia: luego se dibuja sobre el frame


def zona_central(frame, fraccion):
    """Rectángulo centrado que ocupa `fraccion` del ancho y del alto del frame."""
    h, w = frame.shape[:2]
    mx, my = w * (1 - fraccion) / 2, h * (1 - fraccion) / 2
    return mx, my, w - mx, h - my


def area(c):
    return (c[2] - c[0]) * (c[3] - c[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nitidez(frame, caja):
    """Varianza del Laplaciano de la caja (sin margen ni relleno), a 128x128 para comparar cajas de distinto tamaño."""
    x1, y1, x2, y2 = (int(v) for v in caja)
    rec = frame[max(y1, 0):y2, max(x1, 0):x2]
    if rec.size == 0:
        return 0.0
    g = cv2.cvtColor(cv2.resize(rec, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def filtrar(r, frame, args, ignorar):
    """Separa cajas de residuo y negativas, y aplica los filtros en orden. Devuelve las cajas de residuo
    (mayor área primero), cuántas descartó cada filtro y las descartadas [(caja, "motivo [prompt]")] para --debug."""
    total = frame.shape[0] * frame.shape[1]
    desc = dict.fromkeys(FILTROS, 0)
    fuera = []
    fondo, cands, conf, prompt = [], [], {}, {}
    for c, cf, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        nombre = r.names[int(k)]
        if nombre in ignorar:
            desc["ignorada"] += 1
            fuera.append((c, f"ignorada [{nombre}]"))
            if nombre in ROPA_MUEBLE:
                fondo.append((c, cf))
        else:
            cands.append((cf, c))
            conf[id(c)], prompt[id(c)] = cf, nombre
    cajas = []
    for _, c in sorted(cands, key=lambda t: -t[0]):  # NMS sin clase entre prompts de residuo (bottle y cup sobre lo mismo)
        if any(iou(c, d) > 0.7 for d in cajas):
            desc["duplicada"] += 1
            fuera.append((c, f"duplicada [{prompt[id(c)]}]"))
        else:
            cajas.append(c)

    def criba(clave, ok):
        nonlocal cajas
        quedan = [c for c in cajas if ok(c)]
        desc[clave] += len(cajas) - len(quedan)
        fuera.extend((c, f"{clave} [{prompt[id(c)]}]") for c in cajas if not ok(c))
        cajas = quedan

    zx1, zy1, zx2, zy2 = zona_central(frame, args.roi)
    criba("roi", lambda c: zx1 <= (c[0] + c[2]) / 2 <= zx2 and zy1 <= (c[1] + c[3]) / 2 <= zy2)
    criba("area", lambda c: args.min_area * total <= area(c) <= args.max_area * total)
    criba("ropa_mueble", lambda c: not any(iou(c, f) >= args.iou_ignorar and cf >= conf[id(c)] for f, cf in fondo))
    if cajas:
        mayor = max(map(area, cajas))
        criba("lejana", lambda c: area(c) >= args.foco_relativo * mayor)
    if args.nitidez and cajas:
        nit = {id(c): nitidez(frame, c) for c in cajas}
        tope = max(nit.values())
        criba("borrosa", lambda c: nit[id(c)] >= args.nitidez_rel * tope)
    cajas.sort(key=area, reverse=True)
    desc["exceso"] += max(0, len(cajas) - args.max_objetos)
    fuera.extend((c, f"exceso [{prompt[id(c)]}]") for c in cajas[args.max_objetos:])
    return cajas[:args.max_objetos], desc, fuera


def clasificar(clf, recortes, imgsz, device, gpu):
    """Probabilidades del clasificador para cada recorte (BGR uint8, como sale de OpenCV). Ultralytics hace
    Resize(lado corto = imgsz) + CenterCrop, BGR->RGB y /255, igual que con una ruta (evaluar.py). En fp32, como
    evaluar.py: con fp16 cambiaban 2 de 293 fotos de val_camara. `gpu` se mantiene por compatibilidad."""
    if not recortes:
        return []
    res = clf.predict(recortes, imgsz=imgsz, device=device, verbose=False)
    return [r.probs.data.float().cpu().numpy() for r in res]


class Pista:
    """Una caja seguida entre frames: historial de apariciones y de probabilidades del clasificador."""

    def __init__(self, caja, historial):
        self.caja = caja
        self.visto = deque(maxlen=historial)
        self.probs = deque(maxlen=historial)
        self.recorte = None

    def actualizar(self, caja, recorte, probs):
        self.caja, self.recorte = caja, recorte
        self.visto.append(True)
        self.probs.append(probs)

    def falta(self):
        self.visto.append(False)


def asociar(pistas, cajas, umbral_iou, historial):
    """Empareja cajas nuevas con pistas por IoU (voraz, mayor IoU primero). Devuelve la pista de cada caja,
    en el mismo orden que `cajas`; las cajas sin pareja abren una pista nueva."""
    pares = sorted(((iou(p.caja, c), i, j) for i, p in enumerate(pistas) for j, c in enumerate(cajas)),
                   reverse=True)
    usadas, out = set(), [None] * len(cajas)
    for v, i, j in pares:
        if v < umbral_iou:
            break
        if i in usadas or out[j] is not None:
            continue
        usadas.add(i)
        out[j] = pistas[i]
    for j, c in enumerate(cajas):
        if out[j] is None:
            out[j] = Pista(c, historial)
            pistas.append(out[j])
    return out


def tinta(color):
    return (0, 0, 0) if sum(color) > 380 else (255, 255, 255)  # texto legible sobre blanco/amarillo/gris


def etiqueta(frame, caja, texto, color):
    x1, y1, x2, y2 = map(int, caja)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y = max(y1, th + 10)
    cv2.rectangle(frame, (x1, y - th - 10), (x1 + tw + 10, y), color, -1)
    cv2.putText(frame, texto, (x1 + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tinta(color), 2)


def descartada(frame, caja, motivo):
    """--debug: caja descartada, gris y fina, con el motivo dentro de la esquina superior."""
    x1, y1, x2, y2 = map(int, caja)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GRIS, 1, cv2.LINE_AA)
    cv2.putText(frame, motivo, (x1 + 4, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRIS, 1, cv2.LINE_AA)


def barra(ancho, objetos, nota=""):
    """Barra superior. objetos: [(tacho, conf, color)] de mayor a menor área; tacho=None si es NO SEGURO.
    Fila 1: el objeto principal (el de mayor área), fondo del color de su tacho y texto grande (+ `nota`).
    Fila 2, solo si los objetos van a tachos distintos: un chip por tacho, con su mayor confianza."""
    e = ancho / 1280
    if not objetos:
        texto, color = "Sin residuo", SIN_RESIDUO
    else:
        t, conf, color = objetos[0]
        texto = (f"NO SEGURO ({conf:.0%})" if t is None else f"TACHO {t} ({conf:.0%})") + nota
    fila = np.full((int(72 * e), ancho, 3), color, np.uint8)
    esc, gr = 1.4 * e, max(2, int(round(3 * e)))
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, esc, gr)
    if tw > ancho - 32 * e:  # textos largos ("... (vista completa)"): se achican para que quepan
        esc *= (ancho - 32 * e) / tw
        (_, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, esc, gr)
    cv2.putText(fila, texto, (int(16 * e), (fila.shape[0] + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, esc, tinta(color),
                gr, cv2.LINE_AA)

    chips = {}  # tacho -> (conf máx, color), en orden de área
    for t, conf, color in objetos:
        k = t or "NO SEGURO"
        if conf > chips.get(k, (-1.0,))[0]:
            chips[k] = (conf, color)  # actualizar no cambia el orden del dict
    if len(chips) < 2:
        return fila
    fila2 = np.full((int(50 * e), ancho, 3), (30, 30, 30), np.uint8)
    esc2, gr2, x = 0.75 * e, max(1, int(round(2 * e))), int(10 * e)
    for k, (conf, color) in chips.items():
        texto = f"{k} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, esc2, gr2)
        if x + tw + 20 * e > ancho:
            break
        y1, y2 = int(7 * e), fila2.shape[0] - int(7 * e)
        cv2.rectangle(fila2, (x, y1), (int(x + tw + 20 * e), y2), color, -1)
        cv2.rectangle(fila2, (x, y1), (int(x + tw + 20 * e), y2), (200, 200, 200), 1)  # borde: chip blanco o negro
        cv2.putText(fila2, texto, (int(x + 10 * e), (y1 + y2 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, esc2, tinta(color),
                    gr2, cv2.LINE_AA)
        x = int(x + tw + 32 * e)
    return np.vstack([fila, fila2])


def crear_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(RAIZ / "modelos" / "etapa2_ext_full.pt"),
                    help="clasificador (modelo actual, ver EXPERIMENTOS.md); también modelos/etapa1_ganador.pt")
    ap.add_argument("--detector", default="yoloe-26n-seg.pt",
                    help="yoloe-26{n,s,m,l,x}-seg.pt (texto), yoloe-26?-seg-pf.pt (prompt-free), yolov8s-worldv2.pt")
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--fuente", help="video o imagen en lugar de la webcam (para pruebas)")
    ap.add_argument("--umbral", type=float, default=0.6)
    ap.add_argument("--conf-det", type=float, default=0.35, help="confianza mínima del detector")
    ap.add_argument("--max-objetos", type=int, default=4, help="máximo de cajas mostradas (las de mayor área)")
    ap.add_argument("--roi", type=float, default=1.0,
                    help="zona central (fracción del ancho y alto) donde debe caer el centro de la caja; 1.0 = todo")
    ap.add_argument("--min-area", type=float, default=0.01, help="área mínima de la caja, fracción del frame")
    ap.add_argument("--max-area", type=float, default=0.85, help="área máxima de la caja, fracción del frame")
    ap.add_argument("--ignorar", default=IGNORAR, help="prompts negativos, separados por coma ('' = ninguno)")
    ap.add_argument("--iou-ignorar", type=float, default=0.6,
                    help="IoU con una caja de ropa/mueble a partir del cual se descarta la caja de residuo")
    ap.add_argument("--foco-relativo", type=float, default=0.25,
                    help="área mínima relativa a la caja de residuo más grande del frame (0 = desactivado)")
    ap.add_argument("--nitidez", action=argparse.BooleanOptionalAction, default=True,
                    help="descarta recortes borrosos (--no-nitidez para desactivar)")
    ap.add_argument("--nitidez-rel", type=float, default=0.3,
                    help="varianza del Laplaciano mínima relativa al recorte más nítido del frame")
    ap.add_argument("--persistencia", type=int, default=3, help="frames (de los últimos --historial) en que debe aparecer")
    ap.add_argument("--historial", type=int, default=5, help="frames recordados por caja (apariciones y probabilidades)")
    ap.add_argument("--iou", type=float, default=0.3, help="IoU mínimo para considerar que es la misma caja")
    ap.add_argument("--prompts", default=PROMPTS, help="separados por coma")
    ap.add_argument("--imgsz", type=int, default=224, help="tamaño del clasificador (con el que se entrenó)")
    ap.add_argument("--imgsz-det", type=int, default=640, help="tamaño del detector (menor = más rápido)")
    ap.add_argument("--bench", type=int, default=0, help="N frames sin ventana y muestra FPS por etapa")
    ap.add_argument("--debug", action="store_true",
                    help="dibuja en gris las cajas descartadas con su motivo (filtro, persistencia o sin_residuo)")
    ap.add_argument("--sin-respaldo", action="store_true",
                    help="no clasificar el frame completo cuando el detector no deja ninguna caja")
    return ap


def main():
    args = crear_parser().parse_args()

    ignorar = [p.strip() for p in args.ignorar.split(",") if p.strip()]
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip() and p.strip() not in ignorar]
    ruta = Path(args.modelo)
    revisar_versiones(ruta)
    gpu = torch.cuda.is_available()
    device = 0 if gpu else "cpu"
    clf = YOLO(str(ruta))
    nombres = clf.names
    det = cargar_detector(args.detector, prompts + ignorar)
    print(f"{'GPU ' + torch.cuda.get_device_name(0) if gpu else 'CPU'} | detector {args.detector} | "
          f"clasificador {ruta.name}\nprompts: {prompts}\nignorar: {ignorar}\nclases: {list(nombres.values())}")

    imagen = None
    if args.fuente:
        imagen = cv2.imread(args.fuente) if Path(args.fuente).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else None
        cap = None if imagen is not None else cv2.VideoCapture(args.fuente)
    else:
        cap = abrir_camara(args.camara)
    ventana = "Deteccion + clasificacion - NTP 900.058"
    if not args.bench:
        crear_ventana(ventana)

    t0, fps, n = time.time(), 0.0, 0
    tiempos = {"det": [], "clf": [], "total": []}
    pistas = []
    completo = deque(maxlen=args.historial)  # respaldo: probabilidades del frame completo, se promedian como una pista
    while True:
        if imagen is not None:
            frame = imagen.copy()
        else:
            ok, frame = cap.read()
            if not ok:
                if args.fuente:
                    break  # fin del video
                continue
        ta = time.perf_counter()
        # NMS por clase: así una caja de residuo no se pierde contra la de ropa/mueble que la cubre (se decide en filtrar)
        r = det.predict(frame, conf=args.conf_det, imgsz=args.imgsz_det, device=device, quantize=16 if gpu else None,
                        agnostic_nms=False, max_det=50, verbose=False)[0]
        tb = time.perf_counter()

        cajas, descartes, fuera = filtrar(r, frame, args, ignorar)
        recortes = [recorte_cuadrado(frame, c) for c in cajas]
        de_caja = asociar(pistas, cajas, args.iou, args.historial)
        actualizadas = {id(p) for p in de_caja}
        for p in pistas:
            if id(p) not in actualizadas:
                p.falta()
        for p, caja, rec, pr in zip(de_caja, cajas, recortes, clasificar(clf, recortes, args.imgsz, device, gpu)):
            p.actualizar(caja, rec, pr)
        pistas = [p for p in pistas if any(p.visto)]  # olvida las que no aparecieron en todo el historial

        dibujos = []  # (caja, recorte, tacho o None, conf, color, clase), de mayor a menor área
        estables = sorted((p for p in pistas if sum(p.visto) >= args.persistencia),
                          key=lambda p: area(p.caja), reverse=True)[:args.max_objetos]
        for p in estables:
            prom = np.mean(p.probs, axis=0)
            i = int(prom.argmax())
            clase, conf = nombres[i], float(prom[i])
            if clase == "sin_residuo":
                fuera.append((p.caja, f"sin_residuo {conf:.0%}"))
            elif conf < args.umbral:
                dibujos.append((p.caja, p.recorte, None, conf, NO_SEGURO, "NOSEGURO"))
            else:
                dibujos.append((p.caja, p.recorte, TACHO[clase][0], conf, TACHO[clase][1], clase))
        if args.debug:
            fuera += [(p.caja, f"persistencia {sum(p.visto)}/{args.persistencia}") for p in pistas
                      if p.visto[-1] and sum(p.visto) < args.persistencia]

        # Respaldo: si los filtros no dejan ninguna caja, se clasifica el frame completo (mismo camino que evaluar.py,
        # ver diagnostico_inferencia.py). sin_residuo -> "Sin residuo"; conf >= umbral -> su tacho; si no, NO SEGURO.
        vista = []  # [(tacho o None, conf, color, clase)] de la vista completa, para la barra
        if cajas or args.sin_respaldo:
            completo.clear()
        else:
            completo.append(clasificar(clf, [frame], args.imgsz, device, gpu)[0])
            prom = np.mean(completo, axis=0)
            i = int(prom.argmax())
            clase, conf = nombres[i], float(prom[i])
            if clase != "sin_residuo":
                vista = [(TACHO[clase][0], conf, TACHO[clase][1], clase) if conf >= args.umbral
                         else (None, conf, NO_SEGURO, "NOSEGURO")]
        tc = time.perf_counter()
        tiempos["det"].append(tb - ta)
        tiempos["clf"].append(tc - tb)
        tiempos["total"].append(tc - ta)
        n += 1

        if args.bench:
            if n >= args.bench:
                break
            continue

        if args.roi < 1.0:
            zx1, zy1, zx2, zy2 = zona_central(frame, args.roi)
            cv2.rectangle(frame, (int(zx1), int(zy1)), (int(zx2), int(zy2)), (200, 200, 200), 1, cv2.LINE_AA)
        if args.debug:
            for caja, motivo in fuera:
                descartada(frame, caja, motivo)
        for caja, _, tacho, conf, color, _ in dibujos:
            etiqueta(frame, caja, f"NO SEGURO {conf:.0%}" if tacho is None else f"{tacho} {conf:.0%}", color)
        ahora = time.time()
        fps = 0.9 * fps + 0.1 / max(ahora - t0, 1e-6)
        t0 = ahora
        limpio = frame.copy() if vista else None  # para S: el frame sin dibujos
        cv2.putText(frame, f"{fps:.0f} FPS | {len(dibujos)} obj" + (" | vista completa" if completo else "")
                    + (" | DEBUG" if args.debug else ""),
                    (12, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if vista:
            sup = barra(frame.shape[1], [v[:3] for v in vista], nota=" (vista completa)")
        else:
            sup = barra(frame.shape[1], [(t, c, col) for _, _, t, c, col, _ in dibujos])
        cv2.imshow(ventana, np.vstack([sup, frame]))

        k = teclas_ventana(ventana)
        if k == "q":
            break
        if k == "s":
            d = RAIZ / "capturas_revisar"
            d.mkdir(exist_ok=True)
            marca = time.strftime("%Y%m%d_%H%M%S")
            guardar = [(rec, clase) for _, rec, _, _, _, clase in dibujos]
            guardar += [(limpio, f"{v[3]}_completo") for v in vista]
            for j, (rec, clase) in enumerate(guardar):
                f = d / f"pred-{clase}_{marca}_{j}.jpg"
                cv2.imwrite(str(f), rec)
                print(f"Guardado {f}")
            if guardar:
                print("Clasifícalos TÚ y muévelos a dataset/juliaca/<split>/<clase>/objNNN/")

    if args.bench and n:
        w = max(1, n // 10)  # descarta el calentamiento inicial
        for k, v in tiempos.items():
            ms = 1000 * float(np.mean(v[w:]))
            print(f"{k:6s} {ms:7.1f} ms  ({1000 / ms:5.1f} FPS)")
        print(f"frames medidos: {n - w} (descartados {w} de calentamiento) | último frame: {len(r.boxes)} cajas del "
              f"detector -> {len(cajas)} tras filtros -> {len(dibujos)} mostradas")
        print("descartes último frame: " + ", ".join(f"{k}={v}" for k, v in descartes.items()))
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
