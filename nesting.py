"""
Nesting real de letras de acrílico sobre una hoja de corte.

Reutiliza el contorno REAL de cada letra (no su caja) para acomodarlas en
la hoja con una tolerancia de separación, y genera el plano en PDF.

No usa la medida ya calculada por _extract_pieces_from_pdf /
_extract_pieces_geometric (esas dan alto/ancho, no el contorno). En vez de
duplicar todo ese pipeline de "Medidas: A x B mm" + cotas, este módulo:

  1. Llama a esas mismas funciones (ya probadas) para encontrar el grupo de
     letras que corresponde a la fila que el usuario aplicó (se identifica
     por las alturas individuales, en el mismo orden izquierda->derecha
     que ya viajó al navegador como `heights_detail[]`).
  2. Vuelve a abrir el PDF, re-emparejando cada trazo del grupo por su
     bounding box exacto (misma llamada a page.get_drawings(), resultado
     determinista) para recuperar sus puntos de trayectoria reales.
  3. Acomoda esos contornos en la hoja con un algoritmo de relleno
     esquina-a-esquina sobre una grilla de ocupación (numpy + OpenCV).
  4. Dibuja el plano de corte en un PDF (una página por hoja usada).
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - ya se valida en app.py antes de llamar aquí
    fitz = None

SHEET_W_MM_DEFAULT = 1800.0  # 180 cm — hoja de acrílico
SHEET_H_MM_DEFAULT = 2400.0  # 240 cm
GAP_MM_DEFAULT = 6.0
GRID_RES_MM = 3.0            # resolución de la grilla de ocupación (más fino = más lento y más ajustado)
MATCH_TOL_PT = 0.05          # tolerancia al re-emparejar un trazo por su bounding box exacto

# Tamaño de hoja según el material — cada material se corta de su propia
# lámina, así que cada uno arma su(s) hoja(s) por separado (no se pueden
# mezclar letras de materiales distintos en un mismo plano de corte).
# Se busca por coincidencia de texto (ver _material_for_subprocess) usando
# las mismas palabras clave que ya usa get_material_suggestions en app.py.
MATERIAL_SHEETS = [
    ("acrílico",  "Acrílico blanco",  1800.0, 2400.0),
    ("lámina",    "Lámina cal. 19",   1220.0, 3000.0),
    ("trovicel",  "Trovicel",         1220.0, 2440.0),
]


def material_for_subprocess(subprocess: str) -> Optional[Tuple[str, float, float]]:
    """Si el nombre del subproceso de CORTE menciona un material con hoja
    conocida, devuelve (etiqueta, ancho_mm, alto_mm) — None si no aplica
    (p.ej. subprocesos de CORTE que no son de una lámina con medida fija)."""
    sub_lower = subprocess.lower()
    for keyword, label, w_mm, h_mm in MATERIAL_SHEETS:
        if keyword in sub_lower:
            return label, w_mm, h_mm
    return None


# ══ 1) CONTORNO REAL DE CADA LETRA (a partir del grupo ya identificado) ═══════

def _flatten_drawing_items(items, bezier_steps: int = 8) -> List[List[Tuple[float, float]]]:
    """Convierte los items de un dibujo de page.get_drawings() (líneas,
    curvas bezier, rects, quads) en una lista de polígonos — uno por cada
    subtrazo (p.ej. el aro exterior y el interior de una "O"). No se
    distingue relleno/hueco: todos se tratan como sólidos al rasterizar
    (ver _rasterize_polygons), que es la silueta que importa para nesting.
    """
    JUMP_TOL = 0.05  # pt — mismo margen que MATCH_TOL_PT para bounding boxes

    polygons: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    last_point: Optional[Tuple[float, float]] = None

    def flush():
        nonlocal current
        if len(current) >= 3:
            polygons.append(current)
        current = []

    def maybe_break(p1: Tuple[float, float]):
        # Un subtrazo nuevo (p.ej. el hueco triangular de una "A", aparte
        # del contorno exterior) no siempre trae su propio "m" explícito
        # en los datos que da LibreOffice/PyMuPDF — si el punto de arranque
        # de este segmento no es donde se quedó el trazo anterior, SÍ es un
        # subtrazo nuevo aunque no haya "m": sin este chequeo, dos contornos
        # separados quedaban unidos por una línea diagonal fantasma
        # (artefacto visible como un triángulo/raya suelta en el plano).
        nonlocal last_point
        if current and last_point is not None and (
                abs(p1[0] - last_point[0]) > JUMP_TOL or abs(p1[1] - last_point[1]) > JUMP_TOL):
            flush()

    for it in items:
        kind = it[0]
        if kind == "m":
            flush()
            current = [(it[1].x, it[1].y)]
            last_point = current[-1]
        elif kind == "l":
            p1, p2 = (it[1].x, it[1].y), (it[2].x, it[2].y)
            maybe_break(p1)
            if not current:
                current.append(p1)
            current.append(p2)
            last_point = p2
        elif kind == "c":
            p1 = (it[1].x, it[1].y)
            p2, p3, p4 = it[2], it[3], it[4]
            maybe_break(p1)
            if not current:
                current.append(p1)
            for s in range(1, bezier_steps + 1):
                t = s / bezier_steps
                mt = 1 - t
                x = mt ** 3 * p1[0] + 3 * mt * mt * t * p2.x + 3 * mt * t * t * p3.x + t ** 3 * p4.x
                y = mt ** 3 * p1[1] + 3 * mt * mt * t * p2.y + 3 * mt * t * t * p3.y + t ** 3 * p4.y
                current.append((x, y))
            last_point = current[-1]
        elif kind == "re":
            flush()
            r = it[1]
            polygons.append([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
            last_point = None
        elif kind == "qu":
            flush()
            q = it[1]
            polygons.append([(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y)])
            last_point = None
    flush()
    return polygons


def _match_group_for_nesting(letter_groups_by_page, letter_heights: List[float], tol: float = 0.05):
    """Encuentra, entre los grupos ya calculados por el pipeline normal de
    medición, el que coincide con las alturas individuales que llegaron del
    formulario (mismo orden izquierda->derecha que ya se mandó al navegador
    en `heightsDetail`) — así identificamos SIN AMBIGÜEDAD cuál de los 1 o 2
    grupos de un archivo es el que el usuario aplicó a esta fila."""
    for page_no, groups in letter_groups_by_page:
        for g in groups:
            pieces = g.get("pieces") or []
            if len(pieces) != len(letter_heights):
                continue
            if all(abs(p["height_cm"] - h) <= tol for p, h in zip(pieces, letter_heights)):
                return page_no, g
    return None, None


def extract_group_polygons_mm(
    cdr_pdf_path: str, letter_heights: List[float]
) -> Optional[List[List[List[Tuple[float, float]]]]]:
    """Devuelve, para el grupo de letras que coincide con `letter_heights`,
    una lista de piezas — cada pieza es una lista de polígonos [(x_mm,
    y_mm), ...] con origen en su propia esquina superior izquierda. None si
    no se pudo identificar o re-emparejar el grupo (mejor no nestear que
    nestear con datos incorrectos)."""
    import app as _app  # import perezoso: nesting.py es importado por app.py

    pieces_from_text = _app._extract_pieces_from_pdf(cdr_pdf_path)
    by_page = [(p["page"], p["letter_groups"]) for p in pieces_from_text]
    page_no, group = _match_group_for_nesting(by_page, letter_heights)

    if group is None:
        geo_groups = _app._extract_pieces_geometric(cdr_pdf_path)
        grouped: Dict[int, List[dict]] = {}
        for g in geo_groups:
            grouped.setdefault(g.get("page", 0), []).append(g)
        page_no, group = _match_group_for_nesting(list(grouped.items()), letter_heights)

    if group is None:
        return None

    target_rects = group["rects"]  # [[x0,y0,x1,y1], ...]
    # OJO: group["rects"] (ordenado por alto) y group["pieces"] (ordenado de
    # izquierda a derecha, ver cluster_by_x en app.py) son DOS ÓRDENES
    # DISTINTAS del mismo conjunto de trazos — no se puede emparejar por
    # índice entre ambos. Tampoco sirve group["height_cm"]: en un concepto
    # de un solo grupo viene sobrescrito con el alto NOMINAL declarado en
    # "Medidas:", que no corresponde a la altura geométrica real de ninguna
    # letra individual. Por eso la escala se reconstruye con PROMEDIOS
    # (independientes del orden): alto promedio real de las piezas (siempre
    # geométrico) contra alto promedio en puntos de los mismos trazos.
    pieces_meta = group.get("pieces") or []
    if len(pieces_meta) != len(target_rects):
        return None
    avg_height_cm = sum(p["height_cm"] for p in pieces_meta) / len(pieces_meta)
    avg_height_pt = sum(r[3] - r[1] for r in target_rects) / len(target_rects)
    if avg_height_pt <= 0 or avg_height_cm <= 0:
        return None
    mm_per_pt = (avg_height_cm * 10.0) / avg_height_pt

    with fitz.open(cdr_pdf_path) as doc:
        if page_no >= doc.page_count:
            return None
        page = doc[page_no]
        drawings = page.get_drawings()
        result: List[List[List[Tuple[float, float]]]] = []
        for tx0, ty0, tx1, ty1 in target_rects:
            match = None
            for d in drawings:
                r = d["rect"]
                if (abs(r.x0 - tx0) < MATCH_TOL_PT and abs(r.y0 - ty0) < MATCH_TOL_PT and
                        abs(r.x1 - tx1) < MATCH_TOL_PT and abs(r.y1 - ty1) < MATCH_TOL_PT):
                    match = d
                    break
            if match is None:
                return None
            polys_pt = _flatten_drawing_items(match["items"])
            if not polys_pt:
                return None
            polys_mm = [
                [((px - tx0) * mm_per_pt, (py - ty0) * mm_per_pt) for px, py in poly]
                for poly in polys_pt
            ]
            result.append(polys_mm)
    return result


# ══ 2) NESTING (relleno esquina-a-esquina sobre grilla de ocupación) ══════════

def _rasterize_polygons(polygons_mm: List[List[Tuple[float, float]]], res_mm: float,
                         pad_mm: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """Rasteriza un conjunto de polígonos (en mm, ya en su propio bbox) a
    una máscara booleana. Todos los subtrazos se unen como sólido (sin
    huecos: para nesting importa la silueta, no el detalle interior).
    Devuelve (máscara, min_x_mm, min_y_mm) — el offset para reubicar la
    máscara sobre la pieza real al dibujar el plano."""
    xs = [p[0] for poly in polygons_mm for p in poly]
    ys = [p[1] for poly in polygons_mm for p in poly]
    min_x, min_y = min(xs) - pad_mm, min(ys) - pad_mm
    max_x, max_y = max(xs) + pad_mm, max(ys) + pad_mm
    w = max(1, math.ceil((max_x - min_x) / res_mm))
    h = max(1, math.ceil((max_y - min_y) / res_mm))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv_polys = [
        np.array([[(x - min_x) / res_mm, (y - min_y) / res_mm] for x, y in poly], dtype=np.int32)
        for poly in polygons_mm
    ]
    # Cada subtrazo se rellena POR SEPARADO (no en una sola llamada a
    # fillPoly con todos juntos): así el hueco interior de una letra como
    # "O"/"e" queda sólido en la máscara de ocupación en vez de quedar como
    # espacio libre donde podría anidarse OTRA pieza — nada de cortar
    # piezas escondidas dentro de una letra, aunque sí se aprovecha el
    # contorno real de su silueta exterior. El plano en PDF sí dibuja el
    # hueco real (ver render_nesting_pdf): esto solo afecta la colisión.
    for poly in cv_polys:
        cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool), min_x, min_y


def _mm_to_radius_cells(radius_mm: float, res_mm: float) -> int:
    # Redondea SIEMPRE hacia arriba: mejor una pieza un poco más separada
    # de lo pedido que una que quede más cerca de lo que el usuario pidió.
    return max(1, math.ceil(radius_mm / res_mm))


def _disk_kernel(radius_cells: int) -> np.ndarray:
    """Disco euclidiano exacto (x²+y² <= r²), no la aproximación de
    cv2.getStructuringElement(MORPH_ELLIPSE, ...) — a radios chicos (2-3
    celdas, que es lo normal aquí) esa aproximación de OpenCV recorta más
    de la cuenta en las diagonales y termina dilatando menos de lo pedido:
    se midieron piezas a 1-3 mm de distancia real con una tolerancia de
    6 mm configurada, por esto mismo."""
    d = radius_cells * 2 + 1
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return ((x * x + y * y) <= radius_cells * radius_cells).astype(np.uint8)


def _dilate_mask(mask: np.ndarray, radius_mm: float, res_mm: float) -> np.ndarray:
    """OJO: cv2.dilate NO agranda el arreglo — si la máscara ya llega justa
    al borde (que es como la arma _rasterize_polygons, sin margen), el halo
    de separación que debería crecer hacia afuera se recorta ahí mismo y
    la pieza queda con MENOS separación real de la pedida (así se
    encontraron piezas a 0-3 mm con 6 mm configurados). Por eso se agrega
    relleno de `radius_cells` en cada lado ANTES de dilatar — quien llame
    a esto debe correr el origen (min_x/min_y) ese mismo tanto hacia atrás,
    ver nest_pieces.

    `radius_mm` es lo que crece ESTA pieza — cuando dos piezas dilatadas
    cada una por la MITAD del gap dejan de traslaparse, quedan exactamente
    `gap_mm` separadas entre sí (por eso nest_pieces llama esto con
    gap_mm/2, no con gap_mm completo)."""
    radius_cells = _mm_to_radius_cells(radius_mm, res_mm)
    padded = np.pad(mask.astype(np.uint8), radius_cells, mode="constant")
    kernel = _disk_kernel(radius_cells)
    dilated = cv2.dilate(padded, kernel)
    return dilated.astype(bool)


def _find_first_fit(occupied: np.ndarray, piece_mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Primera posición (fila, columna == y, x en celdas) donde `piece_mask`
    cabe sobre `occupied` sin traslape, escaneando de arriba-izquierda hacia
    abajo-derecha (relleno esquina-a-esquina). Usa correlación 2D por FFT
    (numpy puro, sin depender de scipy) para probar TODAS las posiciones de
    una sola vez en vez de una por una."""
    oh, ow = occupied.shape
    ph, pw = piece_mask.shape
    if ph > oh or pw > ow:
        return None

    fh = oh + ph - 1
    fw = ow + pw - 1
    occ_f = np.fft.rfft2(occupied.astype(np.float64), s=(fh, fw))
    # correlación válida: invertir el kernel para que fft-conv == correlación
    piece_f = np.fft.rfft2(piece_mask.astype(np.float64)[::-1, ::-1], s=(fh, fw))
    conv = np.fft.irfft2(occ_f * piece_f, s=(fh, fw))
    # conv[ph-1 + y, pw-1 + x] == overlap de piece colocada en (y, x)
    valid = conv[ph - 1: ph - 1 + (oh - ph + 1), pw - 1: pw - 1 + (ow - pw + 1)]
    fits = valid < 0.5  # overlap 0 (con margen por error numérico de la fft)
    if not fits.any():
        return None
    ys, xs = np.nonzero(fits)
    # arriba-izquierda primero: fila mínima y, a igualdad, columna mínima
    idx = np.lexsort((xs, ys))[0]
    return int(ys[idx]), int(xs[idx])


ROTATION_STEP_DEG_DEFAULT = 90.0  # candidatos de rotación por pieza (0°, 90°, 180°, 270°)
# OJO: 90° es el paso más grande que sigue teniendo sentido para mantener el
# nesting instantáneo en el flujo síncrono de la calculadora. Ángulos
# intermedios (30°/45°/etc.) SÍ acomodan letras más apretadas entre sí (como
# en un nesting profesional), pero cada uno agranda el bounding box de la
# pieza de forma irregular — eso, combinado con que el algoritmo coloca cada
# pieza una sola vez sin poder corregir después, deja huecos con formas que
# ya no le quedan a las piezas medianas que faltan (se probó con pasos de
# 30°: terminó usando MÁS hojas, no menos, y tardó ~80s para 15 piezas). Con
# múltiplos de 90° el bounding box nunca crece (solo intercambia ancho por
# alto), así que no tiene ese problema.


def _rotate_polygons(polygons_mm: List[List[Tuple[float, float]]], angle_deg: float
                      ) -> List[List[Tuple[float, float]]]:
    """Rota un conjunto de polígonos (en mm) alrededor del origen. El origen
    de estos polígonos es siempre la esquina de su propio bbox (así vienen
    de extract_group_polygons_mm), así que rotarlos alrededor de (0,0) es
    equivalente a rotarlos alrededor de esa esquina — no importa, lo único
    que importa es la FORMA resultante, ya que _rasterize_polygons vuelve a
    normalizar el origen de cada variante rotada a su propio bbox."""
    if angle_deg % 360 == 0:
        return polygons_mm
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return [
        [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in poly]
        for poly in polygons_mm
    ]


def nest_pieces(
    pieces_polygons_mm: List[List[List[Tuple[float, float]]]],
    sheet_w_mm: float = SHEET_W_MM_DEFAULT,
    sheet_h_mm: float = SHEET_H_MM_DEFAULT,
    gap_mm: float = GAP_MM_DEFAULT,
    res_mm: float = GRID_RES_MM,
    rotation_step_deg: float = ROTATION_STEP_DEG_DEFAULT,
) -> Dict[str, Any]:
    """Acomoda cada pieza (lista de polígonos en mm) en una o más hojas,
    probando la pieza en varias rotaciones (cada `rotation_step_deg`) y
    quedándose con la que permita colocarla más arriba-a-la-izquierda —
    así las piezas de forma irregular (letras) encajan unas en los huecos
    de otras en vez de quedar todas "paradas" perdiendo el espacio entre
    ellas. Ordena de mayor a menor área para empacar mejor (heurística
    estándar). Devuelve {"sheets": [...], "sheet_w_mm", "sheet_h_mm",
    "gap_mm"} donde cada hoja es una lista de piezas ya con su rotación y
    posición final resuelta."""
    sheet_w_cells = max(1, round(sheet_w_mm / res_mm))
    sheet_h_cells = max(1, round(sheet_h_mm / res_mm))

    indexed = list(enumerate(pieces_polygons_mm))

    def _area(polys):
        a = 0.0
        for poly in polys:
            for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
                a += x1 * y2 - x2 * y1
        return abs(a) / 2

    indexed.sort(key=lambda kv: -_area(kv[1]))

    angles = [0.0]
    if rotation_step_deg and rotation_step_deg > 0:
        n_steps = max(1, round(360.0 / rotation_step_deg))
        angles = [i * 360.0 / n_steps for i in range(n_steps)]

    # Cada pieza se dilata por la MITAD del gap (no el gap completo): así,
    # cuando dos piezas dilatadas dejan de traslaparse, sus contornos
    # reales quedan exactamente `gap_mm` aparte — dilatar por el gap
    # completo en ambas daría el DOBLE de separación de la pedida. Por la
    # misma razón el margen contra la orilla de la hoja usa esa misma
    # mitad: la orilla "ya trae puesta" la otra mitad, como si fuera el
    # borde de una pieza fantasma pegada al filo.
    half_gap_radius_cells = _mm_to_radius_cells(gap_mm / 2, res_mm)
    border_cells = half_gap_radius_cells

    def _new_grid() -> np.ndarray:
        g = np.zeros((sheet_h_cells, sheet_w_cells), dtype=bool)
        if border_cells > 0:
            g[:border_cells, :] = True
            g[-border_cells:, :] = True
            g[:, :border_cells] = True
            g[:, -border_cells:] = True
        return g

    sheets: List[List[Dict[str, Any]]] = [[]]
    occupied_grids = [_new_grid()]

    for orig_idx, polys in indexed:
        variants = []
        for ang in angles:
            rp = _rotate_polygons(polys, ang)
            true_mask, min_x, min_y = _rasterize_polygons(rp, res_mm)
            test_mask = _dilate_mask(true_mask, gap_mm / 2, res_mm)
            # _dilate_mask rellenó `half_gap_radius_cells` celdas en cada
            # lado antes de dilatar (para que el halo no se recorte) — el
            # origen real de test_mask queda corrido ese mismo tanto hacia
            # atrás.
            min_x_test = min_x - half_gap_radius_cells * res_mm
            min_y_test = min_y - half_gap_radius_cells * res_mm
            ph, pw = test_mask.shape
            if ph <= sheet_h_cells and pw <= sheet_w_cells:
                variants.append((ang, rp, test_mask, min_x_test, min_y_test))
        if not variants:
            continue  # pieza más grande que la hoja en cualquier rotación: se omite

        placed = False
        sheet_i = 0
        while not placed:
            if sheet_i == len(sheets):
                sheets.append([])
                occupied_grids.append(_new_grid())
            best = None  # (cy, cx, ang, rp, test_mask, min_x, min_y)
            for ang, rp, test_mask, min_x, min_y in variants:
                pos = _find_first_fit(occupied_grids[sheet_i], test_mask)
                if pos is not None and (best is None or pos < (best[0], best[1])):
                    best = (pos[0], pos[1], ang, rp, test_mask, min_x, min_y)
            if best is not None:
                cy, cx, ang, rp, test_mask, min_x, min_y = best
                ph, pw = test_mask.shape
                occupied_grids[sheet_i][cy:cy + ph, cx:cx + pw] |= test_mask
                # Trasladar cada punto original (px,py) del polígono ya
                # rotado con (px + offset_x_mm, py + offset_y_mm) da su
                # posición final real sobre la hoja: la celda (0,0) de la
                # máscara de la pieza —que arrancó en (min_x,min_y) al
                # rasterizar— queda anclada en la celda (cy,cx) de la hoja.
                sheets[sheet_i].append({
                    "orig_index": orig_idx,
                    "polygons_mm": rp,
                    "rotation_deg": ang,
                    "sheet_index": sheet_i,
                    "offset_x_mm": cx * res_mm - min_x,
                    "offset_y_mm": cy * res_mm - min_y,
                })
                placed = True
            else:
                sheet_i += 1

    return {
        "sheets": sheets,
        "sheet_w_mm": sheet_w_mm,
        "sheet_h_mm": sheet_h_mm,
        "gap_mm": gap_mm,
    }


# ══ 3) PLANO EN PDF ════════════════════════════════════════════════════════════
#
# matplotlib exporta PDF vectorial de verdad (no rasteriza el dibujo) y da
# texto/anotaciones mucho más cómodos que dibujar a mano con PyMuPDF —
# número y medida de cada pieza, cuadrícula de referencia en cm, resumen de
# aprovechamiento por hoja. PyMuPDF se sigue usando para TODO lo demás
# (leer el .cdr/PDF, extraer los contornos): esto solo reemplaza el dibujo
# final del plano.

import matplotlib
matplotlib.use("Agg")  # sin esto intenta abrir una ventana — no hay una en el servidor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

MM_TO_IN = 1.0 / 25.4

# Paleta sobria (no arcoíris): grises/azules apagados que se distinguen
# entre piezas vecinas sin verse como una demo — pensada para imprimirse
# también en blanco y negro sin perder legibilidad (contraste por trazo,
# no solo por color).
PIECE_COLORS = [
    "#cfe0f3", "#d9e6d5", "#eadcc9", "#e3d3e0", "#d6e6e6", "#f0ddd2",
]
PIECE_EDGE = "#2c3e50"


def _polygon_signed_area(poly: List[Tuple[float, float]]) -> float:
    a = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        a += x1 * y2 - x2 * y1
    return a / 2


def _polygon_centroid(poly: List[Tuple[float, float]]) -> Tuple[float, float]:
    area = _polygon_signed_area(poly)
    if abs(area) < 1e-9:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return sum(xs) / len(xs), sum(ys) / len(ys)
    cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return cx / (6 * area), cy / (6 * area)


def _piece_total_area(polygons_mm: List[List[Tuple[float, float]]]) -> float:
    """Área real de la pieza (exterior menos huecos), no la de su bbox."""
    if not polygons_mm:
        return 0.0
    areas = sorted((abs(_polygon_signed_area(p)), p) for p in polygons_mm)
    outer_area = areas[-1][0]
    holes_area = sum(a for a, _ in areas[:-1])
    return max(0.0, outer_area - holes_area)


def _piece_to_path(polygons_mm: List[List[Tuple[float, float]]]) -> MplPath:
    """Arma un Path compuesto (exterior + huecos) para que matplotlib
    dibuje los huecos reales de la letra (el aro de una "O", etc.) en vez
    de rellenarla sólida. matplotlib rellena por regla nonzero: el
    contorno más grande se deja en un sentido y el resto (huecos) se
    invierte al sentido contrario — así se ve el hueco sin importar en qué
    orden venían los puntos en el PDF original."""
    if len(polygons_mm) == 1:
        ordered = polygons_mm
    else:
        by_area = sorted(((abs(_polygon_signed_area(p)), p) for p in polygons_mm), key=lambda t: -t[0])
        outer_positive = _polygon_signed_area(by_area[0][1]) > 0
        ordered = []
        for i, (_, p) in enumerate(by_area):
            want_positive = outer_positive if i == 0 else not outer_positive
            is_positive = _polygon_signed_area(p) > 0
            ordered.append(p if want_positive == is_positive else list(reversed(p)))

    verts: List[Tuple[float, float]] = []
    codes: List[int] = []
    for poly in ordered:
        verts.append(poly[0])
        codes.append(MplPath.MOVETO)
        verts.extend(poly[1:])
        codes.extend([MplPath.LINETO] * (len(poly) - 1))
        verts.append(poly[0])
        codes.append(MplPath.CLOSEPOLY)
    return MplPath(verts, codes)


def _draw_sheet_page(pdf: "PdfPages", placements: List[Dict[str, Any]], sheet_w_mm: float,
                      sheet_h_mm: float, gap_mm: float, material_label: str,
                      sheet_i: int, total_sheets: int) -> None:
    """Dibuja UNA página (una hoja) dentro de un PdfPages ya abierto —
    separado de render_nesting_pdf para poder juntar varios materiales
    (cada quien con su propio tamaño de hoja) en un solo PDF."""
    sheet_area_mm2 = sheet_w_mm * sheet_h_mm
    margin_mm = 20.0
    header_mm = 16.0

    # La hoja real mide hasta 180x240cm (70x94 pulgadas) — una página de
    # PDF a ese tamaño 1:1 es inmanejable en pantalla o en una impresora de
    # oficina, y además hace que el texto (que se mide en puntos, tamaño
    # FÍSICO fijo) se vea como puntitos ilegibles. Se dibuja todo en
    # coordenadas reales (mm) como siempre, pero la PÁGINA se encoge a un
    # tamaño de plano normal — eso también agranda el texto/líneas en
    # proporción, ya que su tamaño en puntos no cambia con el encogido.
    MAX_PAGE_IN = 17.0
    full_w_in = (sheet_w_mm + margin_mm * 2) * MM_TO_IN
    full_h_in = (sheet_h_mm + margin_mm * 2 + header_mm) * MM_TO_IN
    shrink = min(1.0, MAX_PAGE_IN / max(full_w_in, full_h_in))

    fig, ax = plt.subplots(figsize=(full_w_in * shrink, full_h_in * shrink))

    used_area_mm2 = sum(_piece_total_area(p["polygons_mm"]) for p in placements)
    utilization_pct = 100.0 * used_area_mm2 / sheet_area_mm2 if sheet_area_mm2 else 0.0

    fig.suptitle(
        f"{material_label}  —  Hoja {sheet_i + 1}/{total_sheets}  —  "
        f"{sheet_w_mm / 10:.0f}×{sheet_h_mm / 10:.0f} cm  —  "
        f"{len(placements)} pza{'s' if len(placements) != 1 else ''}  —  "
        f"sep. {gap_mm:.0f} mm  —  "
        f"aprov. {utilization_pct:.0f}%",
        fontsize=11, fontweight="bold", x=0.02, y=0.99, ha="left", color="#1a1a1a",
    )
    if shrink < 0.999:
        fig.text(0.02, 0.965, "Plano de referencia, no a escala 1:1 — medidas reales indicadas en cm",
                  fontsize=7.5, style="italic", color="#666666", ha="left")

    # Cuadrícula de referencia cada 10 cm, con los ejes en cm (no
    # en mm) para que se lea como se mide en piso de producción.
    grid_step = 100.0
    for gx in np.arange(0, sheet_w_mm + 1, grid_step):
        ax.axvline(gx, color="#e0e0e0", linewidth=0.5, zorder=0)
    for gy in np.arange(0, sheet_h_mm + 1, grid_step):
        ax.axhline(gy, color="#e0e0e0", linewidth=0.5, zorder=0)

    ax.add_patch(Rectangle((0, 0), sheet_w_mm, sheet_h_mm,
                            fill=False, edgecolor="#1a1a1a", linewidth=1.6, zorder=3))

    for i, placement in enumerate(placements):
        off_x, off_y = placement["offset_x_mm"], placement["offset_y_mm"]
        polys = placement["polygons_mm"]
        shifted = [[(x + off_x, y + off_y) for x, y in poly] for poly in polys]
        path = _piece_to_path(shifted)
        color = PIECE_COLORS[i % len(PIECE_COLORS)]
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor=PIECE_EDGE,
                                linewidth=1.0, zorder=2))

        outer = max(shifted, key=lambda p: abs(_polygon_signed_area(p)))
        cx, cy = _polygon_centroid(outer)
        xs = [pt[0] for poly in shifted for pt in poly]
        ys = [pt[1] for poly in shifted for pt in poly]
        w_cm = (max(xs) - min(xs)) / 10
        h_cm = (max(ys) - min(ys)) / 10
        rot = placement.get("rotation_deg", 0) or 0
        label = f"{placement['orig_index'] + 1}"
        sublabel = f"{w_cm:.0f}×{h_cm:.0f} cm" + (f" · {rot:.0f}°" if rot else "")
        ax.text(cx, cy, label, ha="center", va="center", fontsize=11,
                fontweight="bold", color="#1a1a1a", zorder=4)
        # Offset fijo en mm (no proporcional al tamaño de la pieza,
        # que mezclaba cm con mm y casi encimaba las dos líneas):
        # suficiente para separar el número de su medida sin
        # depender de qué tan grande sea la letra.
        ax.text(cx, cy + 34, sublabel, ha="center", va="center",
                fontsize=7, color="#444444", zorder=4)

    ax.set_xlim(-margin_mm, sheet_w_mm + margin_mm)
    ax.set_ylim(sheet_h_mm + margin_mm, -margin_mm)  # y invertido: mismo sentido que el dibujo original
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, sheet_w_mm + 1, grid_step))
    ax.set_xticklabels([f"{v/10:.0f}" for v in np.arange(0, sheet_w_mm + 1, grid_step)], fontsize=7)
    ax.set_yticks(np.arange(0, sheet_h_mm + 1, grid_step))
    ax.set_yticklabels([f"{v/10:.0f}" for v in np.arange(0, sheet_h_mm + 1, grid_step)], fontsize=7)
    ax.set_xlabel("cm", fontsize=8)
    ax.set_ylabel("cm", fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.91, bottom=0.06)

    pdf.savefig(fig)
    plt.close(fig)


def render_nesting_pdf(nest_result: Dict[str, Any], out_path: str,
                        material_label: str = "Acrílico blanco") -> str:
    """Dibuja una página por hoja usada, con el contorno real de cada letra
    en su posición final, numerada y con su medida, más un resumen de
    aprovechamiento de material por hoja."""
    total_sheets = len(nest_result["sheets"])
    with PdfPages(out_path) as pdf:
        for sheet_i, placements in enumerate(nest_result["sheets"]):
            _draw_sheet_page(pdf, placements, nest_result["sheet_w_mm"], nest_result["sheet_h_mm"],
                              nest_result["gap_mm"], material_label, sheet_i, total_sheets)
    return out_path


def build_nesting_pdf_for_rows(
    cdr_pdf_path: str,
    letter_heights_lists: List[List[float]],
    out_path: str,
    material_label: str = "Acrílico blanco",
    sheet_w_mm: float = SHEET_W_MM_DEFAULT,
    sheet_h_mm: float = SHEET_H_MM_DEFAULT,
    gap_mm: float = GAP_MM_DEFAULT,
) -> Optional[str]:
    """Punto de entrada para UN material: del .cdr ya convertido a PDF, a un
    archivo PDF con el plano de corte que junta TODAS las filas de ese
    material que el usuario aplicó (p.ej. Grupo 1 y Grupo 2 del mismo
    archivo, si ambos se mandan a cortar) — no solo la primera. Cada lista
    de alturas identifica un grupo de letras distinto; los que no se
    puedan identificar se omiten (mejor nestear lo que sí se reconoce que
    descartar todo el plano). None si ninguno se pudo identificar."""
    nest_result = _nest_result_for_material(cdr_pdf_path, letter_heights_lists, sheet_w_mm, sheet_h_mm, gap_mm)
    if nest_result is None:
        return None
    return render_nesting_pdf(nest_result, out_path, material_label=material_label)


def _nest_result_for_material(
    cdr_pdf_path: str, letter_heights_lists: List[List[float]],
    sheet_w_mm: float, sheet_h_mm: float, gap_mm: float,
) -> Optional[Dict[str, Any]]:
    all_polygons: List[List[List[Tuple[float, float]]]] = []
    for letter_heights in letter_heights_lists:
        polygons = extract_group_polygons_mm(cdr_pdf_path, letter_heights)
        if polygons:
            all_polygons.extend(polygons)
    if not all_polygons:
        return None
    nest_result = nest_pieces(all_polygons, sheet_w_mm=sheet_w_mm, sheet_h_mm=sheet_h_mm, gap_mm=gap_mm)
    if not any(nest_result["sheets"]):
        return None
    return nest_result


def build_nesting_pdf_multi_material(
    cdr_pdf_path: str,
    material_jobs: List[Dict[str, Any]],
    out_path: str,
    gap_mm: float = GAP_MM_DEFAULT,
) -> Optional[str]:
    """Como build_nesting_pdf_for_rows, pero para varios materiales a la
    vez — cada uno con su propia hoja (no se pueden mezclar letras de
    materiales distintos en el mismo plano de corte), todos juntos en un
    solo PDF, una sección de páginas por material. `material_jobs` es una
    lista de {"material_label", "sheet_w_mm", "sheet_h_mm",
    "letter_heights_lists"}. None si NINGÚN material se pudo nestear."""
    results = []
    for job in material_jobs:
        nest_result = _nest_result_for_material(
            cdr_pdf_path, job["letter_heights_lists"],
            job["sheet_w_mm"], job["sheet_h_mm"], gap_mm,
        )
        if nest_result is not None:
            results.append((job["material_label"], nest_result))
    if not results:
        return None

    with PdfPages(out_path) as pdf:
        for material_label, nest_result in results:
            total_sheets = len(nest_result["sheets"])
            for sheet_i, placements in enumerate(nest_result["sheets"]):
                _draw_sheet_page(pdf, placements, nest_result["sheet_w_mm"], nest_result["sheet_h_mm"],
                                  nest_result["gap_mm"], material_label, sheet_i, total_sheets)
    return out_path
