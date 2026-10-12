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
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - ya se valida en app.py antes de llamar aquí
    fitz = None

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - se valida antes de llamar a las funciones que lo usan
    _ShapelyPolygon = None

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
            # El hueco de una letra (el aro de una "D", los dos lazos de un
            # ampersand, etc.) no siempre viene como subtrazo del MISMO
            # drawing que el contorno exterior — a veces LibreOffice/PyMuPDF
            # lo exporta como un drawing APARTE, con su propio bounding box
            # (más chico, encajado dentro del de la letra). Si solo se toma
            # el drawing que calza exacto con el rect de la letra, esos
            # huecos se pierden por completo (la letra sale sólida). Se
            # agrega aquí cualquier otro drawing cuyo rect quede enteramente
            # DENTRO del de esta letra — es del hueco de esta letra, no de
            # la vecina, porque las letras nunca se traslapan entre sí.
            items_all = list(match["items"])
            for d in drawings:
                if d is match:
                    continue
                r = d["rect"]
                if (r.x0 >= tx0 - MATCH_TOL_PT and r.x1 <= tx1 + MATCH_TOL_PT and
                        r.y0 >= ty0 - MATCH_TOL_PT and r.y1 <= ty1 + MATCH_TOL_PT):
                    items_all.extend(d["items"])
            polys_pt = _flatten_drawing_items(items_all)
            if not polys_pt:
                return None
            polys_mm = [
                [((px - tx0) * mm_per_pt, (py - ty0) * mm_per_pt) for px, py in poly]
                for poly in polys_pt
            ]
            result.append(polys_mm)
    return result


def _polygon_perimeter_mm(poly: List[Tuple[float, float]]) -> float:
    """Longitud del contorno cerrado (suma de cada lado, incluyendo el que
    cierra del último punto al primero)."""
    n = len(poly)
    total = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def piece_perimeters_mm(cdr_pdf_path: str, letter_heights: List[float]) -> Optional[List[float]]:
    """Perímetro real de cada letra del grupo que coincide con
    `letter_heights`, para canto/trimcap: a diferencia del área, un hueco
    (el aro de una "D", el ojal de una "e") SUMA su propio contorno en vez
    de restar, porque igual hay que forrarlo con canto por dentro. No hace
    falta distinguir hueco de sólido (ver _polygon_nesting_depths) — alcanza
    con sumar el perímetro de CADA subtrazo de la pieza, sea exterior o
    interior. None si no se pudo identificar/re-emparejar el grupo (mismo
    caso que extract_group_polygons_mm)."""
    pieces = extract_group_polygons_mm(cdr_pdf_path, letter_heights)
    if pieces is None:
        return None
    return [sum(_polygon_perimeter_mm(poly) for poly in polys) for polys in pieces]


LED_THICK_STROKE_MM = 200.0  # grosor de trazo (ancho de la "cinta" de la letra) a partir del cual no alcanza una sola vuelta de LED pegada al borde
LED_THICK_STROKE_FACTOR = 2.0
LED_INSET_MM = 15.0  # separación del trazo de LED al borde real de la letra
LED_WIDTH_RES_MM = 2.0  # resolución de la grilla para medir el grosor por distance transform

# Parámetros de módulo real (no LEDs individuales sueltos sobre la tira):
# cada módulo mide 6.5cm (medida fija del módulo físico); la SEPARACIÓN
# entre un módulo y el siguiente (el hueco entre bordes, no centro a
# centro) se ajusta para que en 1m de trazo caigan exactamente 7 módulos
# — igual que arma LED Wizard, confirmado contra la ficha de referencia
# del cliente.
LED_MODULE_LEN_MM = 65.0
LED_MODULE_PITCH_MM = 1000.0 / 7.0  # paso centro a centro real: 7 módulos exactos por metro
LED_MODULE_GAP_MM = LED_MODULE_PITCH_MM - LED_MODULE_LEN_MM  # ≈ 7.79cm de separación entre módulos
LEDS_PER_MODULE = 1


def _rasterize_piece_solid(polys: List[List[Tuple[float, float]]], depths: List[int],
                            res_mm: float) -> Optional[np.ndarray]:
    """Rasteriza una pieza respetando huecos (a diferencia de
    _rasterize_polygons, que para el nesting trata todo como sólido): cada
    contorno se pinta 1 (sólido) o 0 (hueco) según su profundidad de
    anidado, en orden de profundidad CRECIENTE para que una isla sólida
    adentro de un hueco (p.ej. el punto de una "i" encajado en un hueco más
    grande) quede pintada por encima del hueco que la contiene."""
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    if not xs:
        return None
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    w = max(1, math.ceil((max_x - min_x) / res_mm))
    h = max(1, math.ceil((max_y - min_y) / res_mm))
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in sorted(range(len(polys)), key=lambda idx: depths[idx]):
        cv_poly = np.array([[(x - min_x) / res_mm, (y - min_y) / res_mm] for x, y in polys[i]], dtype=np.int32)
        cv2.fillPoly(mask, [cv_poly], 1 if depths[i] % 2 == 0 else 0)
    return mask


def _piece_stroke_width_mm(polys: List[List[Tuple[float, float]]], depths: Optional[List[int]] = None,
                            res_mm: float = LED_WIDTH_RES_MM) -> float:
    """Grosor TÍPICO del trazo de una pieza, por distance transform: para
    cada píxel sólido, su distancia al borde más cercano; el grosor local
    en ese punto es 2x esa distancia. Se usa el PROMEDIO de esas distancias
    (no el máximo) porque:
      - el máximo se dispara en uniones/esquinas (donde dos trazos se
        juntan, p.ej. el ápice de una "M") sin que el trazo en sí sea
        grueso ahí — con el máximo, CUALQUIER letra normal con un remate o
        una unión ancha saldría "gruesa".
      - 2*área/perímetro (la fórmula anterior) se rompe en piezas
        compuestas por varias partes muy distintas (p.ej. el logo circular
        de un aro delgado + texto suelto adentro, que extrae como UNA sola
        "pieza" con 30+ subtrazos): el área del texto se suma a la del aro
        pero su perímetro no crece proporcional, y el resultado sale muy
        por encima del grosor real de cualquier trazo individual.
    El promedio por distance transform mide el grosor LOCAL de verdad,
    pieza compuesta o no — queda algo por debajo de una medida de calibre
    tomada a mano en un tramo recto (una letra como la M se afila hacia las
    puntas, y esos píxeles más angostos bajan el promedio un poco), pero es
    la aproximación automática más cercana sin entrar a un análisis de eje
    medio/esqueleto; para la decisión de doblar a partir de 20 cm el margen
    de error no cambia el resultado salvo en piezas que ya estén
    prácticamente en el límite."""
    if not polys:
        return 0.0
    if depths is None:
        depths = _polygon_nesting_depths(polys)
    mask = _rasterize_piece_solid(polys, depths, res_mm)
    if mask is None or not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    vals = dist[mask.astype(bool)]
    return float(np.mean(vals)) * 2.0 * res_mm


def _zhang_suen_thin(mask: np.ndarray) -> np.ndarray:
    """Esqueleto de 1 píxel de ancho por el algoritmo de Zhang-Suen
    (vectorizado con numpy, sin dependencias nuevas) — el clásico de
    erosión/apertura iterativa (cv2.erode + cv2.dilate en loop) deja la
    letra hecha pedazos: decenas de fragmentos de 1 píxel sueltos en vez de
    una línea conectada (se verificó contando componentes conexas). Zhang-
    Suen da un esqueleto limpio y en una sola pieza por cada parte sólida
    real de la letra."""
    img = (mask > 0).astype(np.uint8)
    changed = True
    while changed:
        changed = False
        for step in (1, 2):
            padded = np.pad(img, 1, mode="constant")
            p2 = padded[0:-2, 1:-1]
            p3 = padded[0:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, 0:-2]
            p8 = padded[1:-1, 0:-2]
            p9 = padded[0:-2, 0:-2]
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            a = np.zeros_like(img)
            for k in range(8):
                a += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)
            cond_b = (b >= 2) & (b <= 6)
            cond_a = (a == 1)
            if step == 1:
                cond_extra = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond_extra = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            to_remove = (img == 1) & cond_b & cond_a & cond_extra
            if to_remove.any():
                img[to_remove] = 0
                changed = True
    return img


def _skeleton_dfs_segments(adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]]
                            ) -> List[List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """Recorre el grafo de píxeles del esqueleto (8-conectividad) por
    componente conexa, con un DFS iterativo (pila explícita, no recursión —
    una letra alta puede dar una cadena de cientos de píxeles seguidos, y
    eso se pasa del límite de recursión de Python) que visita cada ARISTA
    una sola vez — si el componente tiene un ciclo (el aro de una "e" o una
    "o"), la arista que cierra el ciclo se cuenta pero no se vuelve a
    explorar desde ahí, así no entra en loop infinito. Devuelve, por
    componente con 2+ píxeles (se descartan píxeles sueltos, ruido del
    rasterizado), la lista de aristas (pares de píxeles vecinos) en el
    orden que las visitó el DFS."""
    visited_global: set = set()
    components: List[List[Tuple[Tuple[int, int], Tuple[int, int]]]] = []
    for start in list(adjacency.keys()):
        if start in visited_global:
            continue
        visited_edges: set = set()
        visited_nodes = {start}
        segs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        stack = [(start, iter(adjacency[start]))]
        while stack:
            node, it = stack[-1]
            advanced = False
            for nb in it:
                edge_key = (node, nb) if node <= nb else (nb, node)
                if edge_key in visited_edges:
                    continue
                visited_edges.add(edge_key)
                segs.append((node, nb))
                if nb not in visited_nodes:
                    visited_nodes.add(nb)
                    stack.append((nb, iter(adjacency[nb])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
        visited_global |= visited_nodes
        if len(visited_nodes) >= 2:
            components.append(segs)
    return components


def _skeleton_mst_adjacency(
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Reduce el grafo de píxeles del esqueleto a su árbol de expansión
    mínima (Kruskal + union-find, sin dependencias nuevas): Zhang-Suen dado
    un trazo con el borde ligeramente dentado (el "escalonado" normal de
    aproximar una curva/diagonal con píxeles) no siempre lo adelgaza a 1
    píxel limpio — deja tramos cortos con DOS caminos casi paralelos que
    se separan y se vuelven a juntar enseguida (un mini-ciclo), no un
    verdadero trazo doble. Eso no se ve como una rama real al dibujar (es
    un zigzag apretado), pero SÍ se suma de más al total de metros/módulos
    porque el recorrido cuenta las dos vueltas del mini-ciclo. El MST
    elimina exactamente esas aristas redundantes (y de paso dos puntos
    conectados por dos caminos nunca necesitan ambos para un LED: un trazo
    de LED siempre es un camino, nunca dos en paralelo) sin tocar las
    ramificaciones reales de la letra (el ápice de una "M", la cruz de una
    "f"), que no son ciclos — y un hueco real (el aro de una "e") se corta
    en un solo punto arbitrario, que es justo como se instala un LED ahí:
    una vuelta casi completa, no un lazo perfectamente cerrado."""
    edges: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = []
    seen: set = set()
    for node, neighbors in adjacency.items():
        for nb in neighbors:
            key = (node, nb) if node <= nb else (nb, node)
            if key in seen:
                continue
            seen.add(key)
            edges.append((math.hypot(nb[0] - node[0], nb[1] - node[1]), key[0], key[1]))
    edges.sort(key=lambda e: e[0])

    parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find(a: Tuple[int, int]) -> Tuple[int, int]:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    mst_adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {n: [] for n in adjacency}
    for _, a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        mst_adjacency[a].append(b)
        mst_adjacency[b].append(a)
    return mst_adjacency


def _piece_skeleton_mm(
    polys: List[List[Tuple[float, float]]], depths: List[int], res_mm: float = LED_WIDTH_RES_MM,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], float]:
    """Eje medio (centerline) de una pieza, igual que la foto de referencia
    del cliente: UNA línea por el centro de cada trazo, que se ramifica
    donde la letra se ramifica de verdad (el ápice de una "M", la cruz de
    una "f") — no una vuelta pegada al borde (eso se ve como dos líneas
    paralelas en un trazo angosto, que es justo la confusión que se quiere
    evitar). Devuelve (segmentos en mm, longitud total en mm)."""
    mask = _rasterize_piece_solid(polys, depths, res_mm)
    if mask is None or not mask.any():
        return [], 0.0
    skel = _zhang_suen_thin(mask)
    ys, xs = np.where(skel > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))
    if len(pts) < 2:
        return [], 0.0
    adjacency = {
        (x, y): [
            (x + dx, y + dy)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if (dx or dy) and (x + dx, y + dy) in pts
        ]
        for x, y in pts
    }
    adjacency = _skeleton_mst_adjacency(adjacency)
    all_xs = [p[0] for poly in polys for p in poly]
    all_ys = [p[1] for poly in polys for p in poly]
    min_x, min_y = min(all_xs), min(all_ys)

    segments_mm: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    total_mm = 0.0
    for comp_segs in _skeleton_dfs_segments(adjacency):
        for (x1, y1), (x2, y2) in comp_segs:
            p1 = (min_x + x1 * res_mm, min_y + y1 * res_mm)
            p2 = (min_x + x2 * res_mm, min_y + y2 * res_mm)
            segments_mm.append((p1, p2))
            total_mm += math.hypot(x2 - x1, y2 - y1) * res_mm
    return segments_mm, total_mm


def _piece_led_stats(polys: List[List[Tuple[float, float]]],
                      depths: Optional[List[int]] = None) -> Dict[str, Any]:
    """Perímetro, grosor estimado, factor (1 o 2) y la longitud/trazo real
    de LED de una pieza — ver _piece_stroke_width_mm y _piece_skeleton_mm
    para la explicación de cada fórmula. Función compartida entre
    piece_led_meters (solo números) y _draw_led_diagram_page (el dibujo),
    para no calcular nada dos veces con fórmulas que se puedan ir
    desalineando.

    Trazo SENCILLO (grosor < 20cm, el caso normal): el LED va por el eje
    medio de la letra, una sola línea que se ramifica donde la letra se
    ramifica — ver _piece_skeleton_mm. Trazo GRUESO (≥ 20cm): una sola
    línea por el centro ya no alcanza a iluminar parejo un canal tan ancho,
    así que se usan dos vueltas cerca de cada borde (el inset de shapely,
    ver _piece_led_segments_inset)."""
    if depths is None:
        depths = _polygon_nesting_depths(polys)
    perimeter_mm = sum(_polygon_perimeter_mm(poly) for poly in polys)
    area_mm2 = _piece_total_area(polys)
    width_mm = _piece_stroke_width_mm(polys, depths=depths)
    factor = LED_THICK_STROKE_FACTOR if width_mm >= LED_THICK_STROKE_MM else 1.0
    if factor > 1:
        mode = "inset"
        led_length_mm = perimeter_mm * factor
    else:
        mode = "skeleton"
        _, led_length_mm = _piece_skeleton_mm(polys, depths)
    return {
        "perimeter_mm": perimeter_mm, "area_mm2": area_mm2, "width_mm": width_mm,
        "factor": factor, "mode": mode, "led_length_mm": led_length_mm,
    }


def _led_module_count(led_length_mm: float, module_pitch_mm: float = LED_MODULE_PITCH_MM) -> int:
    """Cuántos módulos caben en un trazo de `led_length_mm`, a un paso
    centro a centro de `module_pitch_mm` (largo del módulo + separación
    entre módulos, igual que LED Wizard) — no son LEDs sueltos a densidad
    continua, son módulos de varios diodos cada uno (ver LEDS_PER_MODULE).
    Redondeo normal (no hacia arriba): a este paso, un trazo un poco más
    corto que un paso completo de sobra sigue necesitando solo el módulo
    que ya tenía, no uno de más."""
    if led_length_mm <= 0:
        return 0
    return max(1, round(led_length_mm / module_pitch_mm))


def piece_led_meters(
    cdr_pdf_path: str, letter_heights: List[float],
    module_pitch_mm: float = LED_MODULE_PITCH_MM,
    leds_per_module: int = LEDS_PER_MODULE,
) -> Optional[List[Dict[str, float]]]:
    """Metros lineales de LED y cantidad de MÓDULOS (no LEDs sueltos a
    densidad continua) por letra del grupo que coincide con
    `letter_heights`: por defecto, el eje medio (centerline) real de cada
    letra — una sola línea que se ramifica donde la letra se ramifica (ver
    _piece_skeleton_mm), igual que un diseño real de módulos LED. Si el
    trazo de la letra es grueso (≥ LED_THICK_STROKE_MM) una sola línea por
    el centro no alcanza a iluminar parejo un canal tan ancho, así que se
    usan dos vueltas cerca de cada borde en su lugar (ver _piece_led_stats).

    La cantidad de LEDs sale de módulos reales (6.5cm cada uno) acomodados
    a un paso centro a centro de `module_pitch_mm` (módulo + separación) a
    lo largo del trazo — igual que arma LED Wizard, no "metros × densidad
    continua".

    El grosor real de un trazo no es una medida directa de la geometría (un
    polígono no "sabe" su propio ancho) — se aproxima con distance
    transform (ver _piece_stroke_width_mm).

    None si no se pudo identificar/re-emparejar el grupo (mismo caso que
    extract_group_polygons_mm)."""
    pieces = extract_group_polygons_mm(cdr_pdf_path, letter_heights)
    if pieces is None:
        return None
    results = []
    for polys in pieces:
        stats = _piece_led_stats(polys)
        led_meters = stats["led_length_mm"] / 1000.0
        num_modules = _led_module_count(stats["led_length_mm"], module_pitch_mm)
        results.append({
            "perimeter_mm": stats["perimeter_mm"],
            "width_mm": stats["width_mm"],
            "factor": stats["factor"],
            "led_meters": led_meters,
            "num_modules": num_modules,
            "led_count": num_modules * leds_per_module,
        })
    return results


def _piece_led_segments_inset(
    polys: List[List[Tuple[float, float]]], depths: List[int], width_mm: float,
    inset_mm: float = LED_INSET_MM,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Dos vueltas cerca de cada borde (trazo GRUESO, ≥20cm) — usa shapely
    porque un offset hacia adentro correcto en esquinas cóncavas (como el
    interior de una "e") no es un simple desplazamiento perpendicular por
    lado, se puede auto-intersectar. Devuelve una lista plana de segmentos
    (no polilíneas), consistente con _piece_skeleton_mm."""
    if _ShapelyPolygon is None or not polys:
        return []
    insets = [inset_mm, max(inset_mm, width_mm - inset_mm)]
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for poly, depth in zip(polys, depths):
        shp = _ShapelyPolygon(poly)
        if not shp.is_valid:
            shp = shp.buffer(0)
        if shp.is_empty:
            continue
        # Depth par: contorno exterior (o isla sólida) — el sólido queda
        # ADENTRO, así que se encoge. Depth impar: borde de un hueco — el
        # sólido queda AFUERA, así que se crece (buffer positivo).
        sign = -1.0 if depth % 2 == 0 else 1.0
        for ins in insets:
            offset = shp.buffer(sign * ins, join_style="mitre")
            if offset.is_empty:
                continue
            geoms = offset.geoms if offset.geom_type == "MultiPolygon" else [offset]
            for g in geoms:
                if g.exterior is None:
                    continue
                ring = list(g.exterior.coords)
                segments.extend(zip(ring, ring[1:]))
    return segments


def _piece_led_segments(
    polys: List[List[Tuple[float, float]]], depths: Optional[List[int]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Segmentos del trazo de LED de una pieza — eje medio si es sencillo,
    dos vueltas cerca del borde si es grueso (ver _piece_led_stats)."""
    if not polys:
        return []
    if depths is None:
        depths = _polygon_nesting_depths(polys)
    if stats is None:
        stats = _piece_led_stats(polys, depths=depths)
    if stats["mode"] == "inset":
        return _piece_led_segments_inset(polys, depths, stats["width_mm"])
    segments, _ = _piece_skeleton_mm(polys, depths)
    return segments


def _sample_led_points(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], spacing_mm: float,
    min_euclid_mm: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """Puntos a lo largo de una lista de segmentos (no hace falta que
    formen una sola polilínea continua — el eje medio de una letra
    ramificada no lo es, ver _skeleton_dfs_segments), cada `spacing_mm` de
    distancia acumulada — para marcar dónde cae cada módulo sobre el trazo.

    El contador de distancia NO se reinicia al pasar de una rama a otra
    (recorre la lista de segmentos tal cual la entrega el DFS), así que
    justo en una unión/esquina (donde el DFS termina una rama y arranca la
    siguiente) pueden caer dos marcas de "ya pasaron spacing_mm" casi en el
    mismo punto físico aunque estén lejos en la lista — se ve como dos o
    tres módulos encimados en cada esquina de la letra. Por eso se filtra
    después por distancia EUCLIDIANA real entre puntos consecutivos (no por
    posición en la lista): se descarta cualquier punto que haya quedado a
    menos de `min_euclid_mm` del último punto que sí se conservó."""
    pts: List[Tuple[float, float]] = []
    if spacing_mm <= 0:
        return pts
    accum = 0.0
    next_mark = 0.0
    for (x1, y1), (x2, y2) in segments:
        seg_len = math.hypot(x2 - x1, y2 - y1)
        while seg_len > 0 and next_mark <= accum + seg_len:
            t = (next_mark - accum) / seg_len
            pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
            next_mark += spacing_mm
        accum += seg_len

    if min_euclid_mm is None:
        min_euclid_mm = spacing_mm * 0.6
    kept: List[Tuple[float, float]] = []
    for p in pts:
        if not kept or all(math.hypot(p[0] - k[0], p[1] - k[1]) >= min_euclid_mm for k in kept):
            kept.append(p)
    return kept


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


def _compact_sheets(sheets, occupied_grids, variants_by_idx, new_grid_fn,
                     res_mm: float, max_passes: int) -> None:
    """Mejora local después del acomodo inicial (todo en sitio, sobre
    `sheets`/`occupied_grids`): quita cada pieza ya colocada y la vuelve a
    insertar —probando todas sus rotaciones, en cualquier hoja que ya esté
    abierta, no solo la suya— ahora que sabe dónde quedó todo lo demás. El
    acomodo inicial coloca cada pieza una sola vez sin poder corregirla
    después, así que una pieza grande puesta primero puede dejar un hueco
    que ninguna de las que siguen aprovecha aunque sí quepan ahí. No abre
    hojas nuevas (si no mejora en las que ya existen, se queda donde
    estaba) — para en cuanto una pasada completa no mueve nada (convergió)
    o se llega a `max_passes`."""
    location: Dict[int, int] = {}
    for sheet_i, sheet in enumerate(sheets):
        for pl in sheet:
            location[pl["orig_index"]] = sheet_i

    order = list(variants_by_idx.keys())

    for _ in range(max_passes):
        moved_any = False
        for orig_idx in order:
            sheet_i = location[orig_idx]
            sheet = sheets[sheet_i]
            pos_in_sheet = next(i for i, pl in enumerate(sheet) if pl["orig_index"] == orig_idx)
            placement = sheet.pop(pos_in_sheet)
            old_key = (sheet_i, placement["_cy"], placement["_cx"])

            # Reconstruir la hoja de origen SIN esta pieza — libera el halo
            # de separación que dilataba contra sus vecinas.
            occupied_grids[sheet_i] = new_grid_fn()
            for pl in sheet:
                cy, cx = pl["_cy"], pl["_cx"]
                mask = pl["_mask"]
                ph, pw = mask.shape
                occupied_grids[sheet_i][cy:cy + ph, cx:cx + pw] |= mask

            # Mejor posición para esta pieza en CUALQUIER hoja ya abierta
            # (puede terminar en una hoja distinta si ahí le queda mejor).
            best = None  # (sheet_i, cy, cx, ang, rp, mask, min_x, min_y)
            for s_i in range(len(sheets)):
                for ang, rp, mask, min_x, min_y in variants_by_idx[orig_idx]:
                    pos = _find_first_fit(occupied_grids[s_i], mask)
                    if pos is None:
                        continue
                    key = (s_i, pos[0], pos[1])
                    if best is None or key < (best[0], best[1], best[2]):
                        best = (s_i, pos[0], pos[1], ang, rp, mask, min_x, min_y)

            new_key = (best[0], best[1], best[2]) if best is not None else None
            if best is not None and new_key < old_key:
                s_i, cy, cx, ang, rp, mask, min_x, min_y = best
                ph, pw = mask.shape
                occupied_grids[s_i][cy:cy + ph, cx:cx + pw] |= mask
                sheets[s_i].append({
                    "orig_index": orig_idx,
                    "polygons_mm": rp,
                    "rotation_deg": ang,
                    "sheet_index": s_i,
                    "offset_x_mm": cx * res_mm - min_x,
                    "offset_y_mm": cy * res_mm - min_y,
                    "_cy": cy, "_cx": cx, "_mask": mask,
                })
                location[orig_idx] = s_i
                moved_any = True
            else:
                # Sin mejora (o búsqueda sin resultado, lo que no debería
                # pasar ya que antes cabía): se regresa tal cual estaba.
                ph, pw = placement["_mask"].shape
                occupied_grids[sheet_i][placement["_cy"]:placement["_cy"] + ph,
                                         placement["_cx"]:placement["_cx"] + pw] |= placement["_mask"]
                sheet.append(placement)
                location[orig_idx] = sheet_i

        if not moved_any:
            break


def nest_pieces(
    pieces_polygons_mm: List[List[List[Tuple[float, float]]]],
    sheet_w_mm: float = SHEET_W_MM_DEFAULT,
    sheet_h_mm: float = SHEET_H_MM_DEFAULT,
    gap_mm: float = GAP_MM_DEFAULT,
    res_mm: float = GRID_RES_MM,
    rotation_step_deg: float = ROTATION_STEP_DEG_DEFAULT,
    compact_passes: int = 0,
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
    # Variantes (una por rotación) de cada pieza, cacheadas por su índice
    # original — _compact_sheets las reutiliza para no tener que volver a
    # rotar/rasterizar/dilatar cada pieza otra vez en cada pasada.
    variants_by_idx: Dict[int, list] = {}

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
        variants_by_idx[orig_idx] = variants

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
                    # Campos privados (se quitan antes de devolver el
                    # resultado) — solo para que _compact_sheets pueda
                    # quitar/reinsertar esta pieza sin tener que volver a
                    # rasterizarla cada vez.
                    "_cy": cy, "_cx": cx, "_mask": test_mask,
                })
                placed = True
            else:
                sheet_i += 1

    if compact_passes > 0:
        _compact_sheets(sheets, occupied_grids, variants_by_idx, _new_grid, res_mm, compact_passes)

    # Quitar las hojas que la compactación haya dejado vacías (una pieza
    # que se movió a otra hoja no deja "hueco" en el conteo final) y
    # renumerar sheet_index para que no queden huecos en la secuencia.
    sheets = [s for s in sheets if s]
    for new_i, sheet in enumerate(sheets):
        for placement in sheet:
            placement["sheet_index"] = new_i
            placement.pop("_cy", None)
            placement.pop("_cx", None)
            placement.pop("_mask", None)

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
from matplotlib.collections import LineCollection
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


def _point_in_polygon(pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
    """Ray casting estándar: ¿el punto cae dentro del polígono simple?"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


def _polygon_nesting_depths(polygons_mm: List[List[Tuple[float, float]]]) -> List[int]:
    """Para cada polígono, cuenta cuántos de los OTROS polígonos lo
    contienen (por su centroide) — la profundidad de anidado determina si
    es relleno sólido (profundidad par: el contorno exterior, o una "isla"
    sólida dentro de un hueco, como el punto de la "i") o un hueco
    (profundidad impar: el aro de una "O", los dos ojos de una "B"). A
    diferencia de "el de mayor área es el exterior y todo lo demás es
    hueco", esto NO falla cuando una pieza trae partes sueltas que no están
    anidadas en nada (p.ej. el punto y el palo de una "i", o un logo con
    piezas separadas) — esas quedan todas en profundidad 0 (sólidas) en vez
    de que la más chica se reste por error de la más grande."""
    centroids = [_polygon_centroid(p) for p in polygons_mm]
    areas = [abs(_polygon_signed_area(p)) for p in polygons_mm]
    depths = []
    for i in range(len(polygons_mm)):
        depth = 0
        for j, poly_j in enumerate(polygons_mm):
            # Sólo cuenta como "contenedor" un polígono de más área: dos
            # contornos concéntricos (el aro exterior y el interior de una
            # "O") quedan con el centro casi en el mismo punto, así que
            # cada uno "contiene" geométricamente el centroide del otro —
            # sin este desempate por área, ese caso normal (el más común)
            # saldría con profundidad [1,1] en vez de [0,1].
            if i != j and areas[j] > areas[i] and _point_in_polygon(centroids[i], poly_j):
                depth += 1
        depths.append(depth)
    return depths


def _piece_total_area(polygons_mm: List[List[Tuple[float, float]]]) -> float:
    """Área real de la pieza (sólidos menos huecos), no la de su bbox —
    ver _polygon_nesting_depths."""
    if not polygons_mm:
        return 0.0
    if len(polygons_mm) == 1:
        return abs(_polygon_signed_area(polygons_mm[0]))
    depths = _polygon_nesting_depths(polygons_mm)
    total = 0.0
    for depth, poly in zip(depths, polygons_mm):
        area = abs(_polygon_signed_area(poly))
        total += area if depth % 2 == 0 else -area
    return max(0.0, total)


def _piece_to_path(polygons_mm: List[List[Tuple[float, float]]]) -> MplPath:
    """Arma un Path compuesto (sólidos + huecos, con cualquier cantidad de
    partes sueltas o islas anidadas) para que matplotlib dibuje la pieza
    real con regla nonzero. Cada contorno se orienta según su PROFUNDIDAD
    de anidado (ver _polygon_nesting_depths) en vez de asumir que el
    polígono de mayor área es el único exterior — así una letra con partes
    sueltas (p.ej. "i") o un logo con piezas separadas no sale con un
    "mordisco" donde la parte chica se restó por error de la grande."""
    if len(polygons_mm) == 1:
        ordered = polygons_mm
    else:
        depths = _polygon_nesting_depths(polygons_mm)
        ordered = []
        for depth, p in zip(depths, polygons_mm):
            want_positive = (depth % 2 == 0)
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

    # Cuántos cuartos de hoja hay que COMPRAR para este acomodo (1 cuarto ==
    # 1/4 de una hoja completa), no qué tan lleno queda cada uno. El
    # material se compra en unidades enteras de cuarto, así que lo que
    # importa es el RECTÁNGULO que en verdad ocupa el acomodo sobre la hoja
    # (de orilla a orilla de las piezas, sin importar el hueco propio de
    # cada letra) redondeado HACIA ARRIBA al siguiente cuarto — igual que en
    # la realidad no puedes comprar "3.66 cuartos", compras 4 (la hoja
    # completa) si el acomodo ya no cabe en 3.
    footprint_xs: List[float] = []
    footprint_ys: List[float] = []
    for p in placements:
        off_x, off_y = p["offset_x_mm"], p["offset_y_mm"]
        for poly in p["polygons_mm"]:
            for x, y in poly:
                footprint_xs.append(x + off_x)
                footprint_ys.append(y + off_y)
    if footprint_xs and sheet_area_mm2:
        footprint_w = max(footprint_xs) - min(footprint_xs)
        footprint_h = max(footprint_ys) - min(footprint_ys)
        quarters_raw = 4.0 * (footprint_w * footprint_h) / sheet_area_mm2
        quarters_used = min(4, max(1, math.ceil(quarters_raw - 1e-6)))
    else:
        quarters_used = 0

    fig.suptitle(
        f"{material_label}  —  Hoja {sheet_i + 1}/{total_sheets}  —  "
        f"{sheet_w_mm / 10:.0f}×{sheet_h_mm / 10:.0f} cm  —  "
        f"{len(placements)} pza{'s' if len(placements) != 1 else ''}  —  "
        f"sep. {gap_mm:.0f} mm  —  "
        f"{quarters_used}/4 cuartos",
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


def _draw_led_diagram_page(pdf: "PdfPages", polys_list: List[List[List[Tuple[float, float]]]],
                            group_label: str, module_pitch_mm: float = LED_MODULE_PITCH_MM,
                            leds_per_module: int = LEDS_PER_MODULE) -> None:
    """Una página con las letras de un grupo acomodadas en fila (como se
    leen, alineadas por la base) — no es un plano de corte, es un
    DIAGRAMA DE REFERENCIA de dónde va el LED en cada letra: trazo punteado
    (eje medio, doble vuelta cerca del borde en trazos gruesos) con un
    punto por cada MÓDULO real (varios diodos juntos, no LEDs sueltos a
    densidad continua — ver piece_led_meters)."""
    PIECE_GAP_MM = 40.0
    margin_mm = 20.0
    # Fijo relativamente grande: a diferencia del plano de hoja (más o
    # menos cuadrado), este diagrama suele quedar MUY ancho y bajo (una
    # fila de letras), así que el shrink-to-fit por ancho aplasta mucho la
    # altura real en pulgadas — un header_mm chico (como en _draw_sheet_page)
    # deja un encabezado de unos pocos puntos de alto y el título se
    # encima con el subtítulo. Las posiciones de texto se calculan abajo a
    # partir de pulgadas físicas reales, no de fracciones fijas, para que
    # esto no dependa de qué tan ancho salga el diagrama.
    header_mm = 130.0
    spacing_mm = module_pitch_mm

    widths_mm, heights_mm = [], []
    for polys in polys_list:
        xs = [x for poly in polys for x, y in poly]
        ys = [y for poly in polys for x, y in poly]
        widths_mm.append(max(xs) - min(xs) if xs else 0.0)
        heights_mm.append(max(ys) - min(ys) if ys else 0.0)
    max_h = max(heights_mm) if heights_mm else 0.0
    total_w = sum(widths_mm) + PIECE_GAP_MM * max(0, len(polys_list) - 1)

    MAX_PAGE_IN = 17.0
    full_w_in = (total_w + margin_mm * 2) * MM_TO_IN
    full_h_in = (max_h + margin_mm * 2 + header_mm) * MM_TO_IN
    shrink = min(1.0, MAX_PAGE_IN / max(full_w_in, full_h_in, 1e-6))
    fig_h_in = full_h_in * shrink
    fig, ax = plt.subplots(figsize=(full_w_in * shrink, fig_h_in))

    total_led_m = 0.0
    total_led_count = 0
    x_cursor = 0.0
    for i, polys in enumerate(polys_list):
        w, h = widths_mm[i], heights_mm[i]
        y_off = max_h - h  # alinea por la base, como se lee la palabra
        shifted = [[(x + x_cursor, y + y_off) for x, y in poly] for poly in polys]

        path = _piece_to_path(shifted)
        color = PIECE_COLORS[i % len(PIECE_COLORS)]
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor=PIECE_EDGE, linewidth=1.0, zorder=2))

        depths = _polygon_nesting_depths(polys)
        stats = _piece_led_stats(polys, depths=depths)
        led_m = stats["led_length_mm"] / 1000.0
        num_modules = _led_module_count(stats["led_length_mm"], module_pitch_mm)
        led_count = num_modules * leds_per_module
        total_led_m += led_m
        total_led_count += led_count

        segments = _piece_led_segments(polys, depths=depths, stats=stats)
        shifted_segments = [
            ((x1 + x_cursor, y1 + y_off), (x2 + x_cursor, y2 + y_off))
            for (x1, y1), (x2, y2) in segments
        ]
        if shifted_segments:
            ax.add_collection(LineCollection(
                shifted_segments, colors="#c0392b", linewidths=1.0, zorder=3,
                linestyles=(0, (1, 1.8)),
            ))
            pts = _sample_led_points(shifted_segments, spacing_mm)
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           s=22, color="#c0392b", zorder=4, linewidths=0.6, edgecolors="#7a2016")

        cx = x_cursor + w / 2
        mode_label = " · doble" if stats["mode"] == "inset" else ""
        ax.text(cx, max_h + 14, f"{led_m:.2f} m · {num_modules} módulos ({led_count} LEDs){mode_label}",
                ha="center", va="top", fontsize=6.5, color="#444444", zorder=5)

        x_cursor += w + PIECE_GAP_MM

    # Posiciones en fracción de figura calculadas desde pulgadas FÍSICAS
    # reales (no constantes fijas tipo 0.99/0.965): con un diagrama tan
    # ancho como alto en pulgadas terminadas, una fracción fija da un hueco
    # absurdamente chico en inches y el título se encima con el subtítulo.
    top_frac = 1.0 - (header_mm * MM_TO_IN) / full_h_in
    title_y = 1.0 - (0.22 / fig_h_in) if fig_h_in > 0 else 0.97
    sub_y = 1.0 - (0.42 / fig_h_in) if fig_h_in > 0 else 0.90

    fig.suptitle(
        f"{group_label}  —  {len(polys_list)} letra{'s' if len(polys_list) != 1 else ''}  —  "
        f"{total_led_m:.2f} m de LED  —  ~{total_led_count} LEDs",
        fontsize=11, fontweight="bold", x=0.02, y=title_y, ha="left", va="top", color="#1a1a1a",
    )
    fig.text(0.02, sub_y,
              f"Diagrama de referencia — LED por eje medio (doble vuelta cerca del borde en "
              f"trazos ≥ {LED_THICK_STROKE_MM / 10:.0f} cm), módulos de {LED_MODULE_LEN_MM / 10:.1f} cm "
              f"con {LED_MODULE_GAP_MM / 10:.0f} cm de separación entre ellos "
              f"({leds_per_module} LED{'s' if leds_per_module != 1 else ''} por módulo)",
              fontsize=7.5, style="italic", color="#666666", ha="left", va="top")

    ax.set_xlim(-margin_mm, total_w + margin_mm)
    ax.set_ylim(max_h + margin_mm, -margin_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=top_frac, bottom=0.03)

    pdf.savefig(fig)
    plt.close(fig)


def build_led_diagram_pdf(
    cdr_pdf_path: str, letter_heights_groups: List[List[float]], out_path: str,
    module_pitch_mm: float = LED_MODULE_PITCH_MM, leds_per_module: int = LEDS_PER_MODULE,
) -> Optional[str]:
    """Un PDF con un diagrama de referencia de LED (una página por grupo de
    letras aplicado) — ver _draw_led_diagram_page. `letter_heights_groups`
    es una lista de listas de alturas reales (una por fila de ILUMINACIÓN
    aplicada con letra por letra, igual que letter_heights_lists en
    build_nesting_pdf_for_rows). None si ningún grupo se pudo identificar o
    si shapely no está instalado."""
    if _ShapelyPolygon is None:
        return None
    jobs = []
    for idx, letter_heights in enumerate(letter_heights_groups):
        polys_list = extract_group_polygons_mm(cdr_pdf_path, letter_heights)
        if polys_list:
            jobs.append((f"Grupo {idx + 1}", polys_list))
    if not jobs:
        return None

    with PdfPages(out_path) as pdf:
        for group_label, polys_list in jobs:
            _draw_led_diagram_page(pdf, polys_list, group_label,
                                    module_pitch_mm=module_pitch_mm, leds_per_module=leds_per_module)
    return out_path


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
    nest_result = nest_pieces(all_polygons, sheet_w_mm=sheet_w_mm, sheet_h_mm=sheet_h_mm, gap_mm=gap_mm,
                               compact_passes=1)
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
