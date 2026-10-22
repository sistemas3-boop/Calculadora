"""
Extracción de geometría real (no cajas) de letras/piezas desde un PDF ya
convertido (LibreOffice headless) de un .cdr, más el esqueleto (eje medio)
de cada pieza.

Reutiliza el contorno REAL de cada letra para nesting/LED. No usa la
medida ya calculada por _extract_pieces_from_pdf / _extract_pieces_geometric
(esas dan alto/ancho, no el contorno). En vez de duplicar todo ese pipeline
de "Medidas: A x B mm" + cotas, este módulo:

  1. Llama a esas mismas funciones (ya probadas, en app.py) para encontrar
     el grupo de letras que corresponde a la fila que el usuario aplicó (se
     identifica por las alturas individuales, en el mismo orden
     izquierda->derecha que ya viajó al navegador como `heights_detail[]`).
  2. Vuelve a abrir el PDF, re-emparejando cada trazo del grupo por su
     bounding box exacto (misma llamada a page.get_drawings(), resultado
     determinista) para recuperar sus puntos de trayectoria reales.

Antes vivía junto con el nesting y el dibujo del plano en un solo
nesting.py; se separó en geometry.py / led_layout.py / sheet_packing.py /
nesting_pdf.py (ver auditoría de reestructuración) — nesting.py quedó como
fachada de compatibilidad que re-exporta todo esto.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - ya se valida en app.py antes de llamar aquí
    fitz = None

MATCH_TOL_PT = 0.05          # tolerancia al re-emparejar un trazo por su bounding box exacto
LED_WIDTH_RES_MM = 2.0       # resolución de la grilla para medir el grosor por distance transform (ver _piece_skeleton_mm)


# ══ 1) CONTORNO REAL DE CADA LETRA (a partir del grupo ya identificado) ═══════

def dedupe_drawings(drawings: List[dict]) -> List[dict]:
    """Quita de `page.get_drawings()` los trazos EXACTAMENTE duplicados
    (mismo bounding box, visto real en archivos de producción: algunos
    .cdr convertidos por LibreOffice exportan el mismo trazo dos veces, al
    parecer por un relleno + un trazo de borde que terminan como dos
    "drawings" independientes con el mismo rect). Un trazo real nunca
    necesita dos copias idénticas — y una letra con hueco (el aro de una
    "D", una "O" chica) se rompe feo con la copia de más: la lógica que
    junta el hueco de una letra (ver extract_group_polygons_mm) toma
    CUALQUIER otro drawing con el rect contenido en el de la letra como su
    hueco, así que una copia duplicada del contorno EXTERIOR entero (que
    por definición "cabe" dentro de sí mismo) se cuela como si fuera un
    hueco — eso duplica cada punto del contorno exterior Y voltea la
    paridad par/impar de cualquier hueco real (profundidad de anidado +1
    de más), convirtiendo huecos reales en "islas sólidas": la letra sale
    completamente rellena por dentro, sin ningún hueco, y su esqueleto
    colapsa a casi nada al adelgazar (se vio real: 0 módulos/0 metros en
    una letra con tornillos de montaje, perímetro y ancho normales)."""
    seen: set = set()
    result = []
    for d in drawings:
        r = d["rect"]
        key = (round(r.x0, 2), round(r.y0, 2), round(r.x1, 2), round(r.y1, 2))
        if key in seen:
            continue
        seen.add(key)
        result.append(d)
    return result


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


def _group_compound_subpolys(polys: List[List[Tuple[float, float]]]) -> Optional[List[List[int]]]:
    """Agrupa los subtrazos de UN solo trazo compuesto (ver
    split_compound_drawing_into_letters) en letras individuales — core
    compartido entre esa función (solo necesita los rects) y
    _split_compound_drawing_polys (necesita además qué polígonos quedan
    en cada letra, para nesting/LED). None si `polys` ya es una sola
    letra (nada que separar).

    Dos pasadas:
    1) Cada hueco (el aro de una "e", el ojal de una "a") se agrupa con el
       trazo sólido que lo contiene por simple contención de bounding box
       — dos subtrazos de la MISMA letra quedan uno dentro del otro, y dos
       letras distintas nunca se solapan así.
    2) El punto y el palo de una "i"/"j" NO están uno dentro del otro (son
       dos subtrazos disjuntos del mismo objeto) — sin esta segunda
       pasada, el punto se cuenta como su propia "letra" aparte. Se
       reconocen porque están CASI en el mismo rango horizontal (mismo
       ancho de letra) pero casi no se solapan en vertical (el punto va
       arriba, separado del palo) — a diferencia de dos letras vecinas,
       que sí comparten casi todo su rango vertical (misma altura de
       letra) pero se solapan poco o nada en horizontal."""
    if len(polys) < 2:
        return None
    boxes = []
    for poly in polys:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    tol = 0.5
    for i in range(n):
        bi = boxes[i]
        for j in range(n):
            if i == j:
                continue
            bj = boxes[j]
            if (bi[0] >= bj[0] - tol and bi[2] <= bj[2] + tol and
                    bi[1] >= bj[1] - tol and bi[3] <= bj[3] + tol):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    def group_bbox(idxs: List[int]) -> Tuple[float, float, float, float]:
        return (min(boxes[i][0] for i in idxs), min(boxes[i][1] for i in idxs),
                max(boxes[i][2] for i in idxs), max(boxes[i][3] for i in idxs))

    # Segunda pasada: punto + palo de una misma letra.
    group_ids = list(groups.keys())
    changed = True
    while changed and len(group_ids) > 1:
        changed = False
        for a in range(len(group_ids)):
            ga = group_ids[a]
            if ga not in groups:
                continue
            ba = group_bbox(groups[ga])
            for b in range(a + 1, len(group_ids)):
                gb = group_ids[b]
                if gb not in groups:
                    continue
                bb = group_bbox(groups[gb])
                x_overlap = min(ba[2], bb[2]) - max(ba[0], bb[0])
                narrower_w = min(ba[2] - ba[0], bb[2] - bb[0])
                if narrower_w <= 0 or x_overlap / narrower_w < 0.5:
                    continue
                y_overlap = min(ba[3], bb[3]) - max(ba[1], bb[1])
                taller_h = max(ba[3] - ba[1], bb[3] - bb[1])
                if taller_h > 0 and y_overlap / taller_h < 0.15:
                    groups[ga] = groups[ga] + groups[gb]
                    del groups[gb]
                    changed = True
                    break
            if changed:
                break
        group_ids = list(groups.keys())

    if len(groups) < 2:
        return None  # un solo grupo: no había nada que separar

    # Pasada de seguridad: un símbolo de un solo carácter con partes
    # sueltas que NO siguen el patrón punto-sobre-palo de la pasada 2 (el
    # "%" — dos círculos en diagonal + una raya, ninguno dentro del otro)
    # se ve geométricamente igual a un trazo compuesto real... salvo que
    # las letras de una palabra SIEMPRE comparten renglón (alto overlap en
    # Y entre vecinas, aunque difieran en alto por ascendentes/x-height —
    # una "e" minúscula igual queda casi toda dentro del rango vertical de
    # la "D" de al lado) mientras que las partes sueltas de un símbolo así
    # normalmente quedan en alturas distintas sin compartir renglón. Si
    # algún par de grupos consecutivos (de izquierda a derecha) no
    # comparte renglón, mejor no separar nada — una sola pieza gigante sin
    # separar en el peor caso no calibra bien, pero varias piezas FALSAS
    # arruinan el conteo/calibración de todo el grupo.
    final_boxes = sorted((group_bbox(idxs) for idxs in groups.values()), key=lambda b: (b[0] + b[2]) / 2)
    for (x0a, y0a, x1a, y1a), (x0b, y0b, x1b, y1b) in zip(final_boxes, final_boxes[1:]):
        y_overlap = min(y1a, y1b) - max(y0a, y0b)
        shorter_h = min(y1a - y0a, y1b - y0b)
        if shorter_h <= 0 or y_overlap / shorter_h < 0.4:
            return None

    return list(groups.values())


def split_compound_drawing_into_letters(d: dict) -> List["fitz.Rect"]:
    """Algunos .cdr exportan un set de letras como UN SOLO trazo compuesto
    (las letras "soldadas"/combinadas en un solo objeto, en vez de
    convertidas a curvas como objetos separados) — un único `d` de
    page.get_drawings() que en realidad contiene varias letras. El resto
    del pipeline (en app.py: _extract_pieces_geometric, _gather_letter_rects)
    asume que cada `d` es una sola letra, así que con un archivo así no
    encuentra nada y cae al último recurso (conteo por contorno de imagen,
    mucho menos preciso). Ver _group_compound_subpolys.

    Devuelve una lista de rects, uno por letra — [] si `d` ya es una sola
    letra (nada que separar), para que quien llama siga usando d["rect"]
    como venía."""
    polys = _flatten_drawing_items(d["items"])
    groups = _group_compound_subpolys(polys)
    if groups is None:
        return []

    def bbox(idxs):
        xs = [p[0] for i in idxs for p in polys[i]]
        ys = [p[1] for i in idxs for p in polys[i]]
        return fitz.Rect(min(xs), min(ys), max(xs), max(ys))

    return [bbox(idxs) for idxs in groups]


def _split_compound_drawing_polys(
    d: dict,
) -> List[Tuple["fitz.Rect", List[List[Tuple[float, float]]]]]:
    """Como split_compound_drawing_into_letters, pero devuelve además los
    polígonos de cada letra (no solo su rect) — los necesita
    extract_group_polygons_mm para sacar el contorno real de nesting/LED
    de un archivo con las letras soldadas en un solo trazo compuesto, en
    vez de re-emparejar por bounding box exacto contra page.get_drawings()
    (que para este caso no tiene un `d` por letra, solo el compuesto
    entero — ver esa función)."""
    polys = _flatten_drawing_items(d["items"])
    groups = _group_compound_subpolys(polys)
    if groups is None:
        return []

    result = []
    for idxs in groups:
        xs = [p[0] for i in idxs for p in polys[i]]
        ys = [p[1] for i in idxs for p in polys[i]]
        rect = fitz.Rect(min(xs), min(ys), max(xs), max(ys))
        result.append((rect, [polys[i] for i in idxs]))
    return result


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
    import image_analysis as _ia  # import perezoso: evita import circular con nesting.py al cargar

    pieces_from_text = _ia._extract_pieces_from_pdf(cdr_pdf_path)
    by_page = [(p["page"], p["letter_groups"]) for p in pieces_from_text]
    page_no, group = _match_group_for_nesting(by_page, letter_heights)

    if group is None:
        geo_groups = _ia._extract_pieces_geometric(cdr_pdf_path)
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
        drawings = dedupe_drawings(page.get_drawings())
        result: List[List[List[Tuple[float, float]]]] = []
        for tx0, ty0, tx1, ty1 in target_rects:
            match = None
            for d in drawings:
                r = d["rect"]
                if (abs(r.x0 - tx0) < MATCH_TOL_PT and abs(r.y0 - ty0) < MATCH_TOL_PT and
                        abs(r.x1 - tx1) < MATCH_TOL_PT and abs(r.y1 - ty1) < MATCH_TOL_PT):
                    match = d
                    break

            polys_pt: List[List[Tuple[float, float]]] = []
            if match is not None:
                # El hueco de una letra (el aro de una "D", los dos lazos de
                # un ampersand, etc.) no siempre viene como subtrazo del
                # MISMO drawing que el contorno exterior — a veces
                # LibreOffice/PyMuPDF lo exporta como un drawing APARTE, con
                # su propio bounding box (más chico, encajado dentro del de
                # la letra). Si solo se toma el drawing que calza exacto con
                # el rect de la letra, esos huecos se pierden por completo
                # (la letra sale sólida). Se agrega aquí cualquier otro
                # drawing cuyo rect quede enteramente DENTRO del de esta
                # letra — es del hueco de esta letra, no de la vecina,
                # porque las letras nunca se traslapan entre sí.
                items_all = list(match["items"])
                for d in drawings:
                    if d is match:
                        continue
                    r = d["rect"]
                    if (r.x0 >= tx0 - MATCH_TOL_PT and r.x1 <= tx1 + MATCH_TOL_PT and
                            r.y0 >= ty0 - MATCH_TOL_PT and r.y1 <= ty1 + MATCH_TOL_PT):
                        items_all.extend(d["items"])
                polys_pt = _flatten_drawing_items(items_all)
            else:
                # No hay NINGÚN drawing cuyo rect calce exacto con esta
                # letra — pasa cuando el archivo trae varias letras
                # SOLDADAS en un solo trazo compuesto (ver
                # split_compound_drawing_into_letters, usado en app.py para
                # detectar estas letras en primer lugar): acá no se busca
                # por rect exacto contra page.get_drawings() porque ESE
                # rect no existe; se vuelve a separar cada compuesto en
                # letras y se usa la que calce.
                for d in drawings:
                    if d["rect"].x0 > tx1 or d["rect"].x1 < tx0:
                        continue  # ni se solapa en X: no puede contener esta letra
                    for sub_rect, sub_polys in _split_compound_drawing_polys(d):
                        if (abs(sub_rect.x0 - tx0) < MATCH_TOL_PT and abs(sub_rect.y0 - ty0) < MATCH_TOL_PT and
                                abs(sub_rect.x1 - tx1) < MATCH_TOL_PT and abs(sub_rect.y1 - ty1) < MATCH_TOL_PT):
                            polys_pt = sub_polys
                            break
                    if polys_pt:
                        break

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


def _prune_skeleton_spurs(
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]], dist: np.ndarray, max_passes: int = 6,
) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Poda ramas-hoja cortas (espolones) del árbol de esqueleto: Zhang-Suen
    sobre un borde con el "escalonado" normal de rasterizar una curva a
    veces dispara una micro-rama de unos pocos píxeles en una unión, en
    vez de un mini-ciclo (eso ya lo corrige el MST, ver
    _skeleton_mst_adjacency) — no es una ramificación real de la letra (el
    brazo de una "f", el ápice de una "M"), es ruido de borde. Sin podarla,
    cada una de esas puntas falsas además se extiende de más por
    _piece_skeleton_mm (ver ahí) creyendo que es una punta física real,
    así que el error se duplica.

    Se poda cualquier rama-hoja MÁS CORTA que `width_px` — el ancho local
    del trazo justo en su punto de unión (vía distance transform): ESE
    disparo normal de la distancia en una unión/esquina (varios trazos
    confluyendo ahí) es justo lo que hace falta acá, porque un espolón de
    ruido nace exactamente en esas uniones anchas — un umbral más chico
    (el ancho promedio de todo el trazo) deja pasar sin podar espolones de
    borde bastante más largos que el ancho típico del trazo, con tal de
    que sean más cortos que esa unión ancha en particular. Una
    ramificación real (el brazo de una "f") sigue siendo más larga que el
    ancho de la unión que la sostiene; una de ruido no. Varias pasadas
    porque podar una rama puede dejar al descubierto otra más corta (una
    unión de 3+ espolones)."""
    adjacency = {k: list(v) for k, v in adjacency.items()}
    for _ in range(max_passes):
        deg = {n: len(nbs) for n, nbs in adjacency.items()}
        leaves = [n for n, d in deg.items() if d == 1]
        removed_any = False
        for leaf in leaves:
            if deg.get(leaf, 0) != 1:
                continue  # ya se podó en esta misma pasada
            path = [leaf]
            prev, cur = None, leaf
            length = 0.0
            while True:
                nbs = [n for n in adjacency.get(cur, []) if n != prev]
                if not nbs:
                    break
                nxt = nbs[0]
                length += math.hypot(cur[0] - nxt[0], cur[1] - nxt[1])
                path.append(nxt)
                prev, cur = cur, nxt
                if deg.get(cur, 0) != 2:
                    break
            if deg.get(cur, 0) < 3:
                continue  # terminó en otra hoja (componente aislado chico): no podar
            width_px = float(dist[cur[1] + 1, cur[0] + 1]) * 2.0
            if length < width_px:
                for a, b in zip(path, path[1:]):
                    adjacency[a] = [n for n in adjacency[a] if n != b]
                    adjacency[b] = [n for n in adjacency[b] if n != a]
                removed_any = True
        if not removed_any:
            break
    return adjacency


def _piece_skeleton_mm(
    polys: List[List[Tuple[float, float]]], depths: List[int], res_mm: float = LED_WIDTH_RES_MM,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], float, List[Dict[str, Any]], float]:
    """Eje medio (centerline) de una pieza, igual que la foto de referencia
    del cliente: UNA línea por el centro de cada trazo, que se ramifica
    donde la letra se ramifica de verdad (el ápice de una "M", la cruz de
    una "f") — no una vuelta pegada al borde (eso se ve como dos líneas
    paralelas en un trazo angosto, que es justo la confusión que se quiere
    evitar). Devuelve (segmentos en mm, longitud total en mm, lista de
    "brazos" — cada tramo entre dos ramificaciones, o punta a punta si no
    hay ninguna, con su propio largo, y el ANCHO típico del trazo en mm):
    una ramificación real (el ápice de una "M", la cruz de una "f") es una
    separación física entre tiras de LED, así que el sobrante de un paso
    no se puede repartir de un brazo a otro — cada uno redondea su propio
    módulo por separado (ver _piece_led_stats).

    El ancho se mide SOBRE EL ESQUELETO (mediana de 2x la distancia al
    borde en cada píxel del eje medio ya podado), no promediando sobre
    TODO el área sólida como antes: para un trazo de ancho uniforme W, la
    distancia al borde más cercano crece en rampa desde 0 en cada orilla
    hasta W/2 al centro — el promedio de esa rampa sobre toda el área es
    W/4, así que "promedio × 2" da W/2, la MITAD del ancho real (confirmado
    con una barra recta sintética de 220mm: el promedio sobre toda el área
    daba ~110mm). En el centro del esqueleto, en cambio, la distancia YA
    ES el máximo local (W/2 exacto), sin ese sesgo — crítico para la
    decisión de trazo grueso (≥20cm, ver LED_THICK_STROKE_MM): con el
    sesgo viejo, una letra necesitaba ~40cm de ancho REAL para medir 20cm
    y disparar el modo doble."""
    mask = _rasterize_piece_solid(polys, depths, res_mm)
    if mask is None or not mask.any():
        return [], 0.0, [], 0.0

    all_xs = [p[0] for poly in polys for p in poly]
    all_ys = [p[1] for poly in polys for p in poly]
    min_x, min_y = min(all_xs), min(all_ys)

    def to_mm(p: Tuple[float, float]) -> Tuple[float, float]:
        return (min_x + p[0] * res_mm, min_y + p[1] * res_mm)

    # Islas físicas (regiones sólidas DESCONECTADAS entre sí dentro de la
    # misma pieza: el puntito de una "í"/tilde, un adorno chico suelto) —
    # se calculan sobre el raster SÓLIDO, no sobre el esqueleto, porque un
    # puntito de pocos milímetros a esta resolución a veces adelgaza a un
    # solo píxel sin vecinos (ni una arista): eso lo deja totalmente fuera
    # de `arms` más abajo (ni un solo "brazo" lo representa), así que sin
    # este respaldo esa isla entera desaparece del conteo de módulos —
    # confirmado real con un puntito sintético de 6mm aislado. Cada isla
    # con solidez propia es un pedazo de acrílico/canal separado de los
    # demás y necesita su propio módulo mínimo (ver el piso por isla en
    # _piece_led_stats), así que se guarda su centroide para poder
    # agregarle un brazo de respaldo si el esqueleto no la cubre.
    num_islands, island_labels = cv2.connectedComponents(mask, connectivity=8)
    island_centroid_px: Dict[int, Tuple[float, float]] = {}
    if num_islands > 1:
        ys_m, xs_m = np.where(mask > 0)
        sums: Dict[int, List[float]] = {}
        for x, y in zip(xs_m.tolist(), ys_m.tolist()):
            lbl = int(island_labels[y, x])
            s = sums.setdefault(lbl, [0.0, 0.0, 0.0])
            s[0] += x
            s[1] += y
            s[2] += 1.0
        for lbl, (sx, sy, n) in sums.items():
            island_centroid_px[lbl] = (sx / n, sy / n)

    skel = _zhang_suen_thin(mask)
    ys, xs = np.where(skel > 0)
    pts = set(zip(xs.tolist(), ys.tolist()))

    segments_mm: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    total_mm = 0.0
    arms: List[Dict[str, Any]] = []
    width_mm = 0.0
    covered_islands: set = set()

    if len(pts) >= 2:
        raw_adjacency = {
            (x, y): [
                (x + dx, y + dy)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pts
            ]
            for x, y in pts
        }
        adjacency = _skeleton_mst_adjacency(raw_adjacency)
        padded_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
        adjacency = _prune_skeleton_spurs(adjacency, dist)

        skel_nodes = [n for n, nbs in adjacency.items() if nbs]
        if skel_nodes:
            skel_dist_vals = np.array([dist[n[1] + 1, n[0] + 1] for n in skel_nodes])
            width_mm = float(np.median(skel_dist_vals)) * 2.0 * res_mm

        for comp_segs in _skeleton_dfs_segments(adjacency):
            for (x1, y1), (x2, y2) in comp_segs:
                p1 = (min_x + x1 * res_mm, min_y + y1 * res_mm)
                p2 = (min_x + x2 * res_mm, min_y + y2 * res_mm)
                segments_mm.append((p1, p2))
                total_mm += math.hypot(x2 - x1, y2 - y1) * res_mm

        # El eje medio de un trazo con punta RECTA (el palo de una "l"/"i", el
        # remate de una "M") matemáticamente se queda corto por ~medio ancho
        # del trazo en esa punta (es la propiedad normal de un esqueleto: cerca
        # de una esquina recta, el punto más cercano del borde deja de ser el
        # filo que se venía siguiendo, así que la cresta de distancia máxima se
        # corta ahí) — en una pieza corta y angosta (una "l") eso puede ser
        # 15-20% de su largo real, suficiente para perder un módulo entero.
        # Se extiende cada punta VERDADERA (grado 1, no ramificación, y ya sin
        # los espolones falsos que podó _prune_skeleton_spurs arriba — sin esa
        # poda, cada espolón de ruido también se extendía de más acá, creyendo
        # que era una punta física real) hasta el borde real de la pieza, por
        # su ancho local ahí en la dirección de su último tramo — salvo el
        # corte artificial de un hueco circular (el aro de una "e"/"o", cortado
        # en un punto arbitrario por _skeleton_mst_adjacency): esas dos puntas
        # eran vecinas ANTES del MST, así que no son el final físico de un
        # trazo.
        degree = {n: len(nbs) for n, nbs in adjacency.items()}
        endpoints = {n for n, d in degree.items() if d == 1}
        extension_px: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for ep in endpoints:
            if any(raw_nb in endpoints for raw_nb in raw_adjacency.get(ep, [])):
                continue  # corte artificial de un anillo, no una punta real
            nb = adjacency[ep][0]
            dx, dy = ep[0] - nb[0], ep[1] - nb[1]
            norm = math.hypot(dx, dy)
            if norm == 0:
                continue
            half_w_px = float(dist[ep[1] + 1, ep[0] + 1])
            if half_w_px <= 0:
                continue
            ex, ey = ep[0] + dx / norm * half_w_px, ep[1] + dy / norm * half_w_px
            extension_px[ep] = (ex, ey)
            p1, p2 = to_mm(ep), to_mm((ex, ey))
            segments_mm.append((p1, p2))
            total_mm += half_w_px * res_mm

        # Brazos: un tramo entre dos nodos "estructurales" (grado != 2 — una
        # punta o una ramificación real), caminando por los nodos de grado 2
        # de en medio. Cada rama de una "Y" real (el ápice de una "M", la cruz
        # de una "f") o cada "corte" de un aro (el hueco de una "e") sale como
        # su propio brazo — así cada uno redondea su propio módulo por
        # separado en vez de repartir el sobrante de un paso entre brazos (ver
        # _piece_led_stats). Cada brazo se etiqueta con su isla física (ver
        # island_centroid_px arriba) para que el piso de "al menos 1 módulo"
        # en _piece_led_stats se aplique por isla, no por pieza completa.
        visited_edges: set = set()
        structural = [n for n, d in degree.items() if d != 2]
        for n in structural:
            for nb in adjacency[n]:
                key = (n, nb) if n <= nb else (nb, n)
                if key in visited_edges:
                    continue
                visited_edges.add(key)
                path_px = [n, nb]
                prev, cur = n, nb
                while degree.get(cur, 0) == 2:
                    nbs2 = [x for x in adjacency[cur] if x != prev]
                    if not nbs2:
                        break
                    nxt = nbs2[0]
                    k2 = (cur, nxt) if cur <= nxt else (nxt, cur)
                    visited_edges.add(k2)
                    path_px.append(nxt)
                    prev, cur = cur, nxt

                pts_mm = [to_mm(p) for p in path_px]
                if path_px[0] in extension_px:
                    pts_mm.insert(0, to_mm(extension_px[path_px[0]]))
                if path_px[-1] in extension_px:
                    pts_mm.append(to_mm(extension_px[path_px[-1]]))

                arm_segs = list(zip(pts_mm, pts_mm[1:]))
                arm_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in arm_segs)
                if arm_len <= 0:
                    continue
                island = int(island_labels[n[1], n[0]])
                covered_islands.add(island)
                arms.append({"segments_mm": arm_segs, "length_mm": arm_len, "island": island})

    # Islas físicas sin NINGÚN brazo que las represente (ver
    # island_centroid_px arriba: un puntito tan chico que el esqueleto lo
    # adelgazó a un solo píxel suelto sin vecinos, o la pieza entera no
    # tuvo ni 2 píxeles de esqueleto) — se agrega un brazo de largo casi
    # cero en su centroide, solo para que el piso por isla en
    # _piece_led_stats tenga algo a qué agregarle el módulo, y para que el
    # diagrama tenga un punto dónde dibujarlo.
    for island, (cx, cy) in island_centroid_px.items():
        if island in covered_islands:
            continue
        p1 = to_mm((cx, cy))
        p2 = (p1[0] + 0.01, p1[1])
        arms.append({"segments_mm": [(p1, p2)], "length_mm": 0.01, "island": island})

    return segments_mm, total_mm, arms, width_mm


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
