"""
Nesting real de letras de acrílico (u otro material) sobre una hoja de
corte: acomoda el contorno real de cada pieza (no su caja) en una grilla
de ocupación, probando rotaciones, con un algoritmo de relleno
esquina-a-esquina.

Antes vivía junto con la extracción de geometría y el dibujo del plano en
un solo nesting.py; se separó en geometry.py / led_layout.py /
sheet_packing.py / nesting_pdf.py (ver auditoría de reestructuración) —
nesting.py quedó como fachada de compatibilidad que re-exporta todo esto.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from geometry import extract_group_polygons_mm

SHEET_W_MM_DEFAULT = 1800.0  # 180 cm — hoja de acrílico
SHEET_H_MM_DEFAULT = 2400.0  # 240 cm
GAP_MM_DEFAULT = 6.0
GRID_RES_MM = 3.0            # resolución de la grilla de ocupación (más fino = más lento y más ajustado)

# Tamaño de hoja según el material — cada material se corta de su propia
# lámina, así que cada uno arma su(s) hoja(s) por separado (no se pueden
# mezclar letras de materiales distintos en un mismo plano de corte).
# Se busca por coincidencia de texto (ver material_for_subprocess) usando
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


# ══ NESTING (relleno esquina-a-esquina sobre grilla de ocupación) ═════════════

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
