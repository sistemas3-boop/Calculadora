"""
Análisis de archivos de diseño (.cdr, PDF, imagen JPG/PNG): medidas,
cantidad de piezas/letras, y los PNG de preview/anotado que se muestran
en el formulario.

Se separó de app.py (ver auditoría de reestructuración). No depende de
Flask/DB — recibe rutas de archivo, devuelve diccionarios con los
resultados (y escribe los PNG de preview/anotado en el mismo folder que
el archivo analizado, igual que antes).
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from werkzeug.utils import secure_filename

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

import nesting


# LibreOffice (con el filtro libcdr) es lo que convierte .cdr -> PDF en modo
# headless dentro del contenedor Docker. No hay binding Python confiable
# para leer .cdr directamente, y automatizar CorelDRAW por COM no es viable
# porque el servidor corre en Linux/Docker sin CorelDRAW instalado.
def _find_libreoffice_bin() -> Optional[str]:
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    # El instalador de Windows no agrega soffice.exe al PATH; usado para
    # pruebas en local fuera de Docker.
    for candidate in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


LIBREOFFICE_BIN = _find_libreoffice_bin()
LIBREOFFICE_AVAILABLE = bool(LIBREOFFICE_BIN)


# Tesseract OCR lee las cotas (p.ej. "3877", "950") cuando vienen dibujadas
# como píxeles dentro de una imagen plana (JPG/PNG, o un PDF escaneado sin
# datos vectoriales) — no hay texto real "Medidas: ..." que leer en ese
# caso, a diferencia de un .cdr convertido a PDF (ver analyze_image_file).
def _find_tesseract_bin() -> Optional[str]:
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


TESSERACT_BIN = _find_tesseract_bin() if PYTESSERACT_AVAILABLE else None
TESSERACT_AVAILABLE = bool(TESSERACT_BIN)
if TESSERACT_AVAILABLE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_BIN


# ══ CDR (CORELDRAW) FILE ANALYSIS ═════════════════════════════════════════════

def _count_pieces_contours(img) -> Tuple[int, List]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    total_area = h_img * w_img
    edge_margin = 0.015
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    best_count = 0
    best_boxes: List = []

    def _filter(thresh_img):
        n, _, stats, _ = cv2.connectedComponentsWithStats(thresh_img, connectivity=8)
        valid = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if not (total_area * 0.001 <= area <= total_area * 0.60):
                continue
            if (x < w_img * edge_margin or y < h_img * edge_margin or
                    x + cw > w_img * (1 - edge_margin) or
                    y + ch > h_img * (1 - edge_margin)):
                continue
            aspect = cw / max(ch, 1)
            if not (0.08 <= aspect <= 12.0):
                continue
            valid.append((int(x), int(y), int(cw), int(ch)))
        return valid

    for blur_ksize in [(3, 3), (5, 5)]:
        blurred = cv2.GaussianBlur(enhanced, blur_ksize, 0)
        _, t = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        v = _filter(t)
        if len(v) > best_count:
            best_count, best_boxes = len(v), v
        _, t = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        v = _filter(t)
        if len(v) > best_count:
            best_count, best_boxes = len(v), v
        t = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 4)
        v = _filter(t)
        if len(v) > best_count:
            best_count, best_boxes = len(v), v
        t = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        v = _filter(t)
        if len(v) > best_count:
            best_count, best_boxes = len(v), v

    return best_count, best_boxes


PT_TO_CM = 2.54 / 72.0

# Muchas hojas .cdr de producción traen, junto al dibujo a escala, un bloque
# de texto "Concepto: ... / Medidas: A x B mm / Materiales: ... / Cantidad:
# N pieza(s) a M cara(s)" (una fila por pieza). Cuando existe, es mucho más
# confiable leer ese texto real del PDF que tratar de adivinar el contorno
# del dibujo por imagen. El orden de los dos números en "Medidas:" no es
# fijo (a veces es Alto x Ancho, a veces Ancho x Alto), así que se cruzan
# contra las cotas sueltas dibujadas sobre la pieza: la cota escrita en
# horizontal es el ancho, la escrita en vertical (rotada, y por lo tanto
# partida en varias líneas de texto) es el alto.
# La unidad no siempre es "A x B mm" (una sola unidad al final): algunas
# hojas la escriben en cm y repiten la unidad en cada número por separado
# ("A cm X B cm") — la unidad de cada número es opcional y, si falta, se
# toma la del segundo número (ver _parse_meas_match). Piezas grandes
# (fachadas, letreros de varios metros) a veces usan "m" en vez de cm/mm
# — "mm"/"cm" van ANTES que "m" sueltas en la alternancia para que no
# matcheen solo la "m" de "mm"/"cm" y dejen colgando una letra.
_UNIT_RE = r"(mm|cm|m)(?![a-zA-Z])"
_MEAS_RE = re.compile(rf"(\d[\d.,]*)\s*(?:{_UNIT_RE})?\s*[x×]\s*(\d[\d.,]*)\s*{_UNIT_RE}", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(concepto|medidas|materiales|cantidad)\s*:?\s*$", re.IGNORECASE)
_QTY_RE = re.compile(r"(\d+)\s*pieza", re.IGNORECASE)
_CARAS_RE = re.compile(r"(\d+)\s*cara", re.IGNORECASE)
# Cota suelta dibujada junto a una pieza (sin el renglón "Medidas:" al
# lado): no siempre es un entero en mm ("1234") — también se ve como
# decimal en cm ("2.78cm"), a veces partido en varias líneas por el
# conversor de PDF ("2.7\n8 \ncm" para "2.78 cm") que ya se reconstruye
# antes de este match quitando espacios/saltos de línea.
_LOOSE_DIM_RE = re.compile(rf"^(\d+(?:[.,]\d+)?)(?:{_UNIT_RE})?$", re.IGNORECASE)


def _parse_num(s: str) -> float:
    return float(s.replace(",", "."))


def _unit_to_mm_factor(unit: str) -> float:
    unit = unit.lower()
    if unit == "cm":
        return 10.0
    if unit == "m":
        return 1000.0
    return 1.0  # mm, o sin unidad (se asume ya en mm)


def _parse_loose_dimension_mm(text: str) -> Optional[float]:
    """Cota suelta → mm, o None si el texto no es una cota reconocible (p.ej.
    es parte de un ícono o una etiqueta). Sin unidad se asume que YA viene
    en mm (el formato más visto: un entero dibujado junto al trazo)."""
    compact = re.sub(r"\s", "", text)
    m = _LOOSE_DIM_RE.match(compact)
    if not m:
        return None
    mm = _parse_num(m.group(1)) * _unit_to_mm_factor(m.group(2) or "")
    # mismo tope que antes (originalmente "máximo 5 dígitos" en mm): evita
    # leer dos cotas vecinas pegadas en un solo bloque de texto sin
    # separador (p.ej. "56184311") como un valor absurdo de ~100 km.
    if not (0 < mm <= 99999):
        return None
    return mm


def _parse_meas_match(m: "re.Match") -> Tuple[float, float]:
    """(n1_mm, n2_mm) de un match de _MEAS_RE, convirtiendo a mm según la
    unidad de cada número — si el primero no trae unidad propia (formato
    "A x B mm"), hereda la del segundo."""
    unit2 = m.group(4) or ""
    unit1 = m.group(2) or unit2
    return (_parse_num(m.group(1)) * _unit_to_mm_factor(unit1),
            _parse_num(m.group(3)) * _unit_to_mm_factor(unit2))


def _cluster_rects_by_y_gap(rects: List["fitz.Rect"], gap_factor: float = 3.0) -> List[List["fitz.Rect"]]:
    """Agrupa trazos por cercanía vertical (línea base): corta cada vez que
    hay un salto mayor a `gap_factor` veces el alto promedio entre un trazo
    y el siguiente. No descarta nada — cada quien llama decide qué hacer
    con los grupos chicos (ruido) o los agrupa como diseños separados."""
    if not rects:
        return []
    avg_h = sum(r.height for r in rects) / len(rects)
    by_y = sorted(rects, key=lambda r: (r.y0 + r.y1) / 2)
    clusters = [[by_y[0]]]
    for r in by_y[1:]:
        prev_c = (clusters[-1][-1].y0 + clusters[-1][-1].y1) / 2
        cur_c = (r.y0 + r.y1) / 2
        if abs(cur_c - prev_c) > gap_factor * avg_h:
            clusters.append([r])
        else:
            clusters[-1].append(r)
    return clusters


def _drop_isolated_rects(rects: List["fitz.Rect"], gap_factor: float = 3.0) -> List["fitz.Rect"]:
    """Descarta trazos cuya posición vertical está muy lejos de la masa
    principal de trazos de la franja (p.ej. un ícono de la cabecera de la
    hoja que cayó dentro del mismo rango de alto por casualidad).

    Esto corre ANTES de calcular la escala del dibujo (mm por punto): si un
    elemento ajeno queda incluido ahí, ensancha el "ancho dibujado" y
    encoge la escala de TODO el archivo, no solo la de ese elemento — así
    se detectó (una R que debía medir 106.5 cm salía en 100.1 cm porque el
    ancho de referencia incluía un ícono fuera del diseño).
    """
    if len(rects) < 3:
        return rects
    clusters = _cluster_rects_by_y_gap(rects, gap_factor)
    kept = [r for c in clusters if len(c) >= 2 for r in c]
    return kept or rects


def _gather_letter_rects(page, y_lo: float, y_hi: float) -> List["fitz.Rect"]:
    """Trazos rellenos significativos (letras, no blancos ni ruido diminuto)
    dentro de una franja vertical de la página."""
    rects = []
    for d in nesting.dedupe_drawings(page.get_drawings()):
        col, r = d.get("fill"), d["rect"]
        if col is None or min(col) > 0.95:
            continue
        if not (y_lo <= (r.y0 + r.y1) / 2 < y_hi):
            continue
        # Normalmente un trazo = una letra, pero algunos .cdr exportan
        # varias letras SOLDADAS en un solo trazo compuesto (p.ej. un
        # título completo combinado en un objeto) — sin esto, ese trazo
        # cuenta como UNA sola "letra" gigante en vez de varias.
        split = nesting.split_compound_drawing_into_letters(d)
        if split:
            rects.extend(r2 for r2 in split if y_lo <= (r2.y0 + r2.y1) / 2 < y_hi)
        else:
            rects.append(r)
    if len(rects) < 2:
        return []
    max_area = max(r.width * r.height for r in rects)
    rects = [r for r in rects if r.width * r.height >= 0.05 * max_area]
    return _drop_isolated_rects(rects)


def _cluster_boxes_by_height(
    boxes: List[Tuple[float, float, float, float]], mm_per_unit: float,
    gap_ratio: float = 1.4, single_box_ok: bool = False,
    drop_singleton_clusters: bool = False, dedupe_by_y_proximity: bool = False,
) -> List[Dict[str, Any]]:
    """Agrupa cajas (x0, y0, x1, y1) por su ALTO real (no por color) usando
    una escala uniforme ya conocida (`mm_per_unit`) — núcleo compartido
    entre _cluster_rects_by_height (trazos vectoriales de un PDF, en
    puntos del dibujo) y _cluster_boxes_by_height_px (cajas de una imagen
    plana, en píxeles, ver _split_content_and_scale): mismo algoritmo
    sobre dos representaciones de caja distintas.

    Las hojas técnicas dibujan la pieza a escala reducida (p.ej. "Escala
    1:40"), así que el alto de cada caja en su unidad original no es su
    alto real — hay que convertirlo con la escala de ESE dibujo. Un mismo
    concepto puede traer dos tamaños de letra (p.ej. un título grande y un
    subtítulo chico, del mismo color) — separarlas por alto en vez de por
    color detecta ese caso: se ordenan las cajas por alto y se corta cada
    vez que hay un salto de tamaño mayor a `gap_ratio` entre una y la
    siguiente.

    Banderas (cada llamador conserva su propio comportamiento de antes,
    ver los wrappers abajo):
    - `single_box_ok`: con una sola caja, arma un grupo de 1 en vez de
      devolver vacío (el análisis de imagen sí quiere ese caso; el de PDF
      vectorial no, ahí una sola caja suele ser ruido).
    - `drop_singleton_clusters`: descarta un cluster de tamaño 1 (un solo
      trazo de ESE alto, probable ruido) — solo para trazos vectoriales,
      más propensos a ruido que las cajas ya filtradas de una imagen.
    - `dedupe_by_y_proximity`: dentro de un cluster de 3+ cajas, si un
      elemento ajeno (p.ej. un ícono de la cabecera de la hoja) cae en el
      mismo rango de alto por casualidad pero queda lejos verticalmente,
      se separa y solo se conserva el subgrupo más grande.
    """
    if not boxes:
        return []
    if single_box_ok and len(boxes) == 1:
        x0, y0, x1, y1 = boxes[0]
        w_cm, h_cm = round((x1 - x0) * mm_per_unit / 10, 1), round((y1 - y0) * mm_per_unit / 10, 1)
        return [{
            "letters": 1, "height_cm": h_cm, "width_cm": w_cm,
            "rects": [[x0, y0, x1, y1]],
            "pieces": [{"width_cm": w_cm, "height_cm": h_cm}],
        }]
    if len(boxes) < 2:
        return []

    boxes = sorted(boxes, key=lambda b: b[3] - b[1])
    clusters = [[boxes[0]]]
    for b in boxes[1:]:
        prev_h = clusters[-1][-1][3] - clusters[-1][-1][1]
        h = b[3] - b[1]
        if h / max(prev_h, 1e-9) > gap_ratio:
            clusters.append([b])
        else:
            clusters[-1].append(b)

    groups = []
    for cluster in clusters:
        if dedupe_by_y_proximity and len(cluster) >= 3:
            avg_h_all = sum(b[3] - b[1] for b in cluster) / len(cluster)
            by_y = sorted(cluster, key=lambda b: (b[1] + b[3]) / 2)
            sub = [[by_y[0]]]
            for b in by_y[1:]:
                prev_c = (sub[-1][-1][1] + sub[-1][-1][3]) / 2
                cur_c = (b[1] + b[3]) / 2
                if abs(cur_c - prev_c) > 3 * avg_h_all:
                    sub.append([b])
                else:
                    sub[-1].append(b)
            cluster = max(sub, key=len)

        if drop_singleton_clusters and len(cluster) < 2:
            continue  # un solo trazo de ese tamaño: probablemente ruido
        avg_h = sum(b[3] - b[1] for b in cluster) / len(cluster)
        avg_w = sum(b[2] - b[0] for b in cluster) / len(cluster)
        cluster_by_x = sorted(cluster, key=lambda b: b[0])
        groups.append({
            "letters": len(cluster),
            "height_cm": round(avg_h * mm_per_unit / 10, 1),
            # Ancho promedio de ESTE grupo (a partir de sus propias cajas),
            # no el ancho total del letrero entre la cantidad de letras: eso
            # no corresponde al ancho real de ninguna letra.
            "width_cm": round(avg_w * mm_per_unit / 10, 1),
            "rects": [[b[0], b[1], b[2], b[3]] for b in cluster],
            # Medida real de cada caja individual (izquierda a derecha), por
            # si el promedio del grupo mezcla piezas de tamaño distinto
            # (p.ej. el logo y las letras en Mifel) — solo para mostrar en
            # el detalle de la interfaz, el cálculo de tiempo sigue usando
            # el alto/ancho del grupo de arriba.
            "pieces": [
                {"width_cm": round((b[2] - b[0]) * mm_per_unit / 10, 1),
                 "height_cm": round((b[3] - b[1]) * mm_per_unit / 10, 1)}
                for b in cluster_by_x
            ],
        })
    groups.sort(key=lambda g: -g["height_cm"])
    return groups


def _cluster_rects_by_height(rects: List["fitz.Rect"], mm_per_pt: float,
                              gap_ratio: float = 1.4) -> List[Dict[str, Any]]:
    """_cluster_boxes_by_height sobre trazos vectoriales de un PDF
    (`fitz.Rect`, en puntos del dibujo) — ver ahí para el algoritmo. En el
    PDF convertido cada letra queda como un trazo cerrado independiente
    aunque se toque con la vecina (cosa que un conteo por imagen fusiona,
    p.ej. la "i" y la "f" en cursiva)."""
    boxes = [(r.x0, r.y0, r.x1, r.y1) for r in rects]
    return _cluster_boxes_by_height(
        boxes, mm_per_pt, gap_ratio,
        drop_singleton_clusters=True, dedupe_by_y_proximity=True,
    )


def _refine_group_heights(groups: List[Dict[str, Any]], loose_digits: List[Dict[str, Any]],
                           exclude_values: set) -> List[Dict[str, Any]]:
    """Si la hoja trae una cota vertical propia para la línea de un grupo
    (p.ej. "286" junto a un subtítulo chico, aparte del "Medidas:" general
    de toda la pieza), se usa ese valor exacto en vez del alto promedio de
    los trazos — el promedio geométrico puede quedar unos milímetros corto
    porque el recuadro de cada letra no siempre llega al tope de la cota
    (antialiasing del renderizado, curvas que no tocan el borde, etc.).
    Se descartan las cotas ya usadas como medida general de la pieza
    (`exclude_values`) y cualquiera cuyo valor esté muy lejos del alto ya
    estimado por grupo, para no enganchar por error la cota de otra línea.
    """
    for g in groups:
        g_ys = [r[1] for r in g["rects"]] + [r[3] for r in g["rects"]]
        g_center = (min(g_ys) + max(g_ys)) / 2
        best, best_dist = None, None
        for d in loose_digits:
            if d["is_wide"]:
                continue
            val = float(d["value"])
            if val in exclude_values or not g["height_cm"]:
                continue
            ratio = (val / 10) / g["height_cm"]
            if not (0.4 <= ratio <= 2.5):
                continue
            dist = abs(d["y_center"] - g_center)
            if best is None or dist < best_dist:
                best, best_dist = d, dist
        if best is not None:
            g["height_cm"] = round(float(best["value"]) / 10, 1)
            g["height_verified"] = True
        else:
            g["height_verified"] = False
    return groups


def _extract_pieces_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Busca bloques 'Medidas: A x B mm' en el PDF y arma una pieza por cada
    uno, con su nombre (Concepto) y cantidad si están cerca. Ver comentario
    de arriba sobre cómo se decide cuál número es alto y cuál es ancho.
    """
    pieces: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            blocks = [b for b in page.get_text("blocks") if b[4].strip()]
            blocks.sort(key=lambda b: (b[1], b[0]))
            meas_blocks = [b for b in blocks if _MEAS_RE.search(b[4])]
            if not meas_blocks:
                continue

            loose_digits = []
            for b in blocks:
                if b in meas_blocks:
                    continue
                bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                mm_val = _parse_loose_dimension_mm(btext)
                if mm_val is not None:
                    loose_digits.append({
                        "value": str(mm_val),
                        "y_center": (by0 + by1) / 2,
                        "is_wide": (bx1 - bx0) > (by1 - by0),
                    })

            # Cada concepto ocupa la franja vertical hasta el punto medio con
            # el concepto vecino: ahí caen las letras de su dibujo.
            anchors = sorted((b[1] + b[3]) / 2 for b in meas_blocks)

            for mb in meas_blocks:
                x0, y0, x1, y1, text = mb[0], mb[1], mb[2], mb[3], mb[4]
                m = _MEAS_RE.search(text)
                n1, n2 = _parse_meas_match(m)
                y_center = (y0 + y1) / 2

                pos = anchors.index(y_center)
                band_lo = (anchors[pos - 1] + y_center) / 2 if pos > 0 else float("-inf")
                band_hi = (y_center + anchors[pos + 1]) / 2 if pos < len(anchors) - 1 else float("inf")
                band_rects = _gather_letter_rects(page, band_lo, band_hi)
                # Solo las cotas sueltas DENTRO de la franja de este concepto:
                # en una hoja con varios conceptos, cada uno trae sus propias
                # cotas y no deben cruzarse (p.ej. la cota de altura de otro
                # concepto en la misma página).
                band_loose_digits = [d for d in loose_digits if band_lo <= d["y_center"] < band_hi]

                name = None
                best_gap = None
                for b in blocks:
                    bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                    if by1 <= y0 and abs(bx0 - x0) < 60 and not _LABEL_RE.match(btext.strip()):
                        gap = y0 - by1
                        if gap < 60 and (best_gap is None or gap < best_gap):
                            best_gap, name = gap, btext.strip()

                quantity, caras = None, None
                for b in blocks:
                    bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                    if by0 >= y1 and by0 - y1 < 150:
                        qm = _QTY_RE.search(btext)
                        if qm:
                            quantity = int(qm.group(1))
                            cm = _CARAS_RE.search(btext)
                            caras = int(cm.group(1)) if cm else None
                            break

                candidates = sorted(
                    (d for d in loose_digits if float(d["value"]) in (n1, n2)),
                    key=lambda d: abs(d["y_center"] - y_center),
                )
                wide = next((d for d in candidates if d["is_wide"]), None)
                tall = next((d for d in candidates if not d["is_wide"]), None)
                orientation_verified = bool(
                    wide and tall and {float(wide["value"]), float(tall["value"])} == {n1, n2}
                )
                if orientation_verified:
                    width_mm, height_mm = float(wide["value"]), float(tall["value"])
                else:
                    # Sin cotas confiables para verificar: se asume que el
                    # texto viene como Alto x Ancho (lo más visto), pero se
                    # marca como no verificado para revisarlo a mano.
                    height_mm, width_mm = n1, n2

                # Escala real del dibujo (mm por punto de PDF), calculada con
                # el ancho ya conocido de este concepto contra lo dibujado —
                # no con el tamaño de la página, que suele estar a escala
                # reducida (ver docstring de _cluster_rects_by_height).
                if band_rects:
                    drawn_w = max(r.x1 for r in band_rects) - min(r.x0 for r in band_rects)
                    mm_per_pt = width_mm / drawn_w if drawn_w > 0 else PT_TO_CM * 10
                else:
                    mm_per_pt = PT_TO_CM * 10
                letter_groups = _cluster_rects_by_height(band_rects, mm_per_pt)
                if len(letter_groups) == 1:
                    # Un solo tamaño de letra en el concepto: el ALTO ya
                    # declarado en "Medidas: A x B mm" (verificado contra las
                    # cotas) es el oficial para esta pieza — es el que se usa
                    # para categorizar el tiempo por altura, aunque el trazo
                    # más alto (p.ej. la M) sea distinto al promedio de todas
                    # las piezas del set (logo + letras cortas y largas
                    # mezcladas). El ANCHO en cambio sí se deja como el
                    # promedio geométrico por letra: ese se usa en fórmulas
                    # por pieza (p.ej. Ensamble sin flange) y ahí el ancho
                    # total de toda la palabra no tiene sentido físico.
                    letter_groups[0]["height_cm"] = round(height_mm / 10, 1)
                    letter_groups[0]["height_verified"] = orientation_verified
                else:
                    letter_groups = _refine_group_heights(letter_groups, band_loose_digits, {n1, n2})
                letters = sum(g["letters"] for g in letter_groups) or None

                pieces.append({
                    "name": name,
                    "width_cm": round(width_mm / 10, 1),
                    "height_cm": round(height_mm / 10, 1),
                    "quantity": quantity or 1,
                    "caras": caras,
                    "letters": letters,
                    "letter_groups": letter_groups,
                    "page": page.number,
                    "orientation_verified": orientation_verified,
                })
    return pieces


def _extract_pieces_geometric(pdf_path: str) -> List[Dict[str, Any]]:
    """Respaldo para cuando la hoja no trae en ningún lado el texto
    "Medidas: A x B mm" (formato libre: solo cotas sueltas dibujadas junto
    a cada diseño, sin ese renglón — visto en la mitad de los archivos
    reales de producción).

    Sin ese texto no hay un "concepto" ya delimitado para anclar cada
    diseño, así que se agrupan las letras de TODA la página por cercanía
    vertical (un salto grande de posición = diseños distintos, p.ej. dos
    variantes del mismo logo en zonas separadas de la hoja) y se calibra
    cada grupo con las cotas sueltas más cercanas: la más ancha como ancho
    total del grupo, la más cercana en vertical como su alto.

    Cada diseño detectado se devuelve como un "letter_group" — se reutiliza
    el mismo mecanismo de la interfaz que ya muestra hasta 2 resultados
    (Grupo 1 / Grupo 2) por archivo, en vez de tratar de adivinar si estos
    diseños son la misma pieza o piezas distintas (aquí no hay texto que lo
    diga, a diferencia de "Concepto:"/"Bandera ATM" en otros formatos).
    """
    all_groups: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            # Respaldo OCR (ver _ocr_scale_for_page) para cuando esta
            # página no tiene NINGÚN texto en absoluto — se calcula una
            # sola vez por página (no por diseño) porque renderizar +
            # correr Tesseract no es gratis, y todos los diseños sin cota
            # de texto en la misma página comparten el mismo resultado.
            ocr_mm_per_pt: Optional[float] = None
            ocr_tried = False
            blocks = [b for b in page.get_text("blocks") if b[4].strip()]
            loose_digits = []
            for b in blocks:
                bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                mm_val = _parse_loose_dimension_mm(btext)
                if mm_val is not None:
                    loose_digits.append({
                        "value": str(mm_val),
                        "y_center": (by0 + by1) / 2,
                        "is_wide": (bx1 - bx0) > (by1 - by0),
                    })

            raw_rects = []
            page_area = page.rect.width * page.rect.height
            for d in nesting.dedupe_drawings(page.get_drawings()):
                col, r = d.get("fill"), d["rect"]
                if col is None:
                    continue
                # Un relleno casi blanco normalmente es el FONDO de la
                # página (un rect que cubre casi toda la hoja), no una letra
                # — pero una letra real tambien puede ser blanca (vinil
                # blanco sobre fondo oscuro), así que solo se descarta el
                # casi-blanco cuando además es grande como el fondo; una
                # letra blanca chica se conserva.
                if min(col) > 0.95 and r.width * r.height >= 0.5 * page_area:
                    continue
                # Normalmente un trazo = una letra, pero algunos .cdr
                # exportan varias letras SOLDADAS en un solo trazo
                # compuesto (p.ej. un título completo combinado en un
                # objeto, en vez de cada letra convertida a curvas por
                # separado) — sin esto, ese trazo cuenta como UNA sola
                # "letra" gigante en vez de varias, y el archivo cae sin
                # necesidad al último recurso (conteo por contorno de
                # imagen, mucho menos preciso).
                split = nesting.split_compound_drawing_into_letters(d)
                raw_rects.extend(split if split else [r])
            if not raw_rects:
                continue
            max_area = max(r.width * r.height for r in raw_rects)
            raw_rects = [r for r in raw_rects if r.width * r.height >= 0.05 * max_area]

            designs = [c for c in _cluster_rects_by_y_gap(raw_rects, 3.0) if len(c) >= 2]
            if not designs:
                continue
            centers = [
                (min(r.y0 for r in c) + max(r.y1 for r in c)) / 2 for c in designs
            ]
            # Cada cota suelta se asigna al diseño más cercano (no a
            # cualquiera dentro de un radio fijo): con dos diseños en la
            # misma hoja, un radio generoso puede alcanzar la cota del
            # diseño vecino (p.ej. el ancho total de un logo pescando la
            # cota de altura de otro logo más abajo en la página).
            for di, cluster in enumerate(designs):
                nearby = [
                    d for d in loose_digits
                    if min(range(len(centers)), key=lambda j: abs(d["y_center"] - centers[j])) == di
                ]
                wide_candidates = [d for d in nearby if d["is_wide"]]
                tall_candidates = [d for d in nearby if not d["is_wide"]]
                if not wide_candidates or not tall_candidates:
                    # Sin ninguna cota de TEXTO cerca para calibrar (visto
                    # real: un .cdr con el texto convertido a curvas no
                    # trae NI el "Medidas: A x B" NI una sola cota suelta
                    # legible — get_text() no encuentra nada en toda la
                    # página). Antes de rendirse al tamaño crudo en puntos
                    # del PDF, se intenta leer la cota por OCR sobre un
                    # render de la página (ver _ocr_scale_for_page): la
                    # cota puede seguir ahí dibujada como curvas, visible
                    # en un render aunque no sea texto real.
                    if not ocr_tried:
                        ocr_tried = True
                        ocr_mm_per_pt = _ocr_scale_for_page(page)
                    if ocr_mm_per_pt is not None:
                        letter_groups = _cluster_rects_by_height(cluster, ocr_mm_per_pt)
                        for g in letter_groups:
                            g["height_verified"] = True
                            g["page"] = page.number
                        all_groups.extend(letter_groups)
                        continue
                    # Tampoco se pudo leer por OCR — antes esto descartaba
                    # el diseño entero, y _analyze_pdf_common terminaba
                    # cayendo al tamaño de PÁGINA por defecto (Carta,
                    # 21.6x27.9cm, sin ninguna relación con el diseño) — un
                    # número con pinta de medida real pero completamente
                    # inventado. Mejor usar el tamaño propio del trazo
                    # vectorial tal cual lo trae el PDF (1pt = 1/72in, el
                    # único supuesto posible sin ninguna cota): no está
                    # garantizado a escala real si Corel se dibujó con otra
                    # unidad, pero sale del DISEÑO mismo, no de un valor de
                    # página arbitrario — se marca sin verificar (ver
                    # "unscaled") para que la interfaz avise al usuario que
                    # confirme las medidas a mano.
                    letter_groups = _cluster_rects_by_height(cluster, PT_TO_CM * 10)
                    for g in letter_groups:
                        g["height_verified"] = False
                        g["unscaled"] = True
                        g["page"] = page.number
                    all_groups.extend(letter_groups)
                    continue

                # la cota de ancho más grande cercana es el ancho total del
                # diseño (puede haber sub-anchos más chicos, p.ej. el ancho
                # de cada palabra por separado, que se ignoran aquí)
                wide = max(wide_candidates, key=lambda d: float(d["value"]))
                tall = min(tall_candidates, key=lambda d: abs(d["y_center"] - centers[di]))
                width_mm, height_mm = float(wide["value"]), float(tall["value"])

                drawn_w = max(r.x1 for r in cluster) - min(r.x0 for r in cluster)
                mm_per_pt = width_mm / drawn_w if drawn_w > 0 else PT_TO_CM * 10
                letter_groups = _cluster_rects_by_height(cluster, mm_per_pt)
                if len(letter_groups) == 1:
                    letter_groups[0]["height_cm"] = round(height_mm / 10, 1)
                    letter_groups[0]["height_verified"] = True
                else:
                    letter_groups = _refine_group_heights(letter_groups, nearby, set())
                for g in letter_groups:
                    g["page"] = page.number
                all_groups.extend(letter_groups)
    return all_groups


def _unique_upload_name(original_filename: str) -> str:
    """Nombre único para guardar un archivo subido. Dos usuarios (o el
    mismo usuario dos veces) subiendo un archivo con el mismo nombre
    original — muy común con nombres genéricos tipo "Sin título-1.cdr" —
    no deben pisarse el archivo guardado ni, más adelante, el PDF
    convertido a partir de él (que se nombra a partir de este mismo
    nombre base, ver _convert_cdr_to_pdf)."""
    safe = secure_filename(original_filename)
    return f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe}"


def _convert_cdr_to_pdf(cdr_path: str, out_dir: str) -> str:
    """Convierte un .cdr a PDF con LibreOffice (filtro libcdr) en modo
    headless. Cada llamada usa su PROPIO perfil de usuario de LibreOffice
    (-env:UserInstallation) en vez del perfil compartido por defecto: dos
    conversiones al mismo tiempo (dos usuarios subiendo un archivo a la
    vez — ya corre en un servidor real, no solo en una máquina de
    desarrollo) chocan por el candado del perfil compartido y una de las
    dos falla o se queda colgada hasta el timeout."""
    profile_dir = tempfile.mkdtemp(prefix="lo_profile_")
    try:
        profile_uri = "file:///" + profile_dir.replace("\\", "/")
        proc = subprocess.run(
            [LIBREOFFICE_BIN, "--headless", "--norestore",
             f"-env:UserInstallation={profile_uri}",
             "--convert-to", "pdf", "--outdir", out_dir, cdr_path],
            capture_output=True, timeout=120,
        )
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
    base = os.path.splitext(os.path.basename(cdr_path))[0]
    pdf_path = os.path.join(out_dir, f"{base}.pdf")
    if not os.path.exists(pdf_path):
        detail = proc.stderr.decode(errors="ignore") or proc.stdout.decode(errors="ignore")
        raise RuntimeError(f"LibreOffice no pudo convertir el .cdr: {detail.strip() or 'sin detalle'}")
    return pdf_path


def _analyze_pdf_common(pdf_path: str, base: str, folder: str) -> Dict[str, Any]:
    """Núcleo de análisis de un PDF que YA tiene los datos vectoriales
    reales (sea de convertir un .cdr con LibreOffice, o de un .pdf subido
    directo que ya trae esos datos) — toma las medidas de la página y
    genera preview + imagen anotada. Separado de analyze_cdr_file para
    poder reusarlo con un PDF subido directo sin pasar por LibreOffice
    (ver analyze_pdf_file)."""
    result: Dict[str, Any] = {
        "width_cm": None, "height_cm": None,
        "pieces": 0, "annotated_filename": None, "preview_filename": None,
        "img_read": False, "detected_pieces": [],
    }
    chosen_groups: List[Dict[str, Any]] = []
    geo_groups: List[Dict[str, Any]] = []
    try:
        detected_pieces = _extract_pieces_from_pdf(pdf_path)
        result["detected_pieces"] = detected_pieces

        # Respaldo: la hoja no trae el texto "Medidas: A x B mm" en ningún
        # lado (formato libre, solo cotas sueltas junto al dibujo — la
        # mitad de los archivos reales son así). Se agrupan las letras de
        # toda la página por cercanía y se calibran con las cotas cercanas.
        geo_groups: List[Dict[str, Any]] = []
        if not detected_pieces:
            geo_groups = _extract_pieces_geometric(pdf_path)

        doc = fitz.open(pdf_path)
        page_index = 0
        if detected_pieces:
            # Renderizar la primera página que sí trae medidas de texto
            # (p.ej. la portada de datos del cliente no las trae).
            for i, p in enumerate(doc):
                if _MEAS_RE.search(p.get_text()):
                    page_index = i
                    break
        elif geo_groups:
            page_index = geo_groups[0]["page"]
        page = doc[page_index]

        if geo_groups:
            geo_groups = geo_groups[:2]  # la interfaz solo muestra hasta 2 grupos
            result["letter_groups"] = geo_groups
            result["letters"] = sum(g["letters"] for g in geo_groups) or None
            result["width_cm"]  = geo_groups[0]["width_cm"]
            result["height_cm"] = geo_groups[0]["height_cm"]
            result["pieces"] = sum(g["letters"] for g in geo_groups) or 1
            chosen_groups = [g for g in geo_groups if g["page"] == page_index]
            if any(g.get("unscaled") for g in geo_groups):
                # Ver "unscaled" en _extract_pieces_geometric: el archivo no
                # trae NINGUNA cota de texto en toda la página (ni "Medidas:
                # A x B" ni una cota suelta), así que esta medida sale
                # directo de los puntos del PDF (1pt = 1/72in) sin ninguna
                # cota real que la confirme.
                result["scale_warning"] = (
                    "El archivo no trae ninguna cota de medida (ni “Medidas: A x B” "
                    "ni una cota suelta) — esta medida sale directo del dibujo vectorial, "
                    "sin verificar contra ninguna cota real. Confírmala antes de aplicar."
                )
        elif detected_pieces:
            # Un .cdr de producción suele traer, además del set de letras,
            # otras piezas independientes (p.ej. "Bandera ATM") — esas se
            # descartan. Pero también puede traer VARIOS conceptos que SÍ
            # son letras por separado (p.ej. el logotipo arriba y la
            # palabra de abajo, cada uno con su propio renglón "Medidas:"),
            # y antes se perdía por completo cualquiera que no fuera el
            # primero. La interfaz ya sabe mostrar hasta 2 grupos (uno por
            # botón "Aplicar a filas"), así que se juntan los letter_groups
            # de TODOS los conceptos de letras — no solo el primero — y se
            # recorta a 2, de mayor a menor alto (mismo límite que ya
            # aplica para el respaldo geométrico, ver geo_groups arriba).
            letra_pieces = [p for p in detected_pieces if "letra" in (p.get("name") or "").lower()]
            if not letra_pieces:
                letra_pieces = detected_pieces[:1]

            all_groups: List[Dict[str, Any]] = []
            for pi, p in enumerate(letra_pieces):
                for g in p.get("letter_groups") or []:
                    all_groups.append({**g, "_piece_idx": pi, "_piece_page": p.get("page")})
            all_groups.sort(key=lambda g: g.get("height_cm") or 0, reverse=True)
            used_groups = all_groups[:2]
            used_piece_ids = {id(letra_pieces[g["_piece_idx"]]) for g in used_groups}

            # El tiempo se calcula por letra: letras del dibujo x piezas
            # pedidas. Sin letras separables (p.ej. una bandera) cuenta como
            # una sola pieza por cada unidad pedida.
            if used_groups:
                result["width_cm"]  = used_groups[0]["width_cm"]
                result["height_cm"] = used_groups[0]["height_cm"]
                result["letters"] = sum(g["letters"] for g in used_groups) or None
                result["pieces"]  = sum(g["letters"] for g in used_groups) or 1
            else:
                chosen = letra_pieces[0]
                result["width_cm"]  = chosen["width_cm"]
                result["height_cm"] = chosen["height_cm"]
                result["letters"] = chosen["letters"]
                result["pieces"]  = (chosen["letters"] or 1) * chosen["quantity"]
            result["letter_groups"] = used_groups
            chosen_groups = [g for g in used_groups if g.get("_piece_page") == page_index]
            result["skipped_pieces"] = [
                p["name"] for p in detected_pieces if p.get("name") and id(p) not in used_piece_ids
            ]
        else:
            result["width_cm"]  = round(page.rect.width  * PT_TO_CM, 1)
            result["height_cm"] = round(page.rect.height * PT_TO_CM, 1)

        target_max = 1600
        scale = target_max / max(page.rect.width, page.rect.height, 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img_bytes = pix.tobytes("png")
        doc.close()
    except Exception as e:
        result["cdr_error"] = str(e)
        return result

    preview_filename = f"{base}_preview.png"
    cv2.imwrite(os.path.join(folder, preview_filename),
                cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR))
    result["preview_filename"] = preview_filename
    result["img_read"] = True

    if not CV2_AVAILABLE:
        return result

    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    annotated = img.copy()
    if detected_pieces or geo_groups:
        # La cantidad ya viene del texto y de los trazos vectoriales; el
        # conteo por contorno solo aplica al modo de respaldo. Aquí solo se
        # dibujan las letras contadas (un color por grupo de alto) para
        # poder verificarlas a simple vista.
        if not chosen_groups:
            return result
        palette = [(0, 180, 0), (0, 120, 220), (180, 0, 180), (0, 165, 255)]
        i = 0
        for gi, group in enumerate(chosen_groups):
            color = palette[gi % len(palette)]
            for x0, y0, x1, y1 in sorted(group["rects"], key=lambda r: r[0]):
                x, y = int(x0 * scale), int(y0 * scale)
                cw, ch = int((x1 - x0) * scale), int((y1 - y0) * scale)
                i += 1
                cv2.rectangle(annotated, (x, y), (x + cw, y + ch), color, 2)
                cv2.putText(annotated, str(i), (x + 3, max(y - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    else:
        count, boxes = _count_pieces_contours(img)
        result["pieces"] = count
        for i, (x, y, cw, ch) in enumerate(boxes):
            cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (0, 180, 0), 2)
            cv2.putText(annotated, str(i + 1), (x + 3, max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 150, 0), 2)
    annotated_filename = f"{base}_analyzed.png"
    cv2.imwrite(os.path.join(folder, annotated_filename), annotated)
    result["annotated_filename"] = annotated_filename
    return result


def analyze_cdr_file(cdr_path: str) -> Dict[str, Any]:
    """Analiza un archivo .cdr (CorelDRAW).

    No existe un binding de Python confiable para leer .cdr directamente,
    así que se convierte a PDF con LibreOffice (headless) y de ahí se toman
    las medidas exactas de la página directamente de los datos vectoriales,
    igual que antes se hacía con .ai. Se cuentan piezas sobre un render del
    diseño. Solo se analiza la primera página del archivo.
    """
    empty: Dict[str, Any] = {
        "width_cm": None, "height_cm": None,
        "pieces": 0, "annotated_filename": None, "preview_filename": None,
        "img_read": False, "detected_pieces": [],
    }
    if not LIBREOFFICE_AVAILABLE:
        return {**empty, "cdr_error": "LibreOffice no está instalado en el servidor."}
    if not PYMUPDF_AVAILABLE:
        return {**empty, "cdr_error": "PyMuPDF no está instalado."}

    tmp_dir = tempfile.mkdtemp(prefix="cdr_")
    try:
        pdf_path = _convert_cdr_to_pdf(cdr_path, tmp_dir)
        base = os.path.splitext(os.path.basename(cdr_path))[0]
        folder = os.path.dirname(cdr_path)
        return _analyze_pdf_common(pdf_path, base, folder)
    except Exception as e:
        return {**empty, "cdr_error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══ IMAGEN (JPG/PNG) Y PDF ESCANEADO — SIN .CDR ═══════════════════════════════
# Respaldo para cuando no se tiene el .cdr: una imagen plana del diseño con
# las medidas YA DIBUJADAS como píxeles (típico de una captura de pantalla
# de Corel, o un PDF escaneado sin datos vectoriales) — no hay texto real
# que leer ("Medidas: ...") ni trazos vectoriales que re-emparejar, así que
# la escala sale de OCR sobre las cotas dibujadas, y las piezas salen de
# blobs de tinta sólidos (sin el contorno exacto que sí da un .cdr real).
# Por eso el nesting real de corte y el cálculo de metros de LED (que
# necesitan el contorno vectorial exacto) siguen requiriendo el .cdr — esta
# vía solo da medidas y conteo de piezas/letras, igual que la versión
# anterior de esta app (basada en imagen, antes de que existiera el .cdr).

def _digit_cluster_bbox(strip_mask: "np.ndarray", min_area: int = 8) -> Optional[Tuple[int, int, int, int]]:
    """Dentro de una franja binaria (cota de Corel aplanada a píxeles),
    ubica el bounding box del NÚMERO en sí, descartando la línea de cota y
    sus flechas/ticks — si se manda la franja entera a Tesseract (línea +
    flechas + número) el OCR falla casi siempre, aunque el número se vea
    perfectamente legible a simple vista (la línea larga y los ticks
    rompen la segmentación de texto de Tesseract).

    Se descartan los componentes conectados que claramente NO son un
    dígito: muy anchos/altos (la línea misma), poco rellenos (un trazo
    delgado tipo línea/tick, a diferencia del relleno macizo de un
    dígito) o muy alargados (un segmento de línea o un tick). Lo que
    queda son los trazos de los dígitos — se agrupan por cercanía
    horizontal (un número son varios dígitos juntos) y se toma el grupo
    con más tinta (el real, no ruido disperso de líneas punteadas)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(strip_mask, connectivity=8)
    strip_h, strip_w = strip_mask.shape[:2]
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
        if area < min_area or w > 0.25 * strip_w or h > 0.6 * strip_h:
            continue
        fill = area / max(1, w * h)
        aspect = max(w, h) / max(1, min(w, h))
        if fill < 0.2 or aspect > 4:
            continue
        comps.append((x, y, w, h))
    if not comps:
        return None
    comps.sort(key=lambda c: c[0])
    clusters = [[comps[0]]]
    for c in comps[1:]:
        prev = clusters[-1][-1]
        if c[0] - (prev[0] + prev[2]) > 15:
            clusters.append([c])
        else:
            clusters[-1].append(c)
    best = max(clusters, key=lambda cl: sum(c[2] * c[3] for c in cl))
    x0 = min(c[0] for c in best)
    y0 = min(c[1] for c in best)
    x1 = max(c[0] + c[2] for c in best)
    y1 = max(c[1] + c[3] for c in best)
    return x0, y0, x1, y1


def _ocr_number_in_strip(gray_strip: "np.ndarray", rotate: int = 0) -> Optional[int]:
    """Lee un número entero dibujado como píxeles dentro de una franja de
    imagen (una cota de Corel exportada a imagen, no texto real) — primero
    recorta justo al número (ver _digit_cluster_bbox, necesario porque la
    línea/flechas de la cota rompen el OCR) y después prueba varias
    configuraciones de Tesseract, porque el trazo es chico y antialiased a
    la resolución típica de una captura de pantalla."""
    if gray_strip is None or gray_strip.size == 0 or min(gray_strip.shape[:2]) < 3:
        return None
    if rotate == -90:
        gray_strip = cv2.rotate(gray_strip, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotate == 90:
        gray_strip = cv2.rotate(gray_strip, cv2.ROTATE_90_CLOCKWISE)

    _, mask = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bbox = _digit_cluster_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    pad = 4
    crop = gray_strip[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    if crop.size == 0:
        return None
    big = cv2.resize(crop, None, fx=6, fy=6, interpolation=cv2.INTER_LANCZOS4)
    _, bin_img = cv2.threshold(big, 180, 255, cv2.THRESH_BINARY)
    for psm in (7, 6, 8, 11):
        txt = pytesseract.image_to_string(
            bin_img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        ).strip()
        if txt.isdigit():
            return int(txt)
    return None


def _ink_components(gray: "np.ndarray") -> List[Tuple[int, int, int, int, int]]:
    """Componentes conectados de tinta (no fondo) de una imagen en gris —
    (x0, y0, x1, y1, área), de más grande a más chico. Base para separar
    contenido real (letras/logo) de anotaciones (cotas, flechas, líneas
    guía) — ver _split_content_and_scale. Umbral de Otsu (no uno fijo):
    se adapta al contraste de cada imagen (foto, captura, export limpio)
    en vez de asumir siempre fondo blanco puro."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [
        (int(stats[i][0]), int(stats[i][1]), int(stats[i][0] + stats[i][2]),
         int(stats[i][1] + stats[i][3]), int(stats[i][4]))
        for i in range(1, n)
    ]
    comps.sort(key=lambda c: c[4], reverse=True)
    return comps


def _split_content_and_scale(gray: "np.ndarray") -> Optional[Dict[str, Any]]:
    """De una imagen plana del diseño (sin .cdr), separa el CONTENIDO real
    (letras/logo: blobs grandes y macizos) de las ANOTACIONES (cotas,
    flechas, líneas guía: blobs chicos, o grandes pero huecos/delgados),
    ubica la cota de ancho (arriba del contenido) y la de alto (a la
    izquierda, girada 90°) por OCR, y arma la escala real mm/px.

    La escala se calcula con la cota de ANCHO: un remate o acento
    decorativo que sobresale por arriba del resto (p.ej. el adorno de un
    logo) puede quedar pegado a la línea de cota vertical y fusionarse con
    ella en un solo blob — si eso pasa, se excluye como "anotación" (poco
    relleno) y el contenido detectado sale más bajo de lo real, pero el
    ANCHO no se ve afectado (el remate no cambia el borde izquierdo/derecho
    del diseño). Por eso el alto no se usa para la escala, solo como cruce
    de verificación (`verified`).

    None si no se pudo leer ninguna cota (el usuario deberá ingresar las
    medidas a mano, igual que cuando un .cdr no trae "Medidas: ...")."""
    comps = _ink_components(gray)
    if not comps:
        return None
    max_area = comps[0][4]
    floor = max(500, 0.1 * max_area)
    content = [
        c for c in comps
        if c[4] >= floor and c[4] / max(1, (c[2] - c[0]) * (c[3] - c[1])) >= 0.2
    ]
    if not content:
        return None
    cx0 = min(c[0] for c in content)
    cy0 = min(c[1] for c in content)
    cx1 = max(c[2] for c in content)
    cy1 = max(c[3] for c in content)

    pad = 2
    top_strip = gray[0:max(0, cy0 - pad), :]
    # La cota de alto (número + línea + flechas) puede quedar recostada
    # casi debajo del arranque del contenido si el trazo tiene una
    # concavidad ahí (p.ej. el hueco de un acento/swash) — un corte justo
    # en cx0 la parte a la mitad. Se da un margen generoso hacia la
    # derecha; _digit_cluster_bbox igual descarta cualquier trazo real que
    # se cuele (por tamaño/relleno), así que no hace falta ser exacto acá.
    left_margin = max(50, round(0.08 * (cx1 - cx0)))
    left_strip = gray[0:cy1 + pad, 0:min(gray.shape[1], cx0 + left_margin)]

    width_mm = _ocr_number_in_strip(top_strip)
    height_mm = _ocr_number_in_strip(left_strip, rotate=90)
    if height_mm is None:
        height_mm = _ocr_number_in_strip(left_strip, rotate=-90)
    if width_mm is None and height_mm is None:
        return None

    content_w_px, content_h_px = cx1 - cx0, cy1 - cy0
    if width_mm and content_w_px > 0:
        mm_per_px = width_mm / content_w_px
    else:
        mm_per_px = height_mm / content_h_px

    verified = False
    if width_mm and height_mm and content_h_px > 0:
        height_mm_check = content_h_px * mm_per_px
        verified = abs(height_mm_check - height_mm) / height_mm < 0.15

    return {
        "content_bbox": (cx0, cy0, cx1, cy1),
        "content_boxes": [(c[0], c[1], c[2], c[3]) for c in content],
        "mm_per_px": mm_per_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "verified": verified,
    }


def _ocr_scale_for_page(page: "fitz.Page", target_max: int = 2000) -> Optional[float]:
    """Respaldo cuando un .cdr no trae NINGÚN texto real en toda la
    página — ni "Medidas: A x B" ni una sola cota suelta (get_text() no
    encuentra nada) — porque el diseñador convirtió hasta las cotas a
    curvas (visto real: zacatecas.cdr). Esas cotas SÍ se ven en un render
    de la página aunque no sean texto, así que se reusa el mismo OCR del
    modo sin .cdr (ver _split_content_and_scale) sobre un render a buena
    resolución, en vez de rendirse al tamaño crudo en puntos del PDF (que
    no tiene ninguna relación garantizada con el tamaño real — confirmado
    real: para zacatecas.cdr daba 2.4cm de alto cuando el alto real, leído
    de la cota dibujada, es 70.2cm).

    Devuelve mm por PUNTO del PDF (no por píxel del render) para que se
    pueda usar igual que `mm_per_pt` en el resto de _extract_pieces_geometric
    — None si no se pudo leer ninguna cota tampoco por OCR."""
    if not CV2_AVAILABLE or not TESSERACT_AVAILABLE:
        return None
    scale = target_max / max(page.rect.width, page.rect.height, 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), dtype=np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_info = _split_content_and_scale(gray)
    if scale_info is None:
        return None
    # El render se hizo a `scale` píxeles por cada punto del PDF, así que
    # mm/pt = mm/px del render × (px del render / pt del PDF) = mm_per_px × scale.
    return scale_info["mm_per_px"] * scale


def _cluster_boxes_by_height_px(
    boxes: List[Tuple[int, int, int, int]], mm_per_px: float, gap_ratio: float = 1.4,
) -> List[Dict[str, Any]]:
    """_cluster_boxes_by_height sobre cajas en píxeles (de una imagen
    plana, sin .cdr) con una escala mm/px uniforme ya conocida — ver
    _split_content_and_scale. A diferencia de _cluster_rects_by_height,
    una sola caja sí arma un grupo de 1 (una imagen con una sola pieza
    detectada es un caso normal, no ruido)."""
    return _cluster_boxes_by_height(boxes, mm_per_px, gap_ratio, single_box_ok=True)


def _analyze_image_array(img: "np.ndarray") -> Dict[str, Any]:
    """Núcleo de análisis de una imagen YA CARGADA en memoria (sin leer ni
    escribir archivos) — compartido entre analyze_image_file (JPG/PNG
    subido directo) y el respaldo de PDF escaneado en analyze_pdf_file.
    Devuelve el dict de resultado (sin preview_filename/annotated_filename,
    eso lo arma quien llama) más 'annotated_img' para que decida cómo
    guardarla."""
    result: Dict[str, Any] = {
        "width_cm": None, "height_cm": None, "pieces": 0,
        "letters": None, "letter_groups": [], "detected_pieces": [],
        "skipped_pieces": [],
    }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_info = _split_content_and_scale(gray)
    annotated = img.copy()
    if scale_info is None:
        result["cdr_error"] = "No se pudieron leer las cotas de la imagen. Ingresa las medidas manualmente."
        result["annotated_img"] = annotated
        return result

    groups = _cluster_boxes_by_height_px(scale_info["content_boxes"], scale_info["mm_per_px"])
    if not groups:
        cx0, cy0, cx1, cy1 = scale_info["content_bbox"]
        groups = [{
            "letters": 1,
            "height_cm": round((cy1 - cy0) * scale_info["mm_per_px"] / 10, 1),
            "width_cm": round((cx1 - cx0) * scale_info["mm_per_px"] / 10, 1),
            "rects": [[cx0, cy0, cx1, cy1]],
            "pieces": [],
        }]
    for g in groups:
        g["height_verified"] = scale_info["verified"]

    used = groups[:2]
    result["letter_groups"] = used
    result["letters"] = sum(g["letters"] for g in used) or None
    result["width_cm"] = used[0]["width_cm"]
    result["height_cm"] = used[0]["height_cm"]
    result["pieces"] = result["letters"] or 1
    if not scale_info["verified"] and scale_info["width_mm"] and scale_info["height_mm"]:
        result["scale_warning"] = (
            "La cota de alto no coincidió del todo con el dibujo (puede haber un remate o "
            "acento que sobresale) — revisa las medidas antes de aplicar."
        )

    palette = [(0, 180, 0), (0, 120, 220), (180, 0, 180), (0, 165, 255)]
    for gi, group in enumerate(used):
        color = palette[gi % len(palette)]
        for i, (x0, y0, x1, y1) in enumerate(sorted(group["rects"], key=lambda r: r[0])):
            cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
            cv2.putText(annotated, str(i + 1), (x0 + 3, max(y0 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    result["annotated_img"] = annotated
    return result


def analyze_image_file(img_path: str) -> Dict[str, Any]:
    """Analiza una imagen plana (JPG/PNG) del diseño — SIN .cdr: así
    funcionaba la versión anterior de esta app. No hay datos vectoriales
    (el nesting real de corte y el cálculo de metros de LED siguen
    necesitando el .cdr para el contorno exacto), pero sí da medidas
    reales (OCR sobre las cotas YA dibujadas en la imagen, como las
    exporta Corel al capturar pantalla) y conteo de piezas/letras por
    contorno."""
    empty: Dict[str, Any] = {
        "width_cm": None, "height_cm": None, "pieces": 0,
        "annotated_filename": None, "preview_filename": None,
        "img_read": False, "detected_pieces": [], "letter_groups": [],
        "skipped_pieces": [],
    }
    if not CV2_AVAILABLE:
        return {**empty, "cdr_error": "OpenCV no está instalado en el servidor."}
    if not TESSERACT_AVAILABLE:
        return {**empty, "cdr_error": "Tesseract OCR no está instalado en el servidor."}

    img = cv2.imread(img_path)
    if img is None:
        return {**empty, "cdr_error": "No se pudo leer la imagen."}

    base = os.path.splitext(os.path.basename(img_path))[0]
    folder = os.path.dirname(img_path)
    preview_filename = f"{base}_preview.png"
    cv2.imwrite(os.path.join(folder, preview_filename), img)

    core = _analyze_image_array(img)
    annotated = core.pop("annotated_img")
    annotated_filename = f"{base}_analyzed.png"
    cv2.imwrite(os.path.join(folder, annotated_filename), annotated)

    result = {**empty, **core}
    result["img_read"] = True
    result["preview_filename"] = preview_filename
    result["annotated_filename"] = annotated_filename
    return result


def analyze_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """PDF subido directo (sin pasar por .cdr/LibreOffice). Si trae datos
    vectoriales reales (exportado desde Corel/Illustrator con los trazos
    intactos) se analiza igual que un .cdr convertido. Si es un PDF
    ESCANEADO/aplanado (una sola imagen de fondo, sin page.get_drawings())
    se cae al mismo camino de OCR+contornos que una imagen suelta — ver
    analyze_image_file."""
    empty: Dict[str, Any] = {
        "width_cm": None, "height_cm": None, "pieces": 0,
        "annotated_filename": None, "preview_filename": None,
        "img_read": False, "detected_pieces": [], "letter_groups": [],
        "skipped_pieces": [],
    }
    if not PYMUPDF_AVAILABLE:
        return {**empty, "cdr_error": "PyMuPDF no está instalado."}

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    folder = os.path.dirname(pdf_path)
    try:
        with fitz.open(pdf_path) as doc:
            has_vectors = any(page.get_drawings() for page in doc)
    except Exception as e:
        return {**empty, "cdr_error": str(e)}

    if has_vectors:
        return _analyze_pdf_common(pdf_path, base, folder)

    if not CV2_AVAILABLE:
        return {**empty, "cdr_error": "OpenCV no está instalado en el servidor."}
    if not TESSERACT_AVAILABLE:
        return {**empty, "cdr_error": "Tesseract OCR no está instalado en el servidor."}
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[0]
            target_max = 1600
            render_scale = target_max / max(page.rect.width, page.rect.height, 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
            img_bytes = pix.tobytes("png")
    except Exception as e:
        return {**empty, "cdr_error": str(e)}

    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    preview_filename = f"{base}_preview.png"
    cv2.imwrite(os.path.join(folder, preview_filename), img)

    core = _analyze_image_array(img)
    annotated = core.pop("annotated_img")
    annotated_filename = f"{base}_analyzed.png"
    cv2.imwrite(os.path.join(folder, annotated_filename), annotated)

    result = {**empty, **core}
    result["img_read"] = True
    result["preview_filename"] = preview_filename
    result["annotated_filename"] = annotated_filename
    return result
