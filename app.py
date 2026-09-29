import json
import os
import re
import shutil
import subprocess
import tempfile

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, send_from_directory, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
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

UPLOAD_FOLDER = "uploads"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "times.json")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "change-me-lumicolor-2025"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///calculadora.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ══ DATABASE ══════════════════════════════════════════════════════════════════

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Inicia sesión para continuar."


class User(UserMixin, db.Model):
    __tablename__ = "lc_user"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin      = db.Column(db.Boolean, default=False, nullable=False)
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    calculations  = db.relationship("Calculation", backref="user", lazy=True,
                                   cascade="all, delete-orphan")


class Calculation(db.Model):
    __tablename__ = "lc_calculation"
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey("lc_user.id"), nullable=False)
    name         = db.Column(db.String(120), nullable=False, default="Sin nombre")
    created_at   = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    rows_json    = db.Column(db.Text, nullable=False)
    params_json  = db.Column(db.Text, nullable=False)
    results_json = db.Column(db.Text, nullable=False)
    image_url    = db.Column(db.String(512), nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.template_filter("fromjson")
def fromjson_filter(value):
    return json.loads(value)


# ══ CATALOG ═══════════════════════════════════════════════════════════════════

def load_catalog() -> Dict[str, Any]:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_group_time(catalog: Dict, area: str, subprocess: str, group: str) -> float:
    try:
        val = catalog[area][subprocess][group]
        if isinstance(val, dict):
            return float(val["tiempo"])
        return float(val)
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"No se encontró tiempo para {area} > {subprocess} > {group}: {e}")


def _avg_height_from_desc(desc: str) -> Optional[float]:
    if not desc or desc.strip() in ("-", "–", "NA", ""):
        return None
    parts = desc.split("×")
    segment = parts[1] if len(parts) >= 2 else parts[0]
    nums = re.findall(r"\d+(?:\.\d+)?", segment)
    if not nums:
        return None
    return sum(float(n) for n in nums) / len(nums)


def compute_regression(catalog: Dict, area: str, subprocess: str) -> Optional[Tuple[float, float]]:
    groups = catalog.get(area, {}).get(subprocess, {})
    pts: List[Tuple[float, float]] = []
    for datos in groups.values():
        desc = datos.get("desc", "") if isinstance(datos, dict) else ""
        h = _avg_height_from_desc(desc)
        t = float(datos["tiempo"]) if isinstance(datos, dict) else float(datos)
        if h is not None:
            pts.append((h, t))
    if len(pts) < 2:
        return None
    n   = len(pts)
    sx  = sum(p[0] for p in pts)
    sy  = sum(p[1] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    sx2 = sum(p[0] ** 2 for p in pts)
    den = n * sx2 - sx * sx
    if den == 0:
        return None
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    return slope, intercept


# ══ MATERIAL SUGGESTIONS ══════════════════════════════════════════════════════

_MATERIAL_RULES = [
    (lambda area, sub: area == "CORTE" and "acrílico" in sub.lower(),
     "CORTE", "Acrílico 3 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "CORTE" and "lámina" in sub.lower(),
     "CORTE", "Aluminio blanco C.19", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "CORTE" and "trovicel" in sub.lower(),
     "CORTE", "Trovicel 10 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "TRIMCAP",
     "TRIMCAP", 'Silvatrim 1"', "El color varía"),
    (lambda area, sub: area == "TERMOFORMADO",
     "TERMOFORMADO", "Acrílico blanco 3 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "ROTULACIÓN",
     "ROTULACIÓN", "Vinil impreso", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "TENSADO",
     "TENSADO", "Lona translúcida", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "PINTURA",
     "PINTURA", "Pintura para cantos", "Brillante / Mate / Semi mate"),
    (lambda area, sub: area == "PINTURA",
     "PINTURA", "Fondear estructuras", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "PINTURA",
     "PINTURA", "Pintura interior de letras", "El color depende de la variabilidad del componente"),
    (lambda area, sub: area == "PINTURA",
     "PINTURA", "Bicapa", "Mate / Transparente"),
    (lambda area, sub: area == "ILUMINACIÓN" and sub == "Neón",
     "ILUMINACIÓN", "NEON FLEX BLANCO FRIO SIMETRICA", ""),
    (lambda area, sub: area == "ILUMINACIÓN" and sub == "Neón",
     "ILUMINACIÓN", "NEON FLEX BLANCO CALIDO SIMETRICA", ""),
]


def get_material_suggestions(rows: List[Dict], params: Dict = None) -> List[Dict]:
    seen: set = set()
    suggestions: List[Dict] = []
    for row in rows:
        area = row.get("area", "")
        sub  = row.get("subprocess", "")
        for match_fn, tag, material, note in _MATERIAL_RULES:
            if material not in seen and match_fn(area, sub):
                seen.add(material)
                suggestions.append({"area": tag, "material": material, "note": note})
    return suggestions


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
_MEAS_RE = re.compile(r"(\d[\d.,]*)\s*[x×]\s*(\d[\d.,]*)\s*mm", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(concepto|medidas|materiales|cantidad)\s*:?\s*$", re.IGNORECASE)
_QTY_RE = re.compile(r"(\d+)\s*pieza", re.IGNORECASE)
_CARAS_RE = re.compile(r"(\d+)\s*cara", re.IGNORECASE)


def _parse_num(s: str) -> float:
    return float(s.replace(",", "."))


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
    for d in page.get_drawings():
        col, r = d.get("fill"), d["rect"]
        if col is None or min(col) > 0.95:
            continue
        if not (y_lo <= (r.y0 + r.y1) / 2 < y_hi):
            continue
        rects.append(r)
    if len(rects) < 2:
        return []
    max_area = max(r.width * r.height for r in rects)
    rects = [r for r in rects if r.width * r.height >= 0.05 * max_area]
    return _drop_isolated_rects(rects)


def _cluster_rects_by_height(rects: List["fitz.Rect"], mm_per_pt: float,
                              gap_ratio: float = 1.4) -> List[Dict[str, Any]]:
    """Agrupa trazos por su ALTO real (no por color) usando la escala local
    del dibujo (`mm_per_pt`), no la escala de la página.

    Las hojas técnicas dibujan la pieza a escala reducida (p.ej. "Escala
    1:40"), así que el alto de cada trazo en puntos del PDF no es su alto
    real — hay que convertirlo con la escala de ESE dibujo, no con la de la
    página completa.

    En el PDF convertido cada letra queda como un trazo cerrado independiente
    aunque se toque con la vecina (cosa que un conteo por imagen fusiona,
    p.ej. la "i" y la "f" en cursiva). Un mismo concepto puede traer dos
    tamaños de letra (p.ej. un título grande y un subtítulo chico, del mismo
    color) — separarlas por alto en vez de por color detecta ese caso: se
    ordenan los trazos por alto y se corta cada vez que hay un salto de
    tamaño mayor a `gap_ratio` entre un trazo y el siguiente.
    """
    if len(rects) < 2:
        return []
    rects = sorted(rects, key=lambda r: r.height)
    clusters = [[rects[0]]]
    for r in rects[1:]:
        if r.height / clusters[-1][-1].height > gap_ratio:
            clusters.append([r])
        else:
            clusters[-1].append(r)

    groups = []
    for cluster in clusters:
        # Un elemento ajeno (p.ej. un ícono de la cabecera de la hoja) puede
        # caer en el mismo rango de alto que las letras por casualidad, pero
        # queda lejos de ellas en la página. Si eso pasa, se separa por
        # cercanía vertical y solo se conserva la línea de letras real (el
        # subgrupo más grande).
        if len(cluster) >= 3:
            avg_h_all = sum(r.height for r in cluster) / len(cluster)
            by_y = sorted(cluster, key=lambda r: (r.y0 + r.y1) / 2)
            sub = [[by_y[0]]]
            for r in by_y[1:]:
                prev_c = (sub[-1][-1].y0 + sub[-1][-1].y1) / 2
                cur_c = (r.y0 + r.y1) / 2
                if abs(cur_c - prev_c) > 3 * avg_h_all:
                    sub.append([r])
                else:
                    sub[-1].append(r)
            cluster = max(sub, key=len)

        if len(cluster) < 2:
            continue  # un solo trazo de ese tamaño: probablemente ruido
        avg_h = sum(r.height for r in cluster) / len(cluster)
        avg_w = sum(r.width for r in cluster) / len(cluster)
        cluster_by_x = sorted(cluster, key=lambda r: r.x0)
        groups.append({
            "letters": len(cluster),
            "height_cm": round(avg_h * mm_per_pt / 10, 1),
            # Ancho promedio de ESTE grupo (a partir de sus propios trazos),
            # no el ancho total del letrero entre la cantidad de letras: eso
            # no corresponde al ancho real de ninguna letra.
            "width_cm": round(avg_w * mm_per_pt / 10, 1),
            "rects": [[r.x0, r.y0, r.x1, r.y1] for r in cluster],
            # Medida real de cada trazo individual (izquierda a derecha), por
            # si el promedio del grupo mezcla piezas de tamaño distinto
            # (p.ej. el logo y las letras en Mifel) — solo para mostrar en
            # el detalle de la interfaz, el cálculo de tiempo sigue usando
            # el alto/ancho del grupo de arriba.
            "pieces": [
                {"width_cm": round(r.width * mm_per_pt / 10, 1),
                 "height_cm": round(r.height * mm_per_pt / 10, 1)}
                for r in cluster_by_x
            ],
        })
    groups.sort(key=lambda g: -g["height_cm"])
    return groups


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
                compact = re.sub(r"\s", "", btext)
                # tope de 5 dígitos: evita que dos cotas vecinas pegadas en
                # un solo bloque de texto (p.ej. "56184311") se lean como un
                # solo valor absurdo.
                if compact.isdigit() and len(compact) <= 5:
                    loose_digits.append({
                        "value": compact,
                        "y_center": (by0 + by1) / 2,
                        "is_wide": (bx1 - bx0) > (by1 - by0),
                    })

            # Cada concepto ocupa la franja vertical hasta el punto medio con
            # el concepto vecino: ahí caen las letras de su dibujo.
            anchors = sorted((b[1] + b[3]) / 2 for b in meas_blocks)

            for mb in meas_blocks:
                x0, y0, x1, y1, text = mb[0], mb[1], mb[2], mb[3], mb[4]
                m = _MEAS_RE.search(text)
                n1, n2 = _parse_num(m.group(1)), _parse_num(m.group(2))
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
            blocks = [b for b in page.get_text("blocks") if b[4].strip()]
            loose_digits = []
            for b in blocks:
                bx0, by0, bx1, by1, btext = b[0], b[1], b[2], b[3], b[4]
                compact = re.sub(r"\s", "", btext)
                # tope de 5 dígitos: descarta bloques donde el conversor
                # pegó dos cotas vecinas en un solo texto (p.ej. "56184311"
                # en vez de "5618" y "4311" separadas) — un valor real de
                # hasta 99999 mm (~100 m) alcanza de sobra para un letrero.
                if compact.isdigit() and len(compact) <= 5:
                    loose_digits.append({
                        "value": compact,
                        "y_center": (by0 + by1) / 2,
                        "is_wide": (bx1 - bx0) > (by1 - by0),
                    })

            raw_rects = []
            for d in page.get_drawings():
                col, r = d.get("fill"), d["rect"]
                if col is None or min(col) > 0.95:
                    continue
                raw_rects.append(r)
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
                    continue  # sin cotas confiables cerca: no se puede calibrar este diseño

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


def _convert_cdr_to_pdf(cdr_path: str, out_dir: str) -> str:
    """Convierte un .cdr a PDF con LibreOffice (filtro libcdr) en modo headless."""
    proc = subprocess.run(
        [LIBREOFFICE_BIN, "--headless", "--norestore", "--convert-to", "pdf",
         "--outdir", out_dir, cdr_path],
        capture_output=True, timeout=120,
    )
    base = os.path.splitext(os.path.basename(cdr_path))[0]
    pdf_path = os.path.join(out_dir, f"{base}.pdf")
    if not os.path.exists(pdf_path):
        detail = proc.stderr.decode(errors="ignore") or proc.stdout.decode(errors="ignore")
        raise RuntimeError(f"LibreOffice no pudo convertir el .cdr: {detail.strip() or 'sin detalle'}")
    return pdf_path


def analyze_cdr_file(cdr_path: str) -> Dict[str, Any]:
    """Analiza un archivo .cdr (CorelDRAW).

    No existe un binding de Python confiable para leer .cdr directamente,
    así que se convierte a PDF con LibreOffice (headless) y de ahí se toman
    las medidas exactas de la página directamente de los datos vectoriales,
    igual que antes se hacía con .ai. Se cuentan piezas sobre un render del
    diseño. Solo se analiza la primera página del archivo.
    """
    result: Dict[str, Any] = {
        "width_cm": None, "height_cm": None,
        "pieces": 0, "annotated_filename": None, "preview_filename": None,
        "img_read": False, "detected_pieces": [],
    }
    if not LIBREOFFICE_AVAILABLE:
        result["cdr_error"] = "LibreOffice no está instalado en el servidor."
        return result
    if not PYMUPDF_AVAILABLE:
        result["cdr_error"] = "PyMuPDF no está instalado."
        return result

    chosen_groups: List[Dict[str, Any]] = []
    tmp_dir = tempfile.mkdtemp(prefix="cdr_")
    try:
        pdf_path = _convert_cdr_to_pdf(cdr_path, tmp_dir)
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
        elif detected_pieces:
            # Un .cdr de producción suele traer, además del set de letras,
            # otras piezas independientes (p.ej. "Bandera ATM"). Esta app
            # calcula tiempos de letras, así que se prioriza la pieza cuyo
            # Concepto dice "letra*"; si ninguna califica, se usa la primera.
            chosen = next(
                (p for p in detected_pieces if "letra" in (p.get("name") or "").lower()),
                detected_pieces[0],
            )
            result["width_cm"]  = chosen["width_cm"]
            result["height_cm"] = chosen["height_cm"]
            # El tiempo se calcula por letra: letras del dibujo x piezas
            # pedidas. Sin letras separables (p.ej. una bandera) cuenta como
            # una sola pieza por cada unidad pedida. Si el concepto trae dos
            # tamaños de letra (p.ej. título grande + subtítulo chico),
            # letter_groups trae uno por alto detectado, para calcular cada
            # tamaño por separado en vez de mezclarlos en un solo conteo.
            result["letters"] = chosen["letters"]
            result["letter_groups"] = chosen["letter_groups"]
            result["pieces"]  = (chosen["letters"] or 1) * chosen["quantity"]
            if chosen["page"] == page_index:
                chosen_groups = chosen["letter_groups"]
            result["skipped_pieces"] = [
                p["name"] for p in detected_pieces if p is not chosen and p.get("name")
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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    base = os.path.splitext(os.path.basename(cdr_path))[0]
    folder = os.path.dirname(cdr_path)
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


# ══ ROUTES ════════════════════════════════════════════════════════════════════

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    file = request.files.get("design")
    if not file or not file.filename:
        return jsonify({"error": "No se recibió archivo"}), 400
    if not file.filename.lower().endswith(".cdr"):
        return jsonify({"error": "Solo se aceptan archivos .cdr (CorelDRAW)"}), 400
    if not LIBREOFFICE_AVAILABLE:
        return jsonify({"error": "LibreOffice no está instalado en el servidor"}), 500
    if not PYMUPDF_AVAILABLE:
        return jsonify({"error": "PyMuPDF no está instalado en el servidor"}), 500

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    data = analyze_cdr_file(path)
    annotated_filename = data.pop("annotated_filename", None)
    preview_filename    = data.pop("preview_filename", None)
    preview_url   = (url_for("uploaded_file", filename=preview_filename)
                      if preview_filename else None)
    annotated_url = (url_for("uploaded_file", filename=annotated_filename)
                      if annotated_filename else None)
    return jsonify({
        **data,
        "preview_url":     preview_url,
        "annotated_url":   annotated_url,
        "image_url":       annotated_url or preview_url,
        "cv2_available":   CV2_AVAILABLE,
        "pymupdf_available": PYMUPDF_AVAILABLE,
        "libreoffice_available": LIBREOFFICE_AVAILABLE,
        "file_saved":      os.path.exists(path),
        "filename":        os.path.basename(path),
    })


# ── AUTH ──────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for("index"))
        flash("Usuario o contraseña incorrectos.")
    return render_template("auth/login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ── ADMIN USUARIOS ────────────────────────────────────────────────────────────

@app.route("/admin/usuarios", methods=["GET", "POST"])
@login_required
def admin_users():
    if not current_user.is_admin:
        return redirect(url_for("index"))
    if request.method == "POST":
        action = request.form.get("action")
        if action == "create":
            uname    = request.form.get("username", "").strip()
            pwd      = request.form.get("password", "")
            is_admin = bool(request.form.get("is_admin"))
            if uname and pwd:
                if not User.query.filter_by(username=uname).first():
                    db.session.add(User(
                        username=uname,
                        password_hash=generate_password_hash(pwd),
                        is_admin=is_admin,
                    ))
                    db.session.commit()
                    flash(f"Usuario '{uname}' creado correctamente.")
                else:
                    flash(f"El usuario '{uname}' ya existe.")
            else:
                flash("Nombre de usuario y contraseña son obligatorios.")
        elif action == "delete":
            uid = request.form.get("user_id", type=int)
            if uid and uid != current_user.id:
                user = db.session.get(User, uid)
                if user:
                    db.session.delete(user)
                    db.session.commit()
                    flash(f"Usuario '{user.username}' eliminado.")
    users = User.query.order_by(User.created_at).all()
    return render_template("admin/usuarios.html", users=users)


# ── CALCULADORA ───────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    catalog = load_catalog()

    if request.method == "POST":
        hours_per_day     = float(request.form.get("hours_per_day", 9))
        cushion_pct       = float(request.form.get("cushion_pct", 10))
        packaging_minutes = float(request.form.get("packaging_minutes", 40))
        colores_pintura   = max(1, int(request.form.get("colores_pintura", 1)))
        secado_minutes    = float(request.form.get("secado_minutes", 120)) * colores_pintura
        calc_name         = request.form.get("calc_name", "").strip() or "Cálculo sin nombre"


        areas        = request.form.getlist("areas[]")
        subprocesses = request.form.getlist("subprocesses[]")
        groups       = request.form.getlist("groups[]")
        quantities   = request.form.getlist("quantities[]")
        widths       = request.form.getlist("widths[]")
        heights      = request.form.getlist("heights[]")

        if not areas:
            flash("Agrega al menos una área productiva.")
            return redirect(request.url)

        rows: List[Dict] = []
        errors: List[str] = []

        # Pad lists to match row count
        widths  = widths  + [""] * max(0, len(areas) - len(widths))
        heights = heights + [""] * max(0, len(areas) - len(heights))

        for i, (area, subprocess, group, qty_str, width_str, height_str) in enumerate(
                zip(areas, subprocesses, groups, quantities, widths, heights)):
            if not area or not subprocess or not group:
                errors.append(f"Fila {i + 1}: selecciona área, subproceso y grupo.")
                continue
            try:
                qty = float(qty_str) if qty_str else 1.0
                # Ensamble sin flange con ancho conocido: fórmula calibrada
                if area == "ENSAMBLE" and subprocess == "Ensamble sin flange" and width_str:
                    ancho = float(width_str)
                    mins = (0.2545 + 0.01067 * ancho) * 60  # hrs → min
                    group = f"Formula({ancho:.1f}cm)"
                # Regresión lineal: medida fuera del rango conocido
                elif group == "Regresión" and height_str:
                    h = float(height_str)
                    reg = compute_regression(catalog, area, subprocess)
                    if reg:
                        slope, intercept = reg
                        mins = max(0.0, slope * h + intercept)
                        group = f"Regresión lineal ({h:.0f} cm)"
                    else:
                        first = next(iter(catalog.get(area, {}).get(subprocess, {})), None)
                        if first:
                            mins = get_group_time(catalog, area, subprocess, first)
                            group = first
                        else:
                            errors.append(f"Fila {i + 1}: sin datos de regresión para {area}.")
                            continue
                else:
                    mins = get_group_time(catalog, area, subprocess, group)
                rows.append({
                    "area": area,
                    "subprocess": subprocess,
                    "group": group,
                    "quantity": qty,
                    "mins_per_piece": round(mins, 2),
                    "total_mins": round(mins * qty, 2),
                })
            except ValueError as e:
                errors.append(str(e))

        if errors:
            for e in errors:
                flash(e)
            return redirect(request.url)

        production_minutes       = sum(r["total_mins"] for r in rows) + secado_minutes + packaging_minutes
        total_hours              = production_minutes / 60.0
        total_hours_with_cushion = total_hours * (1 + cushion_pct / 100)

        params = {
            "hours_per_day":     hours_per_day,
            "cushion_pct":       cushion_pct,
            "packaging_minutes": packaging_minutes,
            "secado_minutes":    secado_minutes,
            "colores_pintura":   colores_pintura,
        }
        results = {
            "production_minutes":       production_minutes,
            "total_hours":              total_hours,
            "total_hours_with_cushion": total_hours_with_cushion,
            "days":                     total_hours / hours_per_day,
            "days_with_cushion":        total_hours_with_cushion / hours_per_day,
        }

        calc = Calculation(
            user_id=current_user.id,
            name=calc_name,
            rows_json=json.dumps(rows),
            params_json=json.dumps(params),
            results_json=json.dumps(results),
            image_url=request.form.get("image_url") or None,
        )
        db.session.add(calc)
        db.session.commit()
        return redirect(url_for("result", calc_id=calc.id))

    return render_template("index.html", catalog=catalog)


@app.route("/result/<int:calc_id>")
@login_required
def result(calc_id):
    calc = db.session.get(Calculation, calc_id)
    if calc is None:
        flash("Cálculo no encontrado.")
        return redirect(url_for("index"))
    rows      = json.loads(calc.rows_json)
    params    = json.loads(calc.params_json)
    results   = json.loads(calc.results_json)
    materials = get_material_suggestions(rows, params)
    return render_template("result.html", calc=calc, rows=rows, materials=materials, **params, **results)


@app.route("/result/<int:calc_id>/print")
@login_required
def result_print(calc_id):
    calc = db.session.get(Calculation, calc_id)
    if calc is None:
        return "Cálculo no encontrado", 404
    rows      = json.loads(calc.rows_json)
    params    = json.loads(calc.params_json)
    results   = json.loads(calc.results_json)
    materials = get_material_suggestions(rows, params)
    return render_template("print_result.html", calc=calc, rows=rows, materials=materials, **params, **results)


@app.route("/result/<int:calc_id>/download")
@login_required
def result_download(calc_id):
    from xhtml2pdf import pisa
    from io import BytesIO
    calc = db.session.get(Calculation, calc_id)
    if calc is None:
        return "Cálculo no encontrado", 404
    rows      = json.loads(calc.rows_json)
    params    = json.loads(calc.params_json)
    results   = json.loads(calc.results_json)
    materials = get_material_suggestions(rows, params)
    html = render_template("print_result.html", calc=calc, rows=rows, materials=materials,
                           base_url=request.host_url, **params, **results)
    buf = BytesIO()
    pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    buf.seek(0)
    filename = f"calculo_{calc.id}.pdf"
    resp = make_response(buf.read())
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return resp


# ── HISTORIAL ─────────────────────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    calcs = Calculation.query.order_by(Calculation.created_at.desc()).all()
    return render_template("history.html", calcs=calcs)


@app.route("/admin/tiempos", methods=["GET"])
@login_required
def admin_tiempos():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    times = load_catalog()
    return render_template("admin/tiempos.html", times=times)


@app.route("/admin/tiempos/editar", methods=["POST"])
@login_required
def admin_tiempos_editar():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    times = load_catalog()
    for key, value in request.form.items():
        # key format: "CATEGORIA|subcategoria|CODIGO"
        parts = key.split("|")
        if len(parts) == 3:
            cat, sub, cod = parts
            try:
                times[cat][sub][cod]["tiempo"] = float(value)
            except (KeyError, ValueError):
                pass
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(times, f, ensure_ascii=False, indent=2)
    flash("Tiempos actualizados correctamente.")
    return redirect(url_for("admin_tiempos"))


@app.route("/admin/tiempos/zona/agregar", methods=["POST"])
@login_required
def admin_tiempos_zona_agregar():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    nombre = request.form.get("nombre_zona", "").strip().upper()
    if not nombre:
        flash("El nombre de la zona es obligatorio.")
        return redirect(url_for("admin_tiempos"))
    times = load_catalog()
    if nombre in times:
        flash(f"La zona '{nombre}' ya existe.")
    else:
        times[nombre] = {}
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(times, f, ensure_ascii=False, indent=2)
        flash(f"Zona '{nombre}' agregada correctamente.")
    return redirect(url_for("admin_tiempos"))


@app.route("/admin/tiempos/proceso/agregar", methods=["POST"])
@login_required
def admin_tiempos_proceso_agregar():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    zona    = request.form.get("zona", "").strip()
    proceso = request.form.get("nombre_proceso", "").strip()
    codigo  = request.form.get("codigo", "").strip().upper()
    desc    = request.form.get("desc", "").strip()
    try:
        tiempo = float(request.form.get("tiempo", "0") or "0")
    except ValueError:
        tiempo = 0.0
    if not zona or not proceso or not codigo:
        flash("Zona, nombre de proceso y código son obligatorios.")
        return redirect(url_for("admin_tiempos"))
    times = load_catalog()
    if zona not in times:
        flash(f"La zona '{zona}' no existe.")
        return redirect(url_for("admin_tiempos"))
    if proceso in times[zona]:
        flash(f"El proceso '{proceso}' ya existe en '{zona}'.")
        return redirect(url_for("admin_tiempos"))
    times[zona][proceso] = {codigo: {"tiempo": tiempo, "desc": desc}}
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(times, f, ensure_ascii=False, indent=2)
    flash(f"Proceso '{proceso}' agregado a '{zona}'.")
    return redirect(url_for("admin_tiempos"))


@app.route("/admin/tiempos/zona/eliminar", methods=["POST"])
@login_required
def admin_tiempos_zona_eliminar():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    zona = request.form.get("zona", "").strip()
    if not zona:
        return redirect(url_for("admin_tiempos"))
    times = load_catalog()
    if zona in times:
        del times[zona]
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(times, f, ensure_ascii=False, indent=2)
        flash(f"Zona '{zona}' eliminada.")
    else:
        flash(f"La zona '{zona}' no existe.")
    return redirect(url_for("admin_tiempos"))


@app.route("/admin/tiempos/proceso/eliminar", methods=["POST"])
@login_required
def admin_tiempos_proceso_eliminar():
    if not current_user.is_admin:
        flash("Solo el administrador puede editar los tiempos estándar.")
        return redirect(url_for("index"))
    zona    = request.form.get("zona", "").strip()
    proceso = request.form.get("proceso", "").strip()
    if not zona or not proceso:
        return redirect(url_for("admin_tiempos"))
    times = load_catalog()
    if zona in times and proceso in times[zona]:
        del times[zona][proceso]
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(times, f, ensure_ascii=False, indent=2)
        flash(f"Proceso '{proceso}' eliminado.")
    else:
        flash("No se encontró la zona o proceso especificado.")
    return redirect(url_for("admin_tiempos"))


@app.route("/history/<int:calc_id>/delete", methods=["POST"])
@login_required
def delete_calc(calc_id):
    if not current_user.is_admin:
        flash("Solo el administrador puede eliminar cálculos.")
        return redirect(url_for("history"))
    calc = db.session.get(Calculation, calc_id)
    if calc:
        db.session.delete(calc)
        db.session.commit()
        flash("Cálculo eliminado.")
    return redirect(url_for("history"))


# ══ INIT DB ═══════════════════════════════════════════════════════════════════

with app.app_context():
    db.create_all()
    # Migración segura: agrega image_url si no existe
    with db.engine.connect() as _conn:
        cols = [r[1] for r in _conn.execute(db.text("PRAGMA table_info(lc_calculation)")).fetchall()]
        if "image_url" not in cols:
            _conn.execute(db.text("ALTER TABLE lc_calculation ADD COLUMN image_url VARCHAR(512)"))
            _conn.commit()
    if not User.query.first():
        admin = User(
            username="admin",
            password_hash=generate_password_hash("admin123"),
            is_admin=True,
        )
        db.session.add(admin)
        db.session.commit()
        print("[DB] Usuario admin creado — contraseña: admin123")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
