import json
import math
import os
import re
import secrets

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, flash, jsonify, make_response, redirect, render_template, request, send_from_directory, url_for
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user,
    login_required, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

import nesting
from extensions import db

UPLOAD_FOLDER = "uploads"
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "times.json")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
if not os.environ.get("SECRET_KEY"):
    print("[WARN] SECRET_KEY no configurada — se generó una al vuelo "
          "(las sesiones no sobreviven un reinicio). Defínela en .env para producción.")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///calculadora.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# ══ DATABASE ══════════════════════════════════════════════════════════════════

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Inicia sesión para continuar."

# Modelos, análisis de archivos de diseño, sugerencia de materiales y
# generación de PDF viven en sus propios módulos (ver auditoría de
# reestructuración) — se importan aquí, DESPUÉS de montar `db` sobre `app`.
# `models.py` importa `db` desde extensions.py (no desde aquí) justamente
# para que correr `python app.py` directamente no cargue este archivo dos
# veces bajo dos nombres de módulo distintos ("__main__" y "app") y rompa
# el import circular — ver extensions.py.
from models import User, Calculation  # noqa: E402
from image_analysis import (  # noqa: E402
    CV2_AVAILABLE, LIBREOFFICE_AVAILABLE, TESSERACT_AVAILABLE,
    analyze_cdr_file, analyze_image_file, analyze_pdf_file,
    _convert_cdr_to_pdf, _unique_upload_name,
)
from materials import get_material_suggestions  # noqa: E402
from pdf_export import (  # noqa: E402
    _build_nesting_pdf, _build_led_diagram_pdf,
    _nesting_pdf_path, _led_diagram_pdf_path, _pdf_page_count,
)


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
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in (".cdr", ".jpg", ".jpeg", ".png", ".pdf"):
        return jsonify({"error": "Solo se aceptan archivos .cdr, .jpg, .png o .pdf"}), 400
    if ext == ".cdr" and not LIBREOFFICE_AVAILABLE:
        return jsonify({"error": "LibreOffice no está instalado en el servidor"}), 500
    if ext in (".cdr", ".pdf") and not PYMUPDF_AVAILABLE:
        return jsonify({"error": "PyMuPDF no está instalado en el servidor"}), 500
    if ext in (".jpg", ".jpeg", ".png") and not TESSERACT_AVAILABLE:
        return jsonify({"error": "Tesseract OCR no está instalado en el servidor"}), 500

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    filename = _unique_upload_name(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    if ext == ".cdr":
        data = analyze_cdr_file(path)
    elif ext == ".pdf":
        data = analyze_pdf_file(path)
    else:
        data = analyze_image_file(path)
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
        "tesseract_available": TESSERACT_AVAILABLE,
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
        led_brand         = (request.form.get("led_brand") or "lumicolor").strip().lower()
        if led_brand not in ("dekorama", "lumicolor"):
            led_brand = "lumicolor"

        areas         = request.form.getlist("areas[]")
        subprocesses  = request.form.getlist("subprocesses[]")
        groups        = request.form.getlist("groups[]")
        quantities    = request.form.getlist("quantities[]")
        widths        = request.form.getlist("widths[]")
        heights       = request.form.getlist("heights[]")
        heights_detail = request.form.getlist("heights_detail[]")

        if not areas:
            flash("Agrega al menos una área productiva.")
            return redirect(request.url)

        rows: List[Dict] = []
        errors: List[str] = []
        # Medidas reales de letra de cada fila de CORTE cuyo material tenga
        # una hoja conocida (Acrílico blanco, Lámina cal. 19, Trovicel —
        # ver nesting.MATERIAL_SHEETS) y traiga alturas individuales del
        # .cdr. Puede haber más de una fila por material (p.ej. Grupo 1 y
        # Grupo 2 del mismo archivo si ambos se mandan a cortar) — todas
        # las del mismo material se juntan en su propio plano más abajo.
        nesting_letter_heights_by_material: Dict[str, List[List[float]]] = {}

        # Alturas reales de cada fila de ILUMINACIÓN (no Neón) aplicada con
        # letra por letra — para el diagrama de referencia de LED (ver
        # _build_led_diagram_pdf), una página por grupo de letras.
        led_letter_heights: List[List[float]] = []

        # Conversión del .cdr a PDF, perezosa y en caché: la necesitan tanto
        # el nesting (más abajo) como el perímetro real por letra de CANTO/
        # TRIMCAP (ver piece_perimeters_mm) — una sola conversión para
        # ambos usos, no una por fila.
        cdr_filename = request.form.get("cdr_filename", "").strip()
        _cdr_pdf_cache: Dict[str, Optional[str]] = {}

        def _get_cdr_pdf_path() -> Optional[str]:
            if "path" not in _cdr_pdf_cache:
                path = None
                if (cdr_filename and cdr_filename.lower().endswith(".cdr")
                        and LIBREOFFICE_AVAILABLE and PYMUPDF_AVAILABLE):
                    try:
                        cdr_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(cdr_filename))
                        if os.path.exists(cdr_path):
                            path = _convert_cdr_to_pdf(cdr_path, app.config["UPLOAD_FOLDER"])
                    except Exception:
                        path = None
                _cdr_pdf_cache["path"] = path
            return _cdr_pdf_cache["path"]

        # Pad lists to match row count
        widths         = widths         + [""] * max(0, len(areas) - len(widths))
        heights        = heights        + [""] * max(0, len(areas) - len(heights))
        heights_detail = heights_detail + [""] * max(0, len(areas) - len(heights_detail))

        for i, (area, subprocess, group, qty_str, width_str, height_str, heights_detail_str) in enumerate(
                zip(areas, subprocesses, groups, quantities, widths, heights, heights_detail)):
            if not area or not subprocess or not group:
                errors.append(f"Fila {i + 1}: selecciona área, subproceso y grupo.")
                continue
            try:
                qty = float(qty_str) if qty_str else 1.0
                if qty <= 0:
                    errors.append(f"Fila {i + 1}: la cantidad debe ser mayor a 0.")
                    continue
                # Medida real de cada letra (no el promedio/nominal del
                # grupo): la regresión es lineal, así que sumar el tiempo de
                # cada altura real da un total más ajustado que "un alto ×
                # cantidad" cuando el grupo mezcla piezas de tamaño distinto
                # (p.ej. el logo y las letras de un mismo set).
                letter_heights: Optional[List[float]] = None
                if group == "Regresión" and heights_detail_str:
                    try:
                        parsed = json.loads(heights_detail_str)
                        if isinstance(parsed, list) and parsed:
                            letter_heights = [float(h) for h in parsed]
                    except (ValueError, TypeError):
                        letter_heights = None

                # Ensamble sin flange con ancho conocido: fórmula calibrada
                if area == "ENSAMBLE" and subprocess == "Ensamble sin flange" and width_str:
                    ancho = float(width_str)
                    if ancho <= 0:
                        errors.append(f"Fila {i + 1}: el ancho debe ser mayor a 0.")
                        continue
                    mins = (0.2545 + 0.01067 * ancho) * 60  # hrs → min
                    group = f"Formula({ancho:.1f}cm)"
                    total_mins = mins * qty
                elif letter_heights:
                    reg = compute_regression(catalog, area, subprocess)
                    if reg:
                        slope, intercept = reg
                        total_mins = sum(max(0.0, slope * h + intercept) for h in letter_heights)
                        qty = float(len(letter_heights))
                        mins = total_mins / qty
                        group = f"Regresión letra por letra ({len(letter_heights)} piezas)"
                    else:
                        first = next(iter(catalog.get(area, {}).get(subprocess, {})), None)
                        if first:
                            mins = get_group_time(catalog, area, subprocess, first)
                            group = first
                            total_mins = mins * qty
                        else:
                            errors.append(f"Fila {i + 1}: sin datos de regresión para {area}.")
                            continue
                # Regresión lineal: medida fuera del rango conocido
                elif group == "Regresión" and height_str:
                    h = float(height_str)
                    if h <= 0:
                        errors.append(f"Fila {i + 1}: el alto debe ser mayor a 0.")
                        continue
                    reg = compute_regression(catalog, area, subprocess)
                    if reg:
                        slope, intercept = reg
                        mins = max(0.0, slope * h + intercept)
                        group = f"Regresión lineal ({h:.0f} cm)"
                        total_mins = mins * qty
                    else:
                        first = next(iter(catalog.get(area, {}).get(subprocess, {})), None)
                        if first:
                            mins = get_group_time(catalog, area, subprocess, first)
                            group = first
                            total_mins = mins * qty
                        else:
                            errors.append(f"Fila {i + 1}: sin datos de regresión para {area}.")
                            continue
                else:
                    mins = get_group_time(catalog, area, subprocess, group)
                    total_mins = mins * qty

                # Perímetro real de cada letra (contorno + huecos, ver
                # nesting.piece_perimeters_mm) para CANTO/TRIMCAP: no afecta
                # el tiempo (sigue siendo por alto, como siempre), es un
                # dato aparte para la lista de materiales — cuántos cm de
                # canto/Silvatrim se necesitan de verdad.
                perimeter_cm = None
                if area in ("CANTO", "TRIMCAP") and letter_heights:
                    pdf_path = _get_cdr_pdf_path()
                    if pdf_path:
                        try:
                            perims_mm = nesting.piece_perimeters_mm(pdf_path, letter_heights)
                        except Exception:
                            perims_mm = None
                        if perims_mm:
                            perimeter_cm = round(sum(perims_mm) / 10.0, 1)

                # Metros lineales de LED (contorno real, doblado si el
                # trazo de la letra es grueso — ver nesting.piece_led_meters)
                # para ILUMINACIÓN (no Neón, que se mide distinto): tampoco
                # afecta el tiempo, es el dato real para pedir material.
                led_meters = None
                led_count = None
                if area == "ILUMINACIÓN" and subprocess != "Neón" and letter_heights:
                    pdf_path = _get_cdr_pdf_path()
                    if pdf_path:
                        try:
                            led_results = nesting.piece_led_meters(pdf_path, letter_heights)
                        except Exception:
                            led_results = None
                        if led_results:
                            led_meters = round(sum(r["led_meters"] for r in led_results), 2)
                            led_count = math.ceil(sum(r["led_count"] for r in led_results))

                rows.append({
                    "area": area,
                    "subprocess": subprocess,
                    "group": group,
                    "quantity": qty,
                    "mins_per_piece": round(mins, 2),
                    "total_mins": round(total_mins, 2),
                    "perimeter_cm": perimeter_cm,
                    "led_meters": led_meters,
                    "led_count": led_count,
                })
                if area == "CORTE" and letter_heights:
                    material = nesting.material_for_subprocess(subprocess)
                    if material:
                        material_label = material[0]
                        nesting_letter_heights_by_material.setdefault(material_label, []).append(letter_heights)
                if area == "ILUMINACIÓN" and subprocess != "Neón" and letter_heights:
                    led_letter_heights.append(letter_heights)
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

        cdr_file_ready = bool(cdr_filename and cdr_filename.lower().endswith(".cdr")
                              and LIBREOFFICE_AVAILABLE and PYMUPDF_AVAILABLE)

        nesting_pdf_url = None
        if nesting_letter_heights_by_material and cdr_file_ready:
            try:
                nesting_pdf_url = _build_nesting_pdf(cdr_filename, nesting_letter_heights_by_material)
            except Exception:
                nesting_pdf_url = None  # el cálculo no debe fallar por esto

        led_diagram_url = None
        if led_letter_heights and cdr_file_ready:
            try:
                led_diagram_url = _build_led_diagram_pdf(cdr_filename, led_letter_heights)
            except Exception:
                led_diagram_url = None  # el cálculo no debe fallar por esto

        calc = Calculation(
            user_id=current_user.id,
            name=calc_name,
            rows_json=json.dumps(rows),
            params_json=json.dumps(params),
            results_json=json.dumps(results),
            image_url=request.form.get("image_url") or None,
            nesting_pdf_url=nesting_pdf_url,
            led_brand=led_brand,
            led_diagram_url=led_diagram_url,
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
    materials = get_material_suggestions(rows, params, calc.led_brand or "lumicolor")
    nesting_pdf_pages = _pdf_page_count(_nesting_pdf_path(calc))
    led_diagram_pdf_pages = _pdf_page_count(_led_diagram_pdf_path(calc))
    return render_template("result.html", calc=calc, rows=rows, materials=materials,
                           nesting_pdf_pages=nesting_pdf_pages,
                           led_diagram_pdf_pages=led_diagram_pdf_pages, **params, **results)


@app.route("/result/<int:calc_id>/nesting_preview/<int:page>.png")
@login_required
def nesting_preview_page(calc_id, page):
    """Rasteriza una página del PDF de nesting a PNG, para incrustarla como
    <img> dentro de result.html y que el 'Imprimir' del navegador la incluya
    como páginas extra del mismo PDF (mismo formato de siempre, nesting
    agregado al final)."""
    calc = db.session.get(Calculation, calc_id)
    if calc is None or not PYMUPDF_AVAILABLE:
        return "", 404
    nesting_path = _nesting_pdf_path(calc)
    if not nesting_path:
        return "", 404
    with fitz.open(nesting_path) as doc:
        if page < 0 or page >= doc.page_count:
            return "", 404
        pix = doc[page].get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
    resp = make_response(png_bytes)
    resp.headers["Content-Type"] = "image/png"
    return resp


@app.route("/result/<int:calc_id>/led_diagram_preview/<int:page>.png")
@login_required
def led_diagram_preview_page(calc_id, page):
    """Igual que nesting_preview_page pero para el diagrama de LED — para
    que 'Imprimir' desde result.html también lo incluya en el mismo PDF,
    junto con el plano de corte y el proceso/lista de materiales (pedido
    del cliente: todo en un solo PDF, no solo en la descarga por
    /result/<id>/download)."""
    calc = db.session.get(Calculation, calc_id)
    if calc is None or not PYMUPDF_AVAILABLE:
        return "", 404
    led_path = _led_diagram_pdf_path(calc)
    if not led_path:
        return "", 404
    with fitz.open(led_path) as doc:
        if page < 0 or page >= doc.page_count:
            return "", 404
        pix = doc[page].get_pixmap(dpi=150)
        png_bytes = pix.tobytes("png")
    resp = make_response(png_bytes)
    resp.headers["Content-Type"] = "image/png"
    return resp


@app.route("/result/<int:calc_id>/print")
@login_required
def result_print(calc_id):
    calc = db.session.get(Calculation, calc_id)
    if calc is None:
        return "Cálculo no encontrado", 404
    rows      = json.loads(calc.rows_json)
    params    = json.loads(calc.params_json)
    results   = json.loads(calc.results_json)
    materials = get_material_suggestions(rows, params, calc.led_brand or "lumicolor")
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
    materials = get_material_suggestions(rows, params, calc.led_brand or "lumicolor")
    html = render_template("print_result.html", calc=calc, rows=rows, materials=materials,
                           base_url=request.host_url, **params, **results)
    buf = BytesIO()
    pisa.CreatePDF(html, dest=buf, encoding="utf-8")
    buf.seek(0)
    pdf_bytes = buf.read()

    # Un solo PDF con todo: la cotización, el plano de corte (nesting) y el
    # diagrama de LED, en ese orden — antes solo se pegaba el nesting, el
    # diagrama de LED se quedaba como descarga aparte (pedido del cliente:
    # "quiero que todo salga en un solo pdf, también el diagrama led").
    if PYMUPDF_AVAILABLE:
        for extra_url in (calc.nesting_pdf_url, calc.led_diagram_url):
            if not extra_url:
                continue
            extra_filename = extra_url.rsplit("/", 1)[-1]
            extra_path = os.path.join(app.config["UPLOAD_FOLDER"], extra_filename)
            if not os.path.exists(extra_path):
                continue
            try:
                merged = fitz.open(stream=pdf_bytes, filetype="pdf")
                with fitz.open(extra_path) as extra_doc:
                    merged.insert_pdf(extra_doc)
                pdf_bytes = merged.tobytes()
                merged.close()
            except Exception:
                pass  # si falla la fusión de esta parte, se sigue con el resto

    filename = f"calculo_{calc.id}.pdf"
    resp = make_response(pdf_bytes)
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
        if "nesting_pdf_url" not in cols:
            _conn.execute(db.text("ALTER TABLE lc_calculation ADD COLUMN nesting_pdf_url VARCHAR(512)"))
            _conn.commit()
        if "led_brand" not in cols:
            _conn.execute(db.text("ALTER TABLE lc_calculation ADD COLUMN led_brand VARCHAR(20)"))
            _conn.commit()
        if "led_diagram_url" not in cols:
            _conn.execute(db.text("ALTER TABLE lc_calculation ADD COLUMN led_diagram_url VARCHAR(512)"))
            _conn.commit()
    if not User.query.first():
        admin_password = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(9)
        admin = User(
            username="admin",
            password_hash=generate_password_hash(admin_password),
            is_admin=True,
        )
        db.session.add(admin)
        db.session.commit()
        if os.environ.get("ADMIN_PASSWORD"):
            print("[DB] Usuario admin creado con la contraseña de ADMIN_PASSWORD.")
        else:
            print(f"[DB] Usuario admin creado — contraseña generada: {admin_password}  "
                  f"(guárdala: no se vuelve a mostrar; define ADMIN_PASSWORD en .env para fijarla tú)")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
