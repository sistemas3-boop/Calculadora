"""
Generación de los PDFs de plano de corte y diagrama de LED (orquestación
sobre nesting.py) y resolución de rutas/conteo de páginas de esos PDFs ya
guardados para un cálculo.

Se separó de app.py (ver auditoría de reestructuración). Usa
`flask.current_app` en vez de `from app import app` para `UPLOAD_FOLDER`:
estas funciones solo se llaman desde dentro de una ruta (ya hay contexto
de aplicación activo), e importar `app` directamente aquí causaría que
correr `python app.py` directamente cargue ese archivo dos veces bajo dos
nombres de módulo distintos ("__main__" y "app") y rompa el import
circular — ver extensions.py, que resuelve el mismo problema para `db`."""
import os
import time
from typing import Any, Dict, List, Optional

from flask import current_app, url_for
from werkzeug.utils import secure_filename

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

import nesting
from image_analysis import _convert_cdr_to_pdf


def _build_nesting_pdf(cdr_filename: str, letter_heights_by_material: Dict[str, List[List[float]]]) -> Optional[str]:
    """Genera el plano de corte (nesting real, 6 mm de separación) para
    cada material presente en `letter_heights_by_material` (clave = label
    del material, ver nesting.MATERIAL_SHEETS), cada uno sobre el tamaño de
    hoja que le corresponde — no se pueden mezclar letras de materiales
    distintos en la misma hoja. Junta todos los grupos de letras que
    coinciden con cada lista (p.ej. Grupo 1 y Grupo 2 del mismo archivo, si
    ambos se mandan a cortar del mismo material), a partir del .cdr ya
    guardado en uploads/ al analizarlo. Devuelve la URL pública del PDF
    combinado, o None si no se pudo generar nada.
    """
    cdr_path = os.path.join(current_app.config["UPLOAD_FOLDER"], secure_filename(cdr_filename))
    if not os.path.exists(cdr_path):
        return None
    pdf_path = _convert_cdr_to_pdf(cdr_path, current_app.config["UPLOAD_FOLDER"])
    base = os.path.splitext(os.path.basename(cdr_path))[0]
    out_name = f"nesting_{base}_{int(time.time())}.pdf"
    out_path = os.path.join(current_app.config["UPLOAD_FOLDER"], out_name)

    material_jobs = []
    for material_label, letter_heights_lists in letter_heights_by_material.items():
        sheet_w_mm, sheet_h_mm = next(
            (w, h) for _, label, w, h in nesting.MATERIAL_SHEETS if label == material_label
        )
        material_jobs.append({
            "material_label": material_label,
            "sheet_w_mm": sheet_w_mm,
            "sheet_h_mm": sheet_h_mm,
            "letter_heights_lists": letter_heights_lists,
        })

    result_path = nesting.build_nesting_pdf_multi_material(pdf_path, material_jobs, out_path)
    if not result_path:
        return None
    return url_for("uploaded_file", filename=out_name)


def _build_led_diagram_pdf(cdr_filename: str, led_letter_heights: List[List[float]]) -> Optional[str]:
    """Genera el diagrama de referencia de LED (ver
    nesting.build_led_diagram_pdf) para todas las filas de ILUMINACIÓN
    aplicadas con letra por letra — una página por grupo de letras. Devuelve
    la URL pública del PDF, o None si no se pudo generar nada."""
    cdr_path = os.path.join(current_app.config["UPLOAD_FOLDER"], secure_filename(cdr_filename))
    if not os.path.exists(cdr_path):
        return None
    pdf_path = _convert_cdr_to_pdf(cdr_path, current_app.config["UPLOAD_FOLDER"])
    base = os.path.splitext(os.path.basename(cdr_path))[0]
    out_name = f"led_{base}_{int(time.time())}.pdf"
    out_path = os.path.join(current_app.config["UPLOAD_FOLDER"], out_name)

    result_path = nesting.build_led_diagram_pdf(pdf_path, led_letter_heights, out_path)
    if not result_path:
        return None
    return url_for("uploaded_file", filename=out_name)


def _nesting_pdf_path(calc) -> Optional[str]:
    """Resuelve la ruta en disco del PDF de nesting de un cálculo, a partir
    de su nesting_pdf_url (p.ej. "/uploads/nesting_x.pdf"). None si no tiene
    nesting o el archivo ya no existe."""
    if not calc.nesting_pdf_url:
        return None
    filename = calc.nesting_pdf_url.rsplit("/", 1)[-1]
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    return path if os.path.exists(path) else None


def _led_diagram_pdf_path(calc) -> Optional[str]:
    """Igual que _nesting_pdf_path pero para el diagrama de LED
    (calc.led_diagram_url)."""
    if not calc.led_diagram_url:
        return None
    filename = calc.led_diagram_url.rsplit("/", 1)[-1]
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    return path if os.path.exists(path) else None


def _pdf_page_count(path: Optional[str]) -> int:
    if not path or not PYMUPDF_AVAILABLE:
        return 0
    try:
        with fitz.open(path) as _doc:
            return _doc.page_count
    except Exception:
        return 0
