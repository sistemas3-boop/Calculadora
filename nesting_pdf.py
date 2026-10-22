"""
Plano de corte y diagrama de LED en PDF, dibujados con matplotlib.

matplotlib exporta PDF vectorial de verdad (no rasteriza el dibujo) y da
texto/anotaciones mucho más cómodos que dibujar a mano con PyMuPDF —
número y medida de cada pieza, cuadrícula de referencia en cm, resumen de
aprovechamiento por hoja. PyMuPDF se sigue usando para TODO lo demás
(leer el .cdr/PDF, extraer los contornos — ver geometry.py): esto solo
dibuja el plano/diagrama final.

Antes vivía junto con la extracción de geometría, el conteo de LED y el
nesting en un solo nesting.py; se separó en geometry.py / led_layout.py /
sheet_packing.py / nesting_pdf.py (ver auditoría de reestructuración) —
nesting.py quedó como fachada de compatibilidad que re-exporta todo esto.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # sin esto intenta abrir una ventana — no hay una en el servidor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath

from geometry import (
    _polygon_centroid,
    _polygon_nesting_depths,
    _polygon_signed_area,
    extract_group_polygons_mm,
)
from led_layout import (
    LED_MODULE_LEN_MM,
    LED_MODULE_PITCH_MM,
    LEDS_PER_MODULE,
    _piece_led_segments,
    _piece_led_stats,
    _sample_led_points,
)
from sheet_packing import GAP_MM_DEFAULT, SHEET_H_MM_DEFAULT, SHEET_W_MM_DEFAULT, _nest_result_for_material

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - se valida antes de llamar a las funciones que lo usan
    _ShapelyPolygon = None

MM_TO_IN = 1.0 / 25.4

# Paleta sobria (no arcoíris): grises/azules apagados que se distinguen
# entre piezas vecinas sin verse como una demo — pensada para imprimirse
# también en blanco y negro sin perder legibilidad (contraste por trazo,
# no solo por color).
PIECE_COLORS = [
    "#cfe0f3", "#d9e6d5", "#eadcc9", "#e3d3e0", "#d6e6e6", "#f0ddd2",
]
PIECE_EDGE = "#2c3e50"


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
    (eje medio, o una vuelta pegada al borde en trazos gruesos) con un
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
    ROW_GAP_MM = 60.0
    # Proporción ancho:alto que se busca para la página — con muchas
    # letras en una sola fila el diagrama salía EXTREMO (una ficha real de
    # 9 letras grandes dio una página de 17 × 3.2 pulgadas, ~5.3:1): a esa
    # proporción un visor de PDF a tamaño real necesita scroll horizontal
    # para ver las últimas letras — "se pierde la info" aunque el PDF en
    # sí no recorte nada. Envolver en varias filas (como un párrafo de
    # texto) cuando hace falta deja una página con una proporción normal
    # de imprimir/ver completa de un vistazo.
    TARGET_ASPECT = 1.5

    widths_mm, heights_mm = [], []
    for polys in polys_list:
        xs = [x for poly in polys for x, y in poly]
        ys = [y for poly in polys for x, y in poly]
        widths_mm.append(max(xs) - min(xs) if xs else 0.0)
        heights_mm.append(max(ys) - min(ys) if ys else 0.0)
    max_h = max(heights_mm) if heights_mm else 0.0
    single_row_w = sum(widths_mm) + PIECE_GAP_MM * max(0, len(polys_list) - 1)

    # Cuántas filas conviene usar: la proporción resultante de una sola
    # fila (single_row_w : max_h) contra TARGET_ASPECT — ver el comentario
    # de arriba para la cuenta. Nunca más filas que piezas.
    n_rows = 1
    if polys_list and max_h > 0:
        est = math.sqrt(single_row_w / (TARGET_ASPECT * (max_h + ROW_GAP_MM)))
        n_rows = max(1, min(len(polys_list), round(est)))

    # Reparte las piezas en filas tipo párrafo: se van acumulando en la
    # fila actual hasta que agregar la siguiente se pasaría del ancho
    # objetivo por fila (total entre filas) — igual que envolver texto.
    target_row_w = single_row_w / n_rows if n_rows > 1 else float("inf")
    rows: List[List[int]] = [[]]
    row_w_acc = 0.0
    for i, w in enumerate(widths_mm):
        piece_w = w + (PIECE_GAP_MM if rows[-1] else 0.0)
        if rows[-1] and row_w_acc + piece_w > target_row_w and len(rows) < n_rows:
            rows.append([])
            row_w_acc = 0.0
            piece_w = w
        rows[-1].append(i)
        row_w_acc += piece_w

    row_widths = [sum(widths_mm[i] for i in row) + PIECE_GAP_MM * max(0, len(row) - 1) for row in rows]
    row_heights = [max((heights_mm[i] for i in row), default=0.0) for row in rows]
    total_w = max(row_widths) if row_widths else 0.0
    total_h = sum(row_heights) + ROW_GAP_MM * max(0, len(rows) - 1)

    MAX_PAGE_IN = 17.0
    full_w_in = (total_w + margin_mm * 2) * MM_TO_IN
    full_h_in = (total_h + margin_mm * 2 + header_mm) * MM_TO_IN
    shrink = min(1.0, MAX_PAGE_IN / max(full_w_in, full_h_in, 1e-6))
    fig_h_in = full_h_in * shrink
    fig, ax = plt.subplots(figsize=(full_w_in * shrink, fig_h_in))

    total_led_m = 0.0
    total_led_count = 0
    y_row_base = 0.0
    for row, row_h in zip(rows, row_heights):
        x_cursor = 0.0
        for i in row:
            polys = polys_list[i]
            w, h = widths_mm[i], heights_mm[i]
            y_off = y_row_base + (row_h - h)  # alinea por la base de SU fila, como se lee la palabra
            shifted = [[(x + x_cursor, y + y_off) for x, y in poly] for poly in polys]

            path = _piece_to_path(shifted)
            color = PIECE_COLORS[i % len(PIECE_COLORS)]
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor=PIECE_EDGE, linewidth=1.0, zorder=2))

            depths = _polygon_nesting_depths(polys)
            stats = _piece_led_stats(polys, depths=depths, module_pitch_mm=module_pitch_mm)
            led_m = stats["led_length_mm"] / 1000.0
            num_modules = stats["num_modules"]
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
                # `num_modules` ya salió calibrado por eje medio (ver
                # _piece_led_stats) pero el TRAZO dibujado es el inset — se
                # reparte esa misma cantidad de puntos parejo por longitud
                # de arco sobre el trazo dibujado (no por brazo del
                # esqueleto: el inset es un solo contorno, no un árbol de
                # brazos).
                pts = _sample_led_points(shifted_segments, num_modules)
                if pts:
                    ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                               s=22, color="#c0392b", zorder=4, linewidths=0.6, edgecolors="#7a2016")

            cx = x_cursor + w / 2
            # Con muchas piezas angostas en fila (una letra chica como
            # "I"/"l" con solo ~45mm de espacio propio) el texto a tamaño
            # fijo 6.5pt ya no cabe en su propio carril y se encima con el
            # de al lado — se vio real en un archivo de 16 letras chicas,
            # ilegible. Se calcula el tamaño (y, si ni reduciendo alcanza,
            # una versión más corta del texto) para que quepa en el ancho
            # real que le toca a ESTA pieza (su ancho + el hueco hasta la
            # siguiente, ya escalado por el shrink-to-fit de la página) —
            # ancho de carácter 0.58× el tamaño de letra, aproximación
            # estándar para texto mixto en una tipografía sans normal.
            slot_in = max((w + PIECE_GAP_MM) * MM_TO_IN * shrink, 0.05)
            CHAR_W_FACTOR = 0.58
            full_text = f"{led_m:.2f} m · {num_modules} módulos ({led_count} LEDs)"
            fs_full = slot_in * 72.0 / (len(full_text) * CHAR_W_FACTOR)
            if fs_full >= 5.0:
                label_text, label_fontsize = full_text, min(6.5, fs_full)
            else:
                short_text = f"{led_m:.2f}m · {num_modules} mód"
                fs_short = slot_in * 72.0 / (len(short_text) * CHAR_W_FACTOR)
                label_text, label_fontsize = short_text, max(3.5, min(6.5, fs_short))
            ax.text(cx, y_row_base + row_h + 14, label_text,
                    ha="center", va="top", fontsize=label_fontsize, color="#444444", zorder=5)

            x_cursor += w + PIECE_GAP_MM
        y_row_base += row_h + ROW_GAP_MM

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
              f"Diagrama de referencia — LED en dos vueltas pegadas hacia adentro de cada borde, "
              f"módulos de {LED_MODULE_LEN_MM / 10:.1f} cm, paso según el alto de cada letra "
              f"({leds_per_module} LED{'s' if leds_per_module != 1 else ''} por módulo)",
              fontsize=7.5, style="italic", color="#666666", ha="left", va="top")

    ax.set_xlim(-margin_mm, total_w + margin_mm)
    ax.set_ylim(total_h + margin_mm, -margin_mm)
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
