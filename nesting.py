"""
Fachada de compatibilidad.

Este módulo vivía solo, con la extracción de geometría, el conteo de LED,
el nesting sobre la hoja y el dibujo del plano en PDF todo junto (~2250
líneas). Se separó en cuatro módulos más chicos y enfocados (ver
auditoría de reestructuración):

  - geometry.py:     extracción del contorno real de cada letra desde el
                      PDF convertido, y su esqueleto (eje medio).
  - led_layout.py:    conteo de módulos de LED por pieza y el trazo de
                      ruteo (las dos vueltas pegadas al borde).
  - sheet_packing.py: nesting (relleno esquina-a-esquina) sobre la hoja
                      de corte, y el tamaño de hoja por material.
  - nesting_pdf.py:   dibujo del plano de corte y del diagrama de LED en
                      PDF (matplotlib).

nesting.py queda como fachada: re-exporta los mismos nombres (públicos y
privados) que tenía antes, para no romper los imports existentes de
app.py ni de los tests de caracterización. Código nuevo debería importar
directamente del módulo correspondiente en vez de pasar por aquí.
"""
from geometry import (
    MATCH_TOL_PT,
    LED_WIDTH_RES_MM,
    dedupe_drawings,
    _flatten_drawing_items,
    _group_compound_subpolys,
    split_compound_drawing_into_letters,
    _split_compound_drawing_polys,
    _match_group_for_nesting,
    extract_group_polygons_mm,
    _polygon_perimeter_mm,
    piece_perimeters_mm,
    _rasterize_piece_solid,
    _zhang_suen_thin,
    _skeleton_dfs_segments,
    _skeleton_mst_adjacency,
    _prune_skeleton_spurs,
    _piece_skeleton_mm,
    _polygon_signed_area,
    _polygon_centroid,
    _point_in_polygon,
    _polygon_nesting_depths,
    _piece_total_area,
)
from led_layout import (
    LED_THICK_STROKE_MM,
    LED_THICK_STROKE_FACTOR,
    LED_INSET_MM,
    LED_INSET_GAP_MM,
    LED_MODULE_LEN_MM,
    LED_MODULE_GAP_MM,
    LED_MODULE_PITCH_MM,
    LEDS_PER_MODULE,
    LED_PITCH_CATEGORIES_CM,
    LED_DOUBLE_ROW_MIN_HEIGHT_CM,
    _module_pitch_for_height_cm,
    _led_row_count_for_height_cm,
    _piece_led_stats,
    _led_module_count,
    _arm_module_count,
    piece_led_meters,
    _piece_led_segments_inset,
    _piece_led_segments,
    _sample_led_points,
)
from sheet_packing import (
    SHEET_W_MM_DEFAULT,
    SHEET_H_MM_DEFAULT,
    GAP_MM_DEFAULT,
    GRID_RES_MM,
    MATERIAL_SHEETS,
    material_for_subprocess,
    _rasterize_polygons,
    _mm_to_radius_cells,
    _disk_kernel,
    _dilate_mask,
    _find_first_fit,
    ROTATION_STEP_DEG_DEFAULT,
    _rotate_polygons,
    _compact_sheets,
    nest_pieces,
    _nest_result_for_material,
)
from nesting_pdf import (
    MM_TO_IN,
    PIECE_COLORS,
    PIECE_EDGE,
    _piece_to_path,
    _draw_sheet_page,
    _draw_led_diagram_page,
    build_led_diagram_pdf,
    render_nesting_pdf,
    build_nesting_pdf_for_rows,
    build_nesting_pdf_multi_material,
)

__all__ = [
    # geometry
    "MATCH_TOL_PT", "LED_WIDTH_RES_MM", "dedupe_drawings", "_flatten_drawing_items",
    "_group_compound_subpolys", "split_compound_drawing_into_letters",
    "_split_compound_drawing_polys", "_match_group_for_nesting", "extract_group_polygons_mm",
    "_polygon_perimeter_mm", "piece_perimeters_mm", "_rasterize_piece_solid", "_zhang_suen_thin",
    "_skeleton_dfs_segments", "_skeleton_mst_adjacency", "_prune_skeleton_spurs", "_piece_skeleton_mm",
    "_polygon_signed_area", "_polygon_centroid", "_point_in_polygon", "_polygon_nesting_depths",
    "_piece_total_area",
    # led_layout
    "LED_THICK_STROKE_MM", "LED_THICK_STROKE_FACTOR", "LED_INSET_MM", "LED_INSET_GAP_MM",
    "LED_MODULE_LEN_MM", "LED_MODULE_GAP_MM", "LED_MODULE_PITCH_MM", "LEDS_PER_MODULE",
    "LED_PITCH_CATEGORIES_CM", "LED_DOUBLE_ROW_MIN_HEIGHT_CM", "_module_pitch_for_height_cm",
    "_led_row_count_for_height_cm", "_piece_led_stats", "_led_module_count", "_arm_module_count",
    "piece_led_meters", "_piece_led_segments_inset", "_piece_led_segments", "_sample_led_points",
    # sheet_packing
    "SHEET_W_MM_DEFAULT", "SHEET_H_MM_DEFAULT", "GAP_MM_DEFAULT", "GRID_RES_MM", "MATERIAL_SHEETS",
    "material_for_subprocess", "_rasterize_polygons", "_mm_to_radius_cells", "_disk_kernel",
    "_dilate_mask", "_find_first_fit", "ROTATION_STEP_DEG_DEFAULT", "_rotate_polygons",
    "_compact_sheets", "nest_pieces", "_nest_result_for_material",
    # nesting_pdf
    "MM_TO_IN", "PIECE_COLORS", "PIECE_EDGE", "_piece_to_path", "_draw_sheet_page",
    "_draw_led_diagram_page", "build_led_diagram_pdf", "render_nesting_pdf",
    "build_nesting_pdf_for_rows", "build_nesting_pdf_multi_material",
]
