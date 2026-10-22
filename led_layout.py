"""
Conteo de módulos de LED y trazo de ruteo por pieza (letra).

Separa dos cosas a propósito: cuántos módulos lleva cada pieza (el
NÚMERO, calibrado contra fichas reales por eje medio/skeleton — ver
_piece_led_stats) y por dónde se dibuja la línea de LED en el plano (el
TRAZO, siempre dos vueltas pegadas hacia adentro de cada borde, ver
_piece_led_segments_inset) — confirmado con el cliente que son dos
geometrías distintas, no una derivada de la otra.

Antes vivía junto con la extracción de geometría y el nesting en un solo
nesting.py; se separó en geometry.py / led_layout.py / sheet_packing.py /
nesting_pdf.py (ver auditoría de reestructuración) — nesting.py quedó como
fachada de compatibilidad que re-exporta todo esto.
"""
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - se valida antes de llamar a las funciones que lo usan
    _ShapelyPolygon = None

from geometry import (
    _piece_skeleton_mm,
    _piece_total_area,
    _polygon_nesting_depths,
    _polygon_perimeter_mm,
    extract_group_polygons_mm,
)

# OJO: esto ya NO decide cómo se ve el plano — el plano SIEMPRE dibuja DOS
# vueltas pegadas hacia adentro de cada borde (ver "draw_segments" en
# _piece_led_stats), confirmado contra dos referencias reales del cliente
# (una muestra una sola línea bordeando el contorno completo de una "M" en
# vez de una línea por el centro; la siguiente corrigió que en realidad
# son DOS columnas de LEDs recorriendo todo ese mismo contorno, no una).
# Este umbral solo afecta cómo se CUENTAN los módulos (el número que sale
# en la etiqueta): por eje medio, letra por letra, contra fichas reales
# (Mifelgrande: 144 calc vs 146 real, incluso en trazos de 200-265mm de
# ancho) salió más preciso que contar sobre el contorno del inset (se
# probó: incluso con una sola vuelta el conteo salía muy por arriba de lo
# real — con dos vueltas sería peor todavía). DESACTIVADO por ahora
# (umbral prácticamente infinito) — baja este número si aparece un caso
# real donde el eje medio deje de contar bien (una letra tan ancha que el
# centro ya no represente el trazo real).
LED_THICK_STROKE_MM = 1e9
LED_THICK_STROKE_FACTOR = 1.0  # ya no se usa para armar el trazo dibujado (ver _piece_led_segments_inset, ahora siempre dos vueltas) — solo queda para el modo "inset" de CONTEO, dormido
LED_INSET_MM = 70.0  # separación entre el canto (borde real de la letra) y el módulo, en doble línea — confirmado real con el cliente (misma medida que el largo del módulo, 7cm)
# Separación entre las dos vueltas del inset doble (ver
# _piece_led_segments_inset): la segunda vuelta va a inset_mm +
# LED_INSET_GAP_MM del borde, NO a "ancho - inset_mm" — esa fórmula
# ubicaba la segunda vuelta pegada al borde OPUESTO (a más de 180mm de
# distancia en una M de 200mm de ancho), que en un trazo angosto se sale
# por completo del trazo y no deja nada que dibujar; la referencia real
# muestra las dos columnas CERCA entre sí, pegadas al mismo borde, no una
# en cada orilla.
LED_INSET_GAP_MM = 20.0

# Parámetros de módulo real (no LEDs individuales sueltos sobre la tira):
# cada módulo mide 7cm (confirmado con foto del módulo real y con la ficha
# técnica del fabricante, "V5 Led - Súper Brillante" — antes se asumían
# 6.5cm). Separación real entre el final de un módulo y el principio del
# siguiente: 10cm (misma foto), paso centro a centro = 7+10 = 17cm — NO
# "7 módulos por metro" (eso daría 14.3cm de paso, un dato aproximado que
# no coincide con la medida real del módulo ni de la separación). Se probó
# un paso variable por qué tan RECTO es cada tramo (contra una ficha real
# de producción, Mifel) pero no se sostuvo: el adorno/swash de esa ficha
# usa el paso más apretado de todos (130mm) a pesar de ser el tramo
# geométricamente más "suave" (un arco abierto de radio grande) por
# cualquier medida de rectitud (punta a punta o por ventana local) — y el
# ancho de trazo tampoco lo explica (la "l" y la "f" tienen casi el mismo
# ancho pero pasos reales muy distintos, 252mm vs ~190mm). Con solo esa
# ficha no hay suficiente señal geométrica para derivar una fórmula
# confiable; queda pendiente de más datos reales.
LED_MODULE_LEN_MM = 70.0
LED_MODULE_GAP_MM = 100.0  # separación real entre módulos (foto + ficha técnica del fabricante)
LED_MODULE_PITCH_MM = LED_MODULE_LEN_MM + LED_MODULE_GAP_MM  # paso centro a centro real: 17cm
LEDS_PER_MODULE = 1

# Categorías de paso por ALTO real de la letra (cm) — dadas por el cliente
# a partir de su propia experiencia de producción, no derivadas de
# geometría (ver el comentario de arriba: el intento de derivarlo de la
# geometría del trazo no se sostuvo). Cada tupla es (alto_min_cm,
# alto_max_cm, paso_mm), límite superior EXCLUSIVO. "Mediana chica" no
# tiene piso (cubre también toda letra chica, sin categoría propia
# todavía). Donde el cliente no dio un conteo directo de LEDs/metro
# (Mediana-Grande) se usa el paso que sí dio (módulo + separación de esa
# categoría — módulo actualizado a 7cm, ver LED_MODULE_LEN_MM). "Grande"
# sí se validó contra una ficha real (zacatecas, 70.2cm: 96 calc vs 95
# real) y se dejó igual (paso dado directo como 7 módulos/metro, no por
# módulo+separación, así que no lo toca la corrección del módulo).
# "Mediana chica" el dato original del cliente (16 módulos/metro, 62.5mm)
# sobreestimaba TODAS las letras de una ficha real a 20.7cm (63 calc vs 45
# real) — se recalibró a 80mm (~12.5 módulos/metro) probando varios pasos
# contra esa misma ficha hasta encontrar el que mejor ajusta (44 vs 45, 8
# de 9 letras exactas). "Extra grande" (≥90cm) se había calibrado
# probando pasos sueltos contra una ficha real de Mifelgrande (99.8-127.7
# cm, 9 piezas: swash×4, M, i, f, l, e) hasta encontrar el que mejor
# ajustaba el TOTAL (150mm, 144 vs 146 real) — pero el cliente confirmó
# con foto + ficha técnica del fabricante que el paso real de este módulo
# (el mismo que usa en letras grandes) es 17cm (7cm de módulo + 10cm de
# separación, ver LED_MODULE_PITCH_MM), así que se usa ese valor real en
# vez del ajuste por curva. OJO: el swash decorativo (sin ramificación, un
# solo arco) necesita un paso bastante más apretado (~105mm) que una letra
# normal en esa misma ficha — un solo paso no les queda igual de bien a
# los dos; no hay (todavía) una señal geométrica confiable para distinguir
# "pieza decorativa" de "letra" (se probó por curvatura en el pasado y no
# se sostuvo, ver el comentario de LED_MODULE_PITCH_MM), así que por ahora
# se usa este mismo paso real para toda la categoría.
LED_PITCH_CATEGORIES_CM: List[Tuple[float, float, float]] = [
    (0.0, 25.0, 80.0),                             # Mediana chica (y menor): ~12.5 módulos/metro (recalibrado, ver arriba)
    (25.0, 40.0, 1000.0 / 8.0),                    # Mediana: 8 módulos/metro
    (40.0, 60.0, LED_MODULE_LEN_MM + 90.0),        # Mediana-grande: módulo 7cm + 9cm de separación
    (60.0, 90.0, 1000.0 / 7.0),                    # Grande: 7 módulos/metro
    (90.0, 1e9, LED_MODULE_PITCH_MM),              # Extra grande: paso real del módulo (7cm + 10cm), confirmado con el cliente
]


def _module_pitch_for_height_cm(height_cm: float, default_pitch_mm: float = LED_MODULE_PITCH_MM) -> float:
    """Paso centro a centro según el ALTO real de la pieza (ver
    LED_PITCH_CATEGORIES_CM) — fuera de las categorías dadas usa
    `default_pitch_mm` (el paso plano de siempre)."""
    for lo_cm, hi_cm, pitch_mm in LED_PITCH_CATEGORIES_CM:
        if lo_cm <= height_cm < hi_cm:
            return pitch_mm
    return default_pitch_mm


# A partir de qué ALTO las dos hileras del trazo (ver
# _piece_led_segments_inset) dejan de ser un mismo reparto de módulos y
# pasan a ser DOS tiras independientes, cada una con su propio conteo
# completo al mismo paso — confirmado real: la M de Mifelgrande (127.7cm,
# trazo de ~200mm de ancho) lleva ~70 módulos reales, el doble de los ~35
# que da el eje medio solo UNA vez (no repartido entre las dos hileras).
# Mismo corte que "Extra grande" en LED_PITCH_CATEGORIES_CM porque es la
# única categoría donde se confirmó esto contra una ficha real; el trazo
# de "Grande" (zacatecas 70.2cm, ~130-160mm de ancho) sigue validado con
# una sola tira (96 calc vs 95 real, sin duplicar) — ahí no se duplica.
LED_DOUBLE_ROW_MIN_HEIGHT_CM = 90.0


def _led_row_count_for_height_cm(height_cm: float) -> int:
    """1 o 2 hileras independientes de módulos según el ALTO real de la
    pieza — ver LED_DOUBLE_ROW_MIN_HEIGHT_CM."""
    return 2 if height_cm >= LED_DOUBLE_ROW_MIN_HEIGHT_CM else 1


def _piece_led_stats(polys: List[List[Tuple[float, float]]],
                      depths: Optional[List[int]] = None,
                      module_pitch_mm: float = LED_MODULE_PITCH_MM) -> Dict[str, Any]:
    """Perímetro, grosor estimado, factor (1 o 2) y la longitud/trazo real
    de LED de una pieza — ver _piece_skeleton_mm para la explicación de
    cada fórmula (grosor incluido). Función compartida entre
    piece_led_meters (solo números) y _draw_led_diagram_page (el dibujo),
    para no calcular nada dos veces con fórmulas que se puedan ir
    desalineando.

    El CONTEO de módulos (cuántos) y el TRAZO dibujado (dónde va la línea
    en el plano, "draw_segments") usan dos geometrías distintas a
    propósito: el conteo sale por eje medio, letra por letra (ver abajo),
    porque contra fichas reales fue lo que mejor ajustó el NÚMERO; el
    trazo dibujado siempre va en DOS vueltas pegadas hacia adentro de cada
    borde (el inset de shapely, ver _piece_led_segments_inset) porque así
    es como el cliente rutea en la realidad (confirmado con dos
    referencias reales: la primera mostró una sola línea bordeando el
    contorno completo en vez de una línea por el centro; la segunda
    corrigió que en realidad son dos columnas recorriendo ese mismo
    contorno, no una sola).

    La cantidad de MÓDULOS (`num_modules`) se calcula acá mismo, no
    aparte: en modo sencillo, por BRAZO (cada tramo entre ramificaciones,
    ver _piece_skeleton_mm — una ramificación real es un corte físico
    entre tiras, el sobrante de un paso no se reparte de un brazo a otro),
    cada uno SIN mínimo propio (ver _arm_module_count — un brazo de puro
    ruido de borde en una letra chica no debe forzar su propio módulo), y
    se suman con un mínimo de 1 POR ISLA FÍSICA (ver "island" en cada
    brazo, _piece_skeleton_mm) si hace falta — no un solo mínimo para toda
    la pieza, porque dos islas sueltas entre sí (el puntito de una "í" y
    su palo) son dos focos de luz distintos: un solo mínimo por pieza deja
    sin módulo al más chico en cuanto el otro ya suma algo por su cuenta
    (confirmado con un puntito sintético de 6mm, que se perdía por
    completo). Sin módulo extra en las intersecciones (se probó sumar 1
    aparte en cada unión real, pero se quitó por pedido del usuario — cada
    brazo ya reparte los suyos por su cuenta). En modo inset, de un tirón
    con el mínimo normal (es un solo contorno cerrado, no un árbol de
    brazos).

    El paso (`module_pitch_mm`) se ajusta según el ALTO real de la PIEZA
    completa (no de cada brazo — ver LED_PITCH_CATEGORIES_CM): categorías
    dadas por el cliente a partir de su propia experiencia de producción
    (se probó derivarlo de la geometría del trazo — qué tan recto es cada
    brazo — pero no se sostuvo contra una ficha real, ver el comentario en
    LED_MODULE_PITCH_MM). El `module_pitch_mm` que recibe esta función
    solo se usa como respaldo fuera de esas categorías (letra muy chica o
    muy grande, sin categoría real todavía)."""
    if depths is None:
        depths = _polygon_nesting_depths(polys)
    perimeter_mm = sum(_polygon_perimeter_mm(poly) for poly in polys)
    area_mm2 = _piece_total_area(polys)
    all_ys = [p[1] for poly in polys for p in poly]
    piece_height_cm = (max(all_ys) - min(all_ys)) / 10.0 if all_ys else 0.0
    module_pitch_mm = _module_pitch_for_height_cm(piece_height_cm, module_pitch_mm)
    # El esqueleto se calcula siempre, aunque la pieza termine en modo
    # doble: es la única fuente de `width_mm` confiable (ver
    # _piece_skeleton_mm) y _piece_led_segments_inset también lo necesita
    # para el inset del modo doble, así que no hay forma de decidir el
    # modo SIN calcularlo primero.
    skeleton_segments, led_length_mm, arms, width_mm = _piece_skeleton_mm(polys, depths)
    is_thick = width_mm >= LED_THICK_STROKE_MM
    factor = LED_THICK_STROKE_FACTOR if is_thick else 1.0
    if is_thick:
        mode = "inset"
        led_length_mm = perimeter_mm * factor
        num_modules = _led_module_count(led_length_mm, module_pitch_mm)
        arms = []
    else:
        mode = "skeleton"
        for arm in arms:
            arm["num_modules"] = _arm_module_count(arm["length_mm"], module_pitch_mm)
        # Mínimo de 1 módulo POR ISLA física, no uno solo para toda la
        # pieza (ver docstring arriba): si todos los brazos de una misma
        # isla redondearon a 0 (letra muy chica, o un puntito aislado
        # representado por su brazo de respaldo casi-cero — ver
        # _piece_skeleton_mm), esa isla en conjunto igual necesita al
        # menos un módulo para encenderse. Se agrega al brazo más largo DE
        # ESA ISLA (el más representativo de su propio trazo, no ruido).
        by_island: Dict[Any, List[Dict[str, Any]]] = {}
        for arm in arms:
            by_island.setdefault(arm.get("island"), []).append(arm)
        for island_arms in by_island.values():
            if sum(a["num_modules"] for a in island_arms) == 0:
                longest = max(island_arms, key=lambda a: a["length_mm"])
                longest["num_modules"] = 1
        num_modules = sum(arm["num_modules"] for arm in arms)
    # A partir de cierto ALTO (ver LED_DOUBLE_ROW_MIN_HEIGHT_CM) el trazo
    # dibujado son DOS hileras independientes, no una sola repartida entre
    # dos líneas — cada hilera necesita su propio conteo completo al mismo
    # paso, confirmado real (la M de Mifelgrande: ~70 módulos reales, el
    # doble de los ~35 que da una sola pasada del eje medio). Se duplica
    # tanto el conteo como los metros reportados, porque son dos tiras
    # físicas de LED instaladas, no una.
    #
    # Se probó (a pedido del cliente) que el conteo saliera del largo REAL
    # de la línea dibujada (el inset) en vez del eje medio, para que mover
    # el inset del canto si cambiara el conteo — se revirtió: el offset de
    # shapely se rompe cuando el inset (90mm con el canto a 70mm + 20mm de
    # separación entre hileras) se acerca al tamaño de la propia letra —
    # confirmado real con una "A" de 20.6cm (170mm de ancho): el ápice de
    # la A se dispara HACIA AFUERA en vez de encogerse, y el largo salía
    # 10× inflado (57 módulos calculados en una letra que necesita ~5).
    # Letras chicas/medianas (la mayoría) caen justo en ese rango roto, así
    # que no es seguro usarlo como base de conteo.
    row_count = _led_row_count_for_height_cm(piece_height_cm)
    num_modules *= row_count
    led_length_mm *= row_count
    # El TRAZO que se dibuja en el diagrama es independiente de cómo se
    # cuentan los módulos (arriba): la cuenta por eje medio ya está
    # validada contra fichas reales (ver LED_PITCH_CATEGORIES_CM), pero la
    # LÍNEA que se dibuja en el plano siempre va en DOS vueltas pegadas
    # hacia adentro de cada borde (ver _piece_led_segments_inset) — así es
    # como el cliente dibuja el ruteo real (confirmado con dos referencias
    # reales: contorno completo en vez de por el centro, y doble columna
    # en vez de una sola). Respaldo al eje medio solo si shapely no está
    # disponible o el inset sale vacío (pieza degenerada).
    #
    # El inset (70mm) y la separación entre vueltas (20mm) son la medida
    # REAL del canto a 7cm, pero esa medida solo tiene sentido si la letra
    # es más ancha que eso: en un trazo de ~50mm de ancho (letra chica,
    # confirmado real con Zacatecas a 20.6cm) un inset de 90mm hace que el
    # offset de shapely se dispare hacia AFUERA de la letra en vez de
    # encogerse (mismo bug que se vio al intentar contar por esta misma
    # línea, ver el comentario de arriba) — salían líneas disparadas
    # cruzando toda la página. Se escala la profundidad total (inset +
    # separación) para que nunca pase de la mitad del ancho real del
    # trazo, manteniendo la misma proporción 70:20 — en una letra grande
    # (como la M, ~200mm) da prácticamente el valor real sin cambios; en
    # una chica se reduce proporcionalmente en vez de romperse.
    ideal_total_depth = LED_INSET_MM + LED_INSET_GAP_MM
    max_total_depth = 0.45 * width_mm
    total_depth = min(ideal_total_depth, max_total_depth) if width_mm > 0 else 0.0
    draw_inset_mm = total_depth * (LED_INSET_MM / ideal_total_depth)
    draw_gap_mm = total_depth * (LED_INSET_GAP_MM / ideal_total_depth)
    draw_segments = (
        _piece_led_segments_inset(polys, depths, width_mm, inset_mm=draw_inset_mm, gap_mm=draw_gap_mm)
        if _ShapelyPolygon is not None else []
    )
    if not draw_segments:
        draw_segments = skeleton_segments
    return {
        "perimeter_mm": perimeter_mm, "area_mm2": area_mm2, "width_mm": width_mm,
        "factor": factor, "mode": mode, "led_length_mm": led_length_mm,
        "num_modules": num_modules, "arms": arms, "draw_segments": draw_segments,
    }


def _led_module_count(led_length_mm: float, module_pitch_mm: float = LED_MODULE_PITCH_MM) -> int:
    """Cuántos módulos caben en un trazo de `led_length_mm`, a un paso
    centro a centro de `module_pitch_mm` (largo del módulo + separación
    entre módulos, igual que LED Wizard) — no son LEDs sueltos a densidad
    continua, son módulos de varios diodos cada uno (ver LEDS_PER_MODULE).
    Redondeo normal (no hacia arriba): a este paso, un trazo un poco más
    corto que un paso completo de sobra sigue necesitando solo el módulo
    que ya tenía, no uno de más. Mínimo 1 si hay algo de trazo: para la
    pieza COMPLETA (no un brazo individual, ver _arm_module_count) un
    trazo real siempre necesita al menos un módulo para encenderse."""
    if led_length_mm <= 0:
        return 0
    return max(1, round(led_length_mm / module_pitch_mm))


def _arm_module_count(length_mm: float, module_pitch_mm: float = LED_MODULE_PITCH_MM) -> int:
    """Como _led_module_count pero SIN el mínimo de 1 — para un BRAZO
    individual (ver _piece_skeleton_mm), no la pieza completa: en una
    letra chica (2-3cm) el esqueleto rasterizado a veces sale fragmentado
    en varios brazos cortos de puro ruido de borde (2-40mm cada uno, muy
    por debajo de un módulo de 6.5cm) en vez de ramificación real — forzar
    un módulo en CADA brazo, por chico que sea, multiplica el conteo real
    varias veces (visto real: una letra de 86mm de trazo total, con el
    piso por pieza completa 1 módulo, con el piso por brazo 7). El piso de
    "al menos 1" se aplica una sola vez, a la SUMA de todos los brazos de
    la pieza (ver _piece_led_stats) — un brazo individual de verdad puede
    redondear a 0 sin problema, mientras la pieza en conjunto no se quede
    sin ningún módulo."""
    if length_mm <= 0:
        return 0
    return max(0, round(length_mm / module_pitch_mm))


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
    trazo de la letra es grueso (≥ LED_THICK_STROKE_MM) se usa una vuelta
    pegada hacia adentro del borde en vez del eje medio por el centro (ver
    _piece_led_stats).

    La cantidad de LEDs sale de módulos reales (6.5cm cada uno) acomodados
    a un paso centro a centro de `module_pitch_mm`, por BRAZO (cada tramo
    entre ramificaciones — ver _piece_skeleton_mm/_piece_led_stats) y no
    de un tirón para toda la pieza, igual que un diseño real de módulos
    LED, no "metros × densidad continua".

    El grosor real de un trazo no es una medida directa de la geometría (un
    polígono no "sabe" su propio ancho) — se aproxima con distance
    transform sobre el eje medio (ver _piece_skeleton_mm).

    None si no se pudo identificar/re-emparejar el grupo (mismo caso que
    extract_group_polygons_mm)."""
    pieces = extract_group_polygons_mm(cdr_pdf_path, letter_heights)
    if pieces is None:
        return None
    results = []
    for polys in pieces:
        stats = _piece_led_stats(polys, module_pitch_mm=module_pitch_mm)
        led_meters = stats["led_length_mm"] / 1000.0
        num_modules = stats["num_modules"]
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
    inset_mm: float = LED_INSET_MM, gap_mm: float = LED_INSET_GAP_MM,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """DOS vueltas, las dos pegadas hacia adentro del MISMO borde (una a
    `inset_mm`, la otra a `inset_mm + gap_mm` — ver LED_INSET_GAP_MM;
    confirmado real contra una segunda referencia del cliente: una M con
    doble columna de LEDs cerca entre sí recorriendo todo su contorno, no
    una vuelta por cada orilla ni una sola línea por el medio). Si el
    trazo es más angosto que `2*inset_mm + gap_mm` la segunda vuelta puede
    salir vacía en ese tramo (no cabe un segundo renglón ahí) y solo queda
    la primera — se acepta, un trazo tan angosto físicamente no tiene
    lugar para dos filas. El NÚMERO de módulos (ver _piece_led_stats)
    sigue saliendo por eje medio — esto solo define por dónde se reparten
    esos puntos al dibujar, no cuántos son. Usa shapely porque un offset
    hacia adentro correcto en esquinas cóncavas (como el interior de una
    "e") no es un simple desplazamiento perpendicular por lado, se puede
    auto-intersectar. Devuelve una lista plana de segmentos (no
    polilíneas), consistente con _piece_skeleton_mm."""
    if _ShapelyPolygon is None or not polys:
        return []
    insets = [inset_mm, inset_mm + gap_mm]
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
            # "round" en vez de "mitre": una esquina MUY aguda (el ápice de
            # una "A") con mitre se estira sin límite por diseño (el largo
            # de la punta de un mitre crece con 1/sin(ángulo/2), no con la
            # distancia del offset) — confirmado real: hasta con el inset
            # ya escalado por tamaño (ver _piece_led_stats) seguía
            # saliendo una puntita disparada hacia afuera en cada "A".
            # "round" nunca se dispara, en la esquina más cerrada como
            # mucho queda ligeramente redondeada.
            offset = shp.buffer(sign * ins, join_style="round")
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
    """Segmentos del trazo de LED de una pieza, para dibujar — siempre dos
    vueltas pegadas hacia adentro de cada borde (ver "draw_segments" en
    _piece_led_stats), independiente de cómo se cuentan los módulos."""
    if not polys:
        return []
    if depths is None:
        depths = _polygon_nesting_depths(polys)
    if stats is None:
        stats = _piece_led_stats(polys, depths=depths)
    return stats["draw_segments"]


def _sample_led_points(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], num_modules: int,
) -> List[Tuple[float, float]]:
    """`num_modules` puntos repartidos UNIFORMEMENTE por longitud de arco a
    lo largo de una lista de segmentos (no hace falta que formen una sola
    polilínea continua — el eje medio de una letra ramificada no lo es,
    ver _skeleton_dfs_segments) — para marcar dónde cae cada módulo sobre
    el trazo.

    `num_modules` es el mismo número que ya calculó _led_module_count a
    partir del largo total (no una marca "cada spacing_mm desde el
    principio" calculada aparte): antes la cantidad de puntos dibujados
    podía NO coincidir con el número de módulos de la etiqueta (p.ej. un
    trazo de 2.3 pasos de largo redondeaba a "2 módulos" en la etiqueta
    pero el muestreo por spacing fijo dibujaba 3 puntos). Cada módulo i
    (0-indexado) se ubica al CENTRO de su propio tramo: a una distancia
    acumulada de (i+0.5)/num_modules del largo total del trazo — con eso
    la separación entre puntos consecutivos es siempre la misma (≈ el paso
    real), sin el hueco más grande de lo normal que dejaba el filtro
    anterior de cercanía euclidiana al descartar puntos amontonados en una
    esquina."""
    if num_modules <= 0 or not segments:
        return []
    total_len = sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in segments)
    if total_len <= 0:
        return []
    targets = [(i + 0.5) / num_modules * total_len for i in range(num_modules)]

    pts: List[Tuple[float, float]] = []
    accum = 0.0
    ti = 0
    for (x1, y1), (x2, y2) in segments:
        seg_len = math.hypot(x2 - x1, y2 - y1)
        while ti < len(targets) and targets[ti] <= accum + seg_len:
            t = (targets[ti] - accum) / seg_len if seg_len > 0 else 0.0
            pts.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
            ti += 1
        accum += seg_len
    # Cola por error de redondeo de punto flotante (el último target puede
    # quedar epsilon por encima del largo total acumulado).
    while ti < len(targets):
        pts.append(segments[-1][1])
        ti += 1
    return pts
