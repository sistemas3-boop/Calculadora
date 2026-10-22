"""
Sugerencia de materiales a partir de las filas (área/subproceso) que el
usuario aplicó en la calculadora.

Se separó de app.py (ver auditoría de reestructuración). Función pura:
recibe listas/diccionarios, no toca Flask/DB.
"""
from typing import Dict, List

_MATERIAL_RULES = [
    (lambda area, sub, brand: area == "CORTE" and "acrílico" in sub.lower(),
     "CORTE", "Acrílico 3 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "CORTE" and "lámina" in sub.lower(),
     "CORTE", "Aluminio blanco C.19", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "CORTE" and "trovicel" in sub.lower(),
     "CORTE", "Trovicel 10 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "CANTO",
     "CANTO", "Canto", ""),
    (lambda area, sub, brand: area == "TRIMCAP",
     "TRIMCAP", 'Silvatrim 1"', "El color varía"),
    (lambda area, sub, brand: area == "TERMOFORMADO",
     "TERMOFORMADO", "Acrílico blanco 3 mm", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "ROTULACIÓN",
     "ROTULACIÓN", "Vinil impreso", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "TENSADO",
     "TENSADO", "Lona translúcida", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "PINTURA",
     "PINTURA", "Pintura para cantos", "Brillante / Mate / Semi mate"),
    (lambda area, sub, brand: area == "PINTURA",
     "PINTURA", "Fondear estructuras", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "PINTURA",
     "PINTURA", "Pintura interior de letras", "El color depende de la variabilidad del componente"),
    (lambda area, sub, brand: area == "PINTURA",
     "PINTURA", "Bicapa", "Mate / Transparente"),
    (lambda area, sub, brand: area == "ILUMINACIÓN" and sub == "Neón",
     "ILUMINACIÓN", "NEON FLEX BLANCO FRIO SIMETRICA", ""),
    (lambda area, sub, brand: area == "ILUMINACIÓN" and sub == "Neón",
     "ILUMINACIÓN", "NEON FLEX BLANCO CALIDO SIMETRICA", ""),
    # Iluminación LED: el módulo depende de si el anuncio es para Dekorama
    # o para Lumicolor (ver selector en el formulario).
    (lambda area, sub, brand: area == "ILUMINACIÓN" and sub != "Neón" and brand == "dekorama",
     "ILUMINACIÓN", "Módulo led Interwan blanco frío o cálido", ""),
    (lambda area, sub, brand: area == "ILUMINACIÓN" and sub != "Neón" and brand != "dekorama",
     "ILUMINACIÓN", "Módulo led Samsung blanco frío", ""),
]


def get_material_suggestions(rows: List[Dict], params: Dict = None, led_brand: str = "lumicolor") -> List[Dict]:
    seen: set = set()
    suggestions: List[Dict] = []
    for row in rows:
        area = row.get("area", "")
        sub  = row.get("subprocess", "")
        for match_fn, tag, material, note in _MATERIAL_RULES:
            if material not in seen and match_fn(area, sub, led_brand):
                seen.add(material)
                suggestions.append({"area": tag, "material": material, "note": note})

    # Perímetro real (contorno + huecos) de las filas de CANTO/TRIMCAP que
    # traen letra por letra geométrica — cuántos metros de canto/Silvatrim
    # hay que pedir, no solo el color.
    for mat_area in ("CANTO", "TRIMCAP"):
        perimeter_cm = sum(
            row["perimeter_cm"] for row in rows
            if row.get("area") == mat_area and row.get("perimeter_cm")
        )
        if perimeter_cm:
            for s in suggestions:
                if s["area"] == mat_area:
                    s["note"] = f'{s["note"]} · {perimeter_cm / 100:.2f} m necesarios'.strip(" ·")

    # Metros lineales de LED (contorno real, doblado en trazos gruesos —
    # ver nesting.piece_led_meters) de las filas de ILUMINACIÓN con letra
    # por letra geométrica — cuántos metros de tira/módulos LED pedir.
    led_meters_total = sum(
        row["led_meters"] for row in rows
        if row.get("area") == "ILUMINACIÓN" and row.get("led_meters")
    )
    led_count_total = sum(
        row["led_count"] for row in rows
        if row.get("area") == "ILUMINACIÓN" and row.get("led_count")
    )
    if led_meters_total:
        for s in suggestions:
            if s["area"] == "ILUMINACIÓN" and "led" in s["material"].lower():
                extra = f'{led_meters_total:.2f} m de LED (~{led_count_total} LEDs)'
                s["note"] = f'{s["note"]} · {extra}'.strip(" ·")

    return suggestions
