"""
Motor de calculo del "Costeador": nesting, materiales, LEDs/fuente de
poder, silvatrim e iluminacion, mano de obra por puesto de trabajo.

Deliberadamente NO implementado todavia - es el esqueleto para que
api/costeador.py tenga algo que llamar mientras se define el algoritmo
real. No toca nada de la calculadora que ya esta en produccion (app.py).

Cuando se implemente, debe quedar como funcion pura: recibe datos, regresa
datos, sin tocar Flask/request/db directamente. Asi la puede usar tanto
este endpoint como, si se decide mas adelante, la interfaz web existente
(hoy esa logica esta embebida en app.py:index()).
"""
from typing import Any, Dict, List


def calcular_cotizacion(params: Dict[str, Any], archivos: List) -> Dict[str, Any]:
    """
    params: filtros/medidas que hoy se llenan a mano en la calculadora
            (ver docstring de api.costeador.cotizar para la forma esperada)
    archivos: FileStorage de Flask, uno por archivo vectorizado subido

    Debe regresar unicamente resultados (por diseno, ver
    /act_calculadora.txt - el Costeador en Odoo es quien escribe en
    CRM/Costos/Puestos de Trabajo con esta respuesta):
        {
          "cotizacion": {...},        # desglose de materiales + mano de obra
          "horas_por_puesto": {...},  # soldadura, pintura, ensamble, canto...
          "materiales": [...],
        }
    """
    raise NotImplementedError(
        "Motor de calculo pendiente de definir (nesting, materiales, LEDs). "
        "Ver /act_calculadora.txt."
    )
