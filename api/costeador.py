"""
Endpoint(s) que llamará el módulo "Costeador" de Odoo.

Estado: esqueleto. Recibe y valida la forma del request, pero delega el
cálculo real a services.cotizador.calcular_cotizacion, que todavía no está
implementado (nesting, materiales, LEDs, etc. siguen por definir).
"""
import json

from flask import jsonify, request

from . import api_bp, require_api_key
from services.cotizador import calcular_cotizacion


@api_bp.route("/costeador/cotizar", methods=["POST"])
@require_api_key
def cotizar():
    """
    Request esperado (multipart/form-data):
      - archivos: uno o más archivos vectorizados (.cdt, .pdf, .ai)
      - params: JSON string con los filtros/medidas que hoy se llenan a mano
                en la calculadora, por ejemplo (forma tentativa, por
                confirmar contra el diseño final del Costeador):
            {
              "categoria": "Anuncios" | "Dekorama",
              "proyecto_nuevo": true,
              "oportunidad_id": 123,
              "medidas": {...}
            }

    Response (JSON): solo resultados (cotización, materiales, horas por
    puesto de trabajo) — ver services/cotizador.py. Este endpoint no
    escribe nada en Odoo; eso lo hace el módulo Costeador con la respuesta.
    """
    params_raw = request.form.get("params")
    if not params_raw:
        return jsonify(error="Falta 'params' (JSON) en el form-data"), 400
    try:
        params = json.loads(params_raw)
    except json.JSONDecodeError:
        return jsonify(error="'params' no es JSON valido"), 400

    archivos = request.files.getlist("archivos")

    resultado = calcular_cotizacion(params=params, archivos=archivos)
    return jsonify(resultado)
