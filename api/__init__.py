"""
Blueprint para el API que consumirá el módulo "Costeador" de Odoo.

Esqueleto: intencionalmente NO está registrado en app.py todavía. Mientras
no se haga `app.register_blueprint(api_bp)` en app.py, estas rutas no
existen para la aplicación en producción — cero riesgo para lo que ya
está corriendo.

Ver /act_calculadora.txt para el flujo completo planeado.
"""
import os
from functools import wraps

from flask import Blueprint, jsonify, request

api_bp = Blueprint("api", __name__, url_prefix="/api")


def require_api_key(view):
    """Protege endpoints que solo Odoo (u otro servicio de confianza) debe llamar.

    La clave esperada se lee de la variable de entorno COSTEADOR_API_KEY. Si
    no está configurada, el endpoint queda inalcanzable (500) en vez de
    quedar abierto por accidente.
    """
    @wraps(view)
    def wrapped(*args, **kwargs):
        expected = os.environ.get("COSTEADOR_API_KEY")
        if not expected:
            return jsonify(error="API no configurada: falta COSTEADOR_API_KEY"), 500
        provided = request.headers.get("X-API-Key", "")
        if provided != expected:
            return jsonify(error="No autorizado"), 401
        return view(*args, **kwargs)
    return wrapped


from . import costeador  # noqa: E402,F401  registra las rutas sobre api_bp
