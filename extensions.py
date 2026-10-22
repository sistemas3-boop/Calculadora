"""
Instancia compartida de Flask-SQLAlchemy, separada de app.py a propósito.

`models.py` necesita `db` para definir sus clases, y `app.py` necesita
`models.py` para las rutas — si `db` viviera en app.py (como antes, con
`db = SQLAlchemy(app)` inline), `models.py` tendría que hacer
`from app import db`, y correr `python app.py` directamente (como hace el
CMD del Dockerfile) carga ese archivo dos veces bajo dos nombres de módulo
distintos ("__main__" y "app"), rompiendo el import circular. Este módulo
sin ninguna dependencia de `app` evita el problema por completo.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
