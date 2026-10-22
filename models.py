"""
Modelos de base de datos (SQLAlchemy) de la calculadora.

Se separó de app.py (ver auditoría de reestructuración). Importa `db`
desde extensions.py (no desde app.py): si importara `from app import db`,
correr `python app.py` directamente cargaría ese archivo dos veces bajo
dos nombres de módulo distintos ("__main__" y "app") y rompería el import
circular — ver extensions.py.
"""
from datetime import datetime, timezone

from flask_login import UserMixin

from extensions import db


class User(UserMixin, db.Model):
    __tablename__ = "lc_user"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin      = db.Column(db.Boolean, default=False, nullable=False)
    created_at    = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    calculations  = db.relationship("Calculation", backref="user", lazy=True,
                                   cascade="all, delete-orphan")


class Calculation(db.Model):
    __tablename__ = "lc_calculation"
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.Integer, db.ForeignKey("lc_user.id"), nullable=False)
    name            = db.Column(db.String(120), nullable=False, default="Sin nombre")
    created_at      = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    rows_json       = db.Column(db.Text, nullable=False)
    params_json     = db.Column(db.Text, nullable=False)
    results_json    = db.Column(db.Text, nullable=False)
    image_url       = db.Column(db.String(512), nullable=True)
    nesting_pdf_url = db.Column(db.String(512), nullable=True)
    led_brand       = db.Column(db.String(20), nullable=True)
    led_diagram_url = db.Column(db.String(512), nullable=True)
