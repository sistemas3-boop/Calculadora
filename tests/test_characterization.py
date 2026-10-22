"""Pruebas de caracterizacion: congelan el comportamiento ACTUAL del
pipeline de LEDs/nesting y de las rutas principales, de cara a la
reestructuracion de app.py/nesting.py en modulos mas chicos (fase 3).

No validan que los numeros sean "correctos" contra la realidad -- esa
validacion contra referencias del cliente ya se hizo a mano durante el
desarrollo (ver DIAGRAMA.md). Aqui solo se fija el resultado de HOY para
detectar cualquier cambio de comportamiento mientras se mueve codigo de
un archivo a otro. Si una de estas pruebas empieza a fallar durante la
reestructuracion, es señal de una regresion real, no de que haya que
"corregir" el valor esperado.

Usan archivos reales en uploads/ (gitignored, no viajan al repo) -- si
no estan presentes en esta maquina, el caso se salta en vez de fallar.
"""
import json
import os
import shutil
import sys

import pytest
from werkzeug.datastructures import MultiDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app as appmod  # noqa: E402
import image_analysis  # noqa: E402
import nesting  # noqa: E402

UPLOADS = os.path.join(ROOT, "uploads")

# Nombre corto -> archivo PDF ya convertido (post-LibreOffice) usado como
# referencia durante esta sesion de desarrollo.
REFERENCE_PDFS = {
    "zacatecas_chico": "1792168253_d02f67b1_zacatecas.pdf",
    "zacatecas_grande": "1792443795_4a2c2534_zacatecas_grande.pdf",
    "mifelgrande": "1792527260_81a61e1f_Mifelgrande.pdf",
}

EXPECTED_GROUP_HEIGHT_CM = {
    "zacatecas_chico": 69.7,
    "zacatecas_grande": 69.7,
    "mifelgrande": 150.0,
}

EXPECTED_PER_PIECE_MODULES = {
    "zacatecas_chico": [10, 13, 13, 8, 11, 13, 11, 10, 10],
    "zacatecas_grande": [10, 12, 10, 12, 8, 11, 10, 12, 11],
    "mifelgrande": [20, 74, 20, 30, 20, 36, 20, 20, 20],
}


def _pdf_path(name: str) -> str:
    path = os.path.join(UPLOADS, REFERENCE_PDFS[name])
    if not os.path.exists(path):
        pytest.skip(f"fixture local no disponible: {path} (uploads/ es gitignored)")
    return path


@pytest.mark.parametrize("name", list(REFERENCE_PDFS))
def test_group_detection_unchanged(name):
    path = _pdf_path(name)
    groups = image_analysis._extract_pieces_geometric(path)
    assert len(groups) == 1
    group = groups[0]
    assert group["letters"] == 9
    assert abs(group["height_cm"] - EXPECTED_GROUP_HEIGHT_CM[name]) < 0.1


@pytest.mark.parametrize("name", list(REFERENCE_PDFS))
def test_led_pipeline_totals_unchanged(name):
    path = _pdf_path(name)
    group = image_analysis._extract_pieces_geometric(path)[0]
    heights = [p["height_cm"] for p in group["pieces"]]

    results = nesting.piece_led_meters(path, heights)

    assert results is not None, "no se pudo re-emparejar el grupo (extract_group_polygons_mm devolvio None)"
    per_piece = [r["num_modules"] for r in results]
    assert per_piece == EXPECTED_PER_PIECE_MODULES[name]
    assert sum(per_piece) == sum(EXPECTED_PER_PIECE_MODULES[name])


MATERIAL_CASES = [
    ("Corte de acrílico blanco 3mm", ("Acrílico blanco", 1800.0, 2400.0)),
    ("Corte de lámina galvanizada cal. 19", ("Lámina cal. 19", 1220.0, 3000.0)),
    ("Corte de trovicel 5mm", ("Trovicel", 1220.0, 2440.0)),
    ("Doblado de perfil", None),
]


@pytest.mark.parametrize("subprocess, expected", MATERIAL_CASES)
def test_material_for_subprocess_unchanged(subprocess, expected):
    assert nesting.material_for_subprocess(subprocess) == expected


def _logged_in_client():
    with appmod.app.app_context():
        user = appmod.User.query.first()
    if user is None:
        pytest.skip("no hay usuarios en la base local (instance/calculadora.db) para simular login")
    client = appmod.app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user.id)
        sess["_fresh"] = True
    return client


def test_index_route_smoke():
    client = _logged_in_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Calculadora" in resp.data or b"calculadora" in resp.data


def test_history_route_smoke():
    client = _logged_in_client()
    resp = client.get("/history")
    assert resp.status_code == 200


def _admin_client():
    with appmod.app.app_context():
        user = appmod.User.query.filter_by(is_admin=True).first()
    if user is None:
        pytest.skip("no hay usuario admin en la base local para simular login")
    client = appmod.app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user.id)
        sess["_fresh"] = True
    return client


def _non_admin_client():
    with appmod.app.app_context():
        user = appmod.User.query.filter_by(is_admin=False).first()
    if user is None:
        pytest.skip("no hay usuario no-admin en la base local para simular login")
    client = appmod.app.test_client()
    with client.session_transaction() as sess:
        sess["_user_id"] = str(user.id)
        sess["_fresh"] = True
    return client


@pytest.fixture
def isolated_catalog(monkeypatch, tmp_path):
    """Copia data/times.json a un archivo temporal y redirige appmod.DATA_FILE
    ahi -- las rutas admin_tiempos_* escriben directo sobre DATA_FILE (ver
    app.py), y ese archivo es el catalogo REAL de produccion, no un dato de
    prueba. Sin este aislamiento, correr estos tests pisaria el catalogo
    real del cliente. monkeypatch restaura appmod.DATA_FILE solo al terminar
    el test (exito o falla); aqui ademas se verifica que el archivo REAL
    quedo intacto, como cinturon y tirantes."""
    real_path = appmod.DATA_FILE
    original_content = open(real_path, "r", encoding="utf-8").read()
    tmp_file = tmp_path / "times.json"
    shutil.copyfile(real_path, tmp_file)
    monkeypatch.setattr(appmod, "DATA_FILE", str(tmp_file))
    yield tmp_file
    assert open(real_path, "r", encoding="utf-8").read() == original_content, (
        "el catalogo REAL (data/times.json) se modifico -- el aislamiento del fixture fallo"
    )


def test_admin_tiempos_requiere_admin():
    client = _non_admin_client()
    resp = client.get("/admin/tiempos")
    assert resp.status_code == 302  # redirige a index, no 200 ni error


def test_admin_tiempos_zona_agregar_y_eliminar(isolated_catalog):
    client = _admin_client()

    resp = client.post("/admin/tiempos/zona/agregar", data={"nombre_zona": "zona_test_auditoria"})
    assert resp.status_code == 302
    catalog = json.loads(isolated_catalog.read_text(encoding="utf-8"))
    assert "ZONA_TEST_AUDITORIA" in catalog  # la ruta normaliza a mayusculas
    assert catalog["ZONA_TEST_AUDITORIA"] == {}

    resp = client.post("/admin/tiempos/zona/eliminar", data={"zona": "ZONA_TEST_AUDITORIA"})
    assert resp.status_code == 302
    catalog = json.loads(isolated_catalog.read_text(encoding="utf-8"))
    assert "ZONA_TEST_AUDITORIA" not in catalog


def test_admin_tiempos_proceso_agregar_editar_eliminar(isolated_catalog):
    client = _admin_client()

    resp = client.post("/admin/tiempos/proceso/agregar", data={
        "zona": "CORTE", "nombre_proceso": "Proceso de prueba (auditoria)",
        "codigo": "zzz-test", "tiempo": "1.23", "desc": "50 cm",
    })
    assert resp.status_code == 302
    catalog = json.loads(isolated_catalog.read_text(encoding="utf-8"))
    assert catalog["CORTE"]["Proceso de prueba (auditoria)"]["ZZZ-TEST"] == {"tiempo": 1.23, "desc": "50 cm"}

    resp = client.post("/admin/tiempos/editar", data={"CORTE|Proceso de prueba (auditoria)|ZZZ-TEST": "4.56"})
    assert resp.status_code == 302
    catalog = json.loads(isolated_catalog.read_text(encoding="utf-8"))
    assert catalog["CORTE"]["Proceso de prueba (auditoria)"]["ZZZ-TEST"]["tiempo"] == 4.56

    resp = client.post("/admin/tiempos/proceso/eliminar", data={
        "zona": "CORTE", "proceso": "Proceso de prueba (auditoria)",
    })
    assert resp.status_code == 302
    catalog = json.loads(isolated_catalog.read_text(encoding="utf-8"))
    assert "Proceso de prueba (auditoria)" not in catalog["CORTE"]


INDEX_FORM_BASE = {
    "hours_per_day": "9", "cushion_pct": "10", "packaging_minutes": "40",
    "colores_pintura": "1", "secado_minutes": "120",
    "calc_name": "TEST_AUDITORIA_caracterizacion", "led_brand": "lumicolor",
    "cdr_filename": "",
}


def _post_index_and_capture(client, rows):
    """POSTea a index() con las filas dadas (lista de dicts con area/
    subprocess/group/quantity/width/height/heights_detail) y devuelve el
    rows_json guardado -- borra el Calculation creado antes de salir, para
    no dejar basura de prueba en la base real del usuario."""
    data = list(INDEX_FORM_BASE.items())
    for row in rows:
        data.append(("areas[]", row.get("area", "")))
        data.append(("subprocesses[]", row.get("subprocess", "")))
        data.append(("groups[]", row.get("group", "")))
        data.append(("quantities[]", row.get("quantity", "")))
        data.append(("widths[]", row.get("width", "")))
        data.append(("heights[]", row.get("height", "")))
        data.append(("heights_detail[]", row.get("heights_detail", "")))
    cdr_filename = rows[0].get("cdr_filename") if rows else None
    if cdr_filename:
        data = [(k, v) if k != "cdr_filename" else (k, cdr_filename) for k, v in data]

    resp = client.post("/", data=MultiDict(data))
    assert resp.status_code == 302, f"esperaba redirect a /result/<id>, dio {resp.status_code}"
    calc_id = int(resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1])
    with appmod.app.app_context():
        calc = appmod.db.session.get(appmod.Calculation, calc_id)
        result = {
            "rows": json.loads(calc.rows_json),
            "nesting_pdf_url": calc.nesting_pdf_url,
            "led_diagram_url": calc.led_diagram_url,
        }
        appmod.db.session.delete(calc)
        appmod.db.session.commit()
    return result


def test_index_post_simple_group_lookup():
    client = _logged_in_client()
    result = _post_index_and_capture(client, [{
        "area": "CORTE", "subprocess": "Acrílico blanco (3 mm sin rebaje perimetral)",
        "group": "CTE-A1", "quantity": "2",
    }])
    assert result["rows"] == [{
        "area": "CORTE", "subprocess": "Acrílico blanco (3 mm sin rebaje perimetral)",
        "group": "CTE-A1", "quantity": 2.0, "mins_per_piece": 3.5, "total_mins": 7.0,
        "perimeter_cm": None, "led_meters": None, "led_count": None,
    }]
    assert result["nesting_pdf_url"] is None
    assert result["led_diagram_url"] is None


def test_index_post_regresion_lineal():
    client = _logged_in_client()
    result = _post_index_and_capture(client, [{
        "area": "CORTE", "subprocess": "Acrílico blanco (3 mm sin rebaje perimetral)",
        "group": "Regresión", "quantity": "3", "height": "60",
    }])
    assert result["rows"] == [{
        "area": "CORTE", "subprocess": "Acrílico blanco (3 mm sin rebaje perimetral)",
        "group": "Regresión lineal (60 cm)", "quantity": 3.0, "mins_per_piece": 4.65, "total_mins": 13.94,
        "perimeter_cm": None, "led_meters": None, "led_count": None,
    }]


def test_index_post_letter_by_letter_full_pipeline():
    """El caso mas completo del formulario: alturas letra-por-letra sobre
    un .cdr real, para una fila de CORTE (dispara nesting real, ver
    _build_nesting_pdf) y una de ILUMINACION (dispara el diagrama de LED,
    ver _build_led_diagram_pdf) -- el mismo .cdr/alturas que validan
    test_led_pipeline_totals_unchanged (mifelgrande, 260 modulos/9
    letras)."""
    cdr_path = os.path.join(UPLOADS, "1792527260_81a61e1f_Mifelgrande.cdr")
    if not os.path.exists(cdr_path):
        pytest.skip(f"fixture local no disponible: {cdr_path} (uploads/ es gitignored)")
    if not appmod.LIBREOFFICE_AVAILABLE:
        pytest.skip("LibreOffice no disponible en esta maquina")

    heights_json = json.dumps([99.8, 99.8, 99.8, 99.8, 127.7, 127.7, 127.7, 100.5, 127.7])
    client = _logged_in_client()
    result = _post_index_and_capture(client, [
        {
            "area": "CORTE", "subprocess": "Acrílico blanco (3 mm sin rebaje perimetral)",
            "group": "Regresión", "heights_detail": heights_json,
            "cdr_filename": "1792527260_81a61e1f_Mifelgrande.cdr",
        },
        {
            "area": "ILUMINACIÓN", "subprocess": "Iluminación",
            "group": "Regresión", "heights_detail": heights_json,
            "cdr_filename": "1792527260_81a61e1f_Mifelgrande.cdr",
        },
    ])

    corte_row, led_row = result["rows"]
    assert corte_row["quantity"] == 9.0
    assert led_row["quantity"] == 9.0
    assert led_row["led_count"] == 260  # mismo total validado en test_led_pipeline_totals_unchanged
    assert led_row["led_meters"] == 43.78
    assert result["nesting_pdf_url"] is not None
    assert result["led_diagram_url"] is not None
