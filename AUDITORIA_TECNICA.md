# Auditoría técnica — reestructuración de `app.py` / `nesting.py`

**Fecha:** 2026-10-22
**Rama:** `feature/led-calculo`
**Alcance de este reporte:** división de los dos módulos monolíticos del proyecto (`nesting.py` y `app.py`) en archivos más chicos y enfocados, preservando el comportamiento exacto de la calculadora. No se tocó ninguna regla de negocio, endpoint, plantilla ni esquema de base de datos.

---

## 1. Estado inicial

### Arquitectura original

El proyecto tenía toda su lógica concentrada en dos archivos:

| Archivo | Líneas | Contenido |
|---|---|---|
| `app.py` | 2276 | App Flask, modelos SQLAlchemy, auth, catálogo de tiempos, sugerencia de materiales, análisis de `.cdr`/PDF/imagen (OCR, clustering, extracción geométrica), generación/merge de PDF, y las 19 rutas. |
| `nesting.py` | 2254 | Extracción de contorno real de cada letra, esqueleto/eje medio, conteo de módulos de LED, nesting (empacado en hoja), dibujo de planos y diagramas en PDF (matplotlib). |

`api/` y `services/` ya existían como paquetes separados, pero son un esqueleto intencional y documentado para una integración futura con Odoo ("Costeador") — no registrado en `app.py`, sin código propio que auditar.

### Problemas detectados (antes de esta fase)

Ya resueltos en trabajo previo de esta misma sesión, no en el alcance de esta fase pero relevantes como antecedente:

- Secret key de Flask y contraseña de admin hardcodeadas → movidas a variables de entorno con fallback seguro autogenerado.
- `OPENAI_API_KEY` sin uso (el proyecto ya no llama a GPT-4o) → eliminada de `.env`.
- `pandas`, `openpyxl`, `pillow` en `requirements.txt` sin ningún `import` real en el proyecto → eliminadas.
- `Dockerfile` no instalaba `tesseract-ocr`/`tesseract-ocr-spa` → el análisis OCR de imágenes quedaba silenciosamente roto en producción.
- Dos funciones de clustering (`_cluster_rects_by_height` / `_cluster_boxes_by_height_px`) con algoritmo duplicado → unificadas en un núcleo compartido parametrizado.
- 3 archivos basura de 0 bytes (`=`, `CACHED`, `[internal]`) y una copia duplicada de `calculadora.db` en la raíz, de commits antiguos.

### Riesgos identificados para esta fase

- **Ningún test automatizado existía** antes de esta fase — cualquier reorganización de código se verificaba a mano, caso por caso.
- Los dos monolitos mezclaban capas muy distintas (framework web, I/O de archivos, algoritmos geométricos puros, dibujo) en el mismo archivo, dificultando ubicar y modificar código con confianza.
- Riesgo real de import circular al separar `app.py`: varias funciones movidas necesitan `app.config` o modelos que hoy viven en el mismo archivo que las rutas.

---

## 2. Modificaciones realizadas

### Paso 0 — Red de seguridad antes de mover código

Como no existían pruebas automatizadas, se creó `tests/test_characterization.py` (pytest) que fija el comportamiento **actual** del sistema como línea base, antes de mover una sola línea:

- Conteo total y por-pieza de módulos de LED (`piece_led_meters`) contra 3 archivos PDF de referencia reales (ya usados antes en esta sesión para calibrar las categorías de paso de LED) — incluye el caso de doble hilera (la "M" de Mifelgrande, 74 módulos).
- Detección de grupo de letras (`_extract_pieces_geometric`) sobre esos mismos 3 archivos.
- `material_for_subprocess` con 4 casos (los 3 materiales conocidos + uno que no debe matchear).
- Smoke test de `/` y `/history` vía `test_client()` con sesión simulada.

Los archivos PDF de referencia viven en `uploads/` (gitignored, datos reales de cliente) — los tests se saltan solos si no están presentes en la máquina, en vez de fallar.

### Paso 1 — División de `nesting.py` (2254 → 123 líneas de fachada)

| Módulo nuevo | Líneas | Contenido |
|---|---|---|
| `geometry.py` | 952 | Extracción del contorno real desde el PDF convertido, esqueleto/eje medio (Zhang-Suen, MST, podado de espolones), utilidades de polígono (área, centroide, profundidad de anidado). |
| `led_layout.py` | 488 | Constantes de módulo/pitch de LED, conteo de módulos por pieza (`_piece_led_stats`, `piece_led_meters`), trazo de doble vuelta para el dibujo. |
| `sheet_packing.py` | 412 | Nesting esquina-a-esquina sobre la hoja (`nest_pieces`), tamaño de hoja por material (`material_for_subprocess`). |
| `nesting_pdf.py` | 474 | Dibujo del plano de corte y del diagrama de LED en PDF (matplotlib/`PdfPages`). |
| `nesting.py` | 123 | Fachada: re-exporta explícitamente los mismos nombres (públicos y privados) que tenía antes, para que `app.py` y los tests no necesiten cambiar sus imports. |

Orden de dependencia entre ellos: `geometry` ← `led_layout` ← `sheet_packing` ← `nesting_pdf` (sin ciclos).

### Paso 2 — División de `app.py` (2276 → 821 líneas)

| Módulo nuevo | Líneas | Contenido |
|---|---|---|
| `extensions.py` | 14 | `db = SQLAlchemy()` sin inicializar — ver el hallazgo de la sección siguiente. |
| `models.py` | 40 | `User`, `Calculation`. |
| `materials.py` | 89 | Reglas de sugerencia de materiales (`_MATERIAL_RULES`, `get_material_suggestions`) — función pura, sin dependencia de Flask. |
| `image_analysis.py` | 1300 | Todo el análisis de `.cdr`/PDF/imagen: detección de LibreOffice/Tesseract, parsing de "Medidas: A x B mm", clustering de letras por alto, OCR de cotas dibujadas, extracción geométrica de piezas. |
| `pdf_export.py` | 114 | Orquestación de la generación de PDF (plano de corte + diagrama de LED) y los helpers de ruta/conteo de páginas para un cálculo guardado. |
| `app.py` | 821 | Queda solo con: creación de la app Flask, config, login manager, el catálogo de tiempos (`load_catalog`/`get_group_time`/`compute_regression`, ver nota de alcance abajo) y las 19 rutas. |

**Nota de alcance:** `load_catalog`, `get_group_time`, `_avg_height_from_desc` y `compute_regression` (lectura/regresión sobre `data/times.json`) se dejaron en `app.py`. No encajan en ninguno de los 4 módulos acordados (no son de materiales, análisis de imagen, PDF ni modelos) y son chicas (~50 líneas) — se decidió no inventar un quinto módulo fuera de lo discutido.

### Hallazgo durante la verificación: import circular real

Al probar el comando de arranque real (`python app.py`, el mismo que usa el `CMD` del `Dockerfile`), el proceso truena con:

```
ImportError: cannot import name 'User' from partially initialized module 'models'
(most likely due to a circular import)
```

**Causa:** `models.py` hacía `from app import db` y `pdf_export.py` hacía `from app import app`. Cuando Python ejecuta `app.py` directamente, lo registra como módulo `__main__`, no como `app` — al llegar a `from models import ...`, tiene que cargar `app.py` *otra vez* desde cero bajo el nombre `app`, lo que dispara un ciclo. Este bug **no lo detectaba** ni `import app` desde `python -c`, ni pytest, ni los tests de caracterización, porque todos esos cargan `app.py` como módulo `app` directamente, nunca como `__main__`.

**Corrección** (patrón estándar de Flask para este problema):
- `db = SQLAlchemy(app)` inline → `extensions.py` con `db = SQLAlchemy()` sin inicializar + `db.init_app(app)` en `app.py`.
- `models.py` importa `db` desde `extensions.py`, no desde `app.py`.
- `pdf_export.py` cambió `from app import app` por `from flask import current_app` (esas funciones solo se llaman dentro de una ruta, con contexto de aplicación ya activo).

Con esto, ningún módulo nuevo importa `app.py` a nivel de módulo — se rompe el ciclo por completo, sin importar si `app.py` se ejecuta como script o se importa como módulo.

### Duplicidades eliminadas

Ya reportado en el ciclo anterior de esta sesión (no repetido aquí en detalle): unificación de `_cluster_rects_by_height`/`_cluster_boxes_by_height_px` en `_cluster_boxes_by_height` (ahora vive en `image_analysis.py`).

### Mejoras de rendimiento

Ninguna buscada ni medida en esta fase — el objetivo fue exclusivamente reorganización sin cambio de comportamiento, tal como pedía el alcance.

---

## 3. Archivos eliminados

| Archivo | Motivo | Verificación | Recuperación |
|---|---|---|---|
| `=` (raíz) | Archivo de 0 bytes, basura del primer commit del repo. | Sin referencias en ningún `.py`/config/template. | `git show <commit-inicial>:=` |
| `CACHED` (raíz) | Igual que arriba. | Igual. | Igual, con su propio path. |
| `[internal]` (raíz) | Igual que arriba. | Igual. | Igual. |
| `calculadora.db` (raíz, duplicado) | Copia vacía sin seguimiento de git; la base real y activa es `instance/calculadora.db`. | Confirmado que la app lee/escribe únicamente `instance/calculadora.db` (`SQLALCHEMY_DATABASE_URI`). | No tenía historial en git — no recuperable, pero tampoco necesaria. |

Los tres primeros están `git rm`'d (quedan como "deleted" en `git status`, **no commiteados todavía**). Ninguno de los cuatro es una base de datos activa, migración, archivo subido por usuario ni credencial.

---

## 4. Arquitectura final

```
lc_flask/
├── app.py                 # Flask app, config, login, catálogo de tiempos, 19 rutas
├── extensions.py          # db = SQLAlchemy() sin inicializar
├── models.py               # User, Calculation
├── materials.py            # Sugerencia de materiales (función pura)
├── image_analysis.py       # Análisis de .cdr/PDF/imagen (OCR, clustering, extracción)
├── pdf_export.py           # Orquestación de generación de PDF (nesting + diagrama LED)
├── nesting.py              # Fachada de compatibilidad (re-exporta los 4 módulos siguientes)
├── geometry.py              # Extracción de contorno real + esqueleto/eje medio
├── led_layout.py            # Conteo de módulos de LED + trazo de ruteo
├── sheet_packing.py         # Nesting esquina-a-esquina + tamaño de hoja por material
├── nesting_pdf.py           # Dibujo de plano/diagrama en PDF (matplotlib)
├── api/                    # Esqueleto intencional, sin registrar — integración futura Odoo
├── services/                # Idem
├── templates/
├── data/                   # times.json + Excel de referencia
├── tests/
│   └── test_characterization.py
├── requirements.txt
├── requirements-dev.txt     # pytest, como dependencia de desarrollo
└── Dockerfile
```

### Relación entre componentes

```
app.py
 ├─ extensions.py (db)
 ├─ models.py (User, Calculation) ──→ extensions.py
 ├─ materials.py (get_material_suggestions)       [sin dependencias]
 ├─ image_analysis.py (analyze_*, _convert_cdr_to_pdf, ...) ──→ nesting.py (dedupe_drawings, ...)
 ├─ pdf_export.py (_build_*_pdf, _*_path, _pdf_page_count) ──→ image_analysis.py, nesting.py, flask.current_app
 └─ nesting.py (fachada)
     ├─ geometry.py ──→ image_analysis.py (import perezoso, solo dentro de extract_group_polygons_mm)
     ├─ led_layout.py ──→ geometry.py
     ├─ sheet_packing.py ──→ geometry.py
     └─ nesting_pdf.py ──→ geometry.py, led_layout.py, sheet_packing.py
```

El único import "hacia atrás" (de `geometry.py` hacia `image_analysis.py`) es perezoso (dentro de la función, no a nivel de módulo) — mismo patrón que ya usaba el código original para evitar el ciclo con `app.py`, solo que ahora apunta al módulo correcto donde realmente viven `_extract_pieces_from_pdf`/`_extract_pieces_geometric`.

---

## 5. Pruebas y resultados

### Pruebas ejecutadas

1. **Sintaxis** (`ast.parse`) sobre los 14 archivos `.py` tocados o nuevos — sin errores.
2. **Import limpio**: `python -c "import app"` — sin errores (esto es lo que NO detectó el bug circular).
3. **Arranque real**: `python app.py` (el comando del `CMD` del Dockerfile) — detectó el bug circular, y tras la corrección arranca limpio, sirviendo en el puerto 5001.
4. **HTTP real** contra el servidor ya corriendo (no `test_client`): `GET /` → 302 (redirect a login, correcto sin sesión), `GET /login` → 200, `GET /uploads/<archivo>` → 200.
5. **Suite de caracterización** (`pytest tests/`, 18 casos — ver detalle de los 6 agregados al final en "Pruebas pendientes") — pasa completa después de cada paso de la reestructuración (Paso 0, Paso 1, Paso 2, y de nuevo tras la corrección del import circular).
6. **Smoke manual end-to-end** de las funciones de generación de PDF (`build_led_diagram_pdf`, `build_nesting_pdf_multi_material`, `piece_perimeters_mm`) contra el archivo real de Mifelgrande, cruzando los 4 módulos nuevos de `nesting.py`.
7. **Smoke manual end-to-end de la ruta real `/api/analyze`** (vía `test_client`, pero disparando la ruta HTTP completa, no las funciones internas) subiendo el `.cdr` real de Mifelgrande — responde 200 con `height_cm=150.0`, `letters=9`, y los 4 flags de disponibilidad (`cv2`/`pymupdf`/`libreoffice`/`tesseract`) en `True`.

### Resultados satisfactorios

Todo lo anterior pasó. En particular, los números de LED que ya se habían validado a mano contra fichas reales del cliente en esta misma sesión (zacatecas_grande=96, mifelgrande=260, M de Mifelgrande=74 módulos) se reproducen exactamente después de mover el código a los 8 módulos nuevos.

### Pruebas pendientes / limitaciones de validación

- **Ya cubierto** (agregado después de la primera versión de este reporte): 6 pruebas nuevas de caracterización —
  - `test_index_post_simple_group_lookup` / `test_index_post_regresion_lineal`: `index()` vía POST real, verificando `rows_json`/`total_mins` exactos.
  - `test_index_post_letter_by_letter_full_pipeline`: el caso más completo — alturas letra-por-letra sobre el `.cdr` real de Mifelgrande, dispara nesting real (`_build_nesting_pdf`) Y diagrama de LED (`_build_led_diagram_pdf`) en la misma request; verifica `led_count == 260` (mismo total ya validado en `test_led_pipeline_totals_unchanged`), `nesting_pdf_url`/`led_diagram_url` no nulos.
  - `test_admin_tiempos_requiere_admin`, `test_admin_tiempos_zona_agregar_y_eliminar`, `test_admin_tiempos_proceso_agregar_editar_eliminar`: las 5 rutas `/admin/tiempos/*`, usando un fixture `isolated_catalog` que copia `data/times.json` a un archivo temporal y redirige `appmod.DATA_FILE` ahí (estas rutas escriben directo sobre el catálogo real — sin este aislamiento, correrlas pisaría datos de producción). El fixture además verifica que el archivo real quedó byte-a-byte idéntico al terminar cada test.
  - Las pruebas que crean un `Calculation` real (`_post_index_and_capture`) lo borran antes de terminar — se confirmó que no quedó ningún registro `TEST_%` en `instance/calculadora.db` ni ningún diff en `data/times.json` tras correr toda la suite.
  - 18/18 pruebas pasan (antes 12/12); la suite completa tarda ~1m40s (la mayoría es la conversión real de LibreOffice en el pipeline completo, que corre varias veces).
- `admin_usuarios` (crear/eliminar usuarios) sigue sin prueba automatizada — no se tocó porque implica contraseñas y la tabla `lc_user` real; se dejó fuera a propósito en esta ronda.
- Los PDFs/`.cdr` de referencia usados en los tests viven en `uploads/` (gitignored) — en otra máquina sin esos archivos, los casos que los necesitan se saltan en vez de ejecutarse. No hay fixtures versionadas en git (decisión deliberada: son archivos de diseño reales de clientes, no se evaluó subirlos al repo).
- No se verificó visualmente (navegador) que las plantillas sigan renderizando igual — solo se confirmó que las rutas devuelven 200/302 y el HTML se genera sin excepción.
- No hay CI configurado que corra `pytest` automáticamente en cada cambio.

### Riesgos residuales

- El import circular encontrado es la clase de bug que **solo aparece al correr el entrypoint real**, no al importar el módulo desde pruebas. Cualquier futuro cambio en el orden de imports de `app.py` debería volver a probarse con `python app.py` directamente, no solo con pytest.
- `image_analysis.py` quedó con 1300 líneas — es el módulo más grande del proyecto ahora. Está cohesionado (todo es "convertir un archivo de diseño en medidas/piezas"), pero si crece más podría valer la pena partirlo en sub-pasos (extracción de texto vs. OCR vs. clustering).

---

## 6. Recomendaciones

### Acciones que requieren autorización (no se tomaron solas)

- **Nada de esto está comiteado.** `git status` muestra todos los archivos nuevos y modificados como cambios locales sin `git commit` — se deja a decisión explícita del usuario cuándo confirmar el commit (y con qué mensaje).
- Los 3 archivos basura y el `calculadora.db` duplicado están `git rm`'d pero tampoco commiteados — mismo punto.

### Mejoras adicionales identificadas (no ejecutadas, fuera de este alcance)

- Agregar prueba de caracterización para `admin_usuarios` (crear/eliminar usuarios) — se dejó fuera deliberadamente en esta ronda por tocar contraseñas y la tabla real de usuarios.
- Versionar 1-2 archivos `.cdr`/PDF de referencia pequeños y no sensibles (o anonimizados) como fixtures de test, para que la suite no dependa de `uploads/` local.
- Configurar un workflow de CI (GitHub Actions) que corra `pytest tests/` en cada PR.

### Deuda técnica pendiente (ya documentada en el propio código, no nueva)

- El desajuste de paso de LED entre letras normales y piezas decorativas (swash) dentro de la categoría "Extra grande" — no hay señal geométrica confiable para distinguirlas automáticamente (comentario ya presente en `led_layout.py`).
- `image_analysis.py` como candidato a sub-dividir si vuelve a crecer (ver "Riesgos residuales").

---

## Criterios de éxito — checklist

- [x] La aplicación conserva todas sus funcionalidades (mismas rutas, mismos nombres de endpoint).
- [x] Los resultados generados permanecen equivalentes (LED, nesting, clustering — verificado contra los mismos archivos reales).
- [x] No hay cambios en reglas de negocio.
- [x] El código está mejor organizado (2 archivos de ~2250 líneas → 12 módulos de 14-1300 líneas cada uno, por responsabilidad).
- [x] Los archivos eliminados tienen evidencia de que no son necesarios (ver sección 3).
- [x] Las pruebas disponibles se ejecutan satisfactoriamente (18/18).
- [x] El proyecto inicia en el entorno de pruebas — **verificado con el comando real** (`python app.py`), no solo con un import.
- [x] Existe este reporte detallado.
