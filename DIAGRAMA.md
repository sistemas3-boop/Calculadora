# Diagrama completo — Calculadora SMV Lumicolor

> Abre este archivo en VS Code y presiona `Ctrl+Shift+V` para ver los diagramas renderizados.

---

## 1 · Flujo general de la aplicación

```mermaid
flowchart TD
    START(["Usuario abre /"])
    LOAD["Servidor carga times.json\nConstruye REGRESSIONS por área/subproceso"]
    IMG_Q{"¿Sube imagen?"}
    ANALYZE["POST /api/analyze"]
    NO_IMG["detectedHeight = null\ndetectedWidth  = null\ndetectedPieces = null"]
    CV2_Q{"¿OpenCV\ndisponible?"}
    OCR_FLOW["OCR Tesseract: lee medidas\nContornos: cuenta piezas"]
    EMPTY["width_cm = null\nheight_cm = null\npieces = 0"]
    CALC_ANCHO["anchoPorPieza = width_cm ÷ max(pieces, 1)\ndetectedHeight = height_cm\ndetectedWidth  = anchoPorPieza\ndetectedPieces = pieces"]
    PANEL["Panel de análisis:\nUsuario revisa y ajusta\nAlto / Ancho / Piezas"]
    APPLY["applyAnalysis:\nRecalcula badges de tiempo\nAsigna qty = pieces a filas marcadas"]
    SELECT["Usuario marca checkboxes\nde subprocesos por área"]
    CGR["computeGroupResult por cada subproceso\nActualiza badge en tiempo real"]
    PARAMS["Configura parámetros:\nHoras/día · Empaque · Secado · Colchón"]
    SUBMIT["Submit: inyecta hidden inputs\nareas, subprocesses, groups,\nquantities, widths, heights"]
    BACKEND_CALC["Backend calcula tiempo por fila\n(ver diagrama 3)"]
    ERR_Q{"¿Filas\ncon error?"}
    FLASH["flash + redirect /\nMuestra errores al usuario"]
    SAVE["Guarda Calculation en BD\nRedirect /result/id"]
    RESULT(["Página /result/{id}"])

    START --> LOAD
    LOAD --> IMG_Q
    IMG_Q -- Sí --> ANALYZE
    IMG_Q -- No --> NO_IMG
    ANALYZE --> CV2_Q
    CV2_Q -- Sí --> OCR_FLOW
    CV2_Q -- No --> EMPTY
    OCR_FLOW --> CALC_ANCHO
    EMPTY --> CALC_ANCHO
    CALC_ANCHO --> PANEL
    PANEL --> APPLY
    APPLY --> SELECT
    NO_IMG --> SELECT
    SELECT --> CGR
    CGR --> PARAMS
    PARAMS --> SUBMIT
    SUBMIT --> BACKEND_CALC
    BACKEND_CALC --> ERR_Q
    ERR_Q -- Sí --> FLASH
    ERR_Q -- No --> SAVE
    SAVE --> RESULT
```

---

## 2 · Análisis de imagen — /api/analyze (detallado)

```mermaid
flowchart TD
    subgraph CV2P ["Ruta OpenCV  (CV2_AVAILABLE = True)"]
        B1["cv2.imread\nfallback: PIL si cv2 falla"]
        B2["Convierte a escala de grises"]
        B3["_split_content_and_scale:\n• _ink_components separa blobs\n• CONTENIDO (letras/logo, macizo)\n  vs ANOTACIONES (cotas/flechas,\n  chicas o huecas/delgadas)\n• OCR cota de ANCHO arriba del\n  contenido, cota de ALTO a la\n  izquierda (rotada 90°)\n• Arma escala real mm/px\n  (con el ANCHO, ver docstring)"]
        B4["_cluster_boxes_by_height_px:\nagrupa las cajas de CONTENIDO\npor su alto real (mm/px) —\nsepara logo vs letras, detecta\nmás de un grupo/tamaño"]
        B5["Dibuja bounding boxes\ny número en imagen anotada\n→ guarda archivo *_analyzed.jpg"]
        B1 --> B2 --> B3 --> B4 --> B5
    end

    subgraph RESP ["Respuesta JSON"]
        C1["width_cm · height_cm · pieces\nletter_groups\noriginal_url · annotated_url · image_url\ncv2_available · tesseract_available\npymupdf_available · libreoffice_available\nimg_read · file_saved · filename"]
    end

    B5 --> RESP

    subgraph JS_POST ["JS tras recibir respuesta"]
        D1["anchoPorPieza = width_cm ÷ max(pieces, 1)"]
        D2["Rellena campos del panel:\nval-height = height_cm\nval-width  = anchoPorPieza\nval-pieces = pieces"]
        D3{"img_read = false?"}
        D4["Toast: No se pudo leer la imagen"]
        D5{"sin height NI\nwidth NI pieces?"}
        D6["Toast: Sin datos — ingresa manualmente"]
        D7{"solo pieces\nsin medidas?"}
        D8["Toast: Piezas OK · medidas manuales"]
        D9{"solo medidas\nsin pieces?"}
        D10["Toast: Medidas OK · piezas manuales"]
        D11["Toast: Todo detectado\nancho×alto cm/pieza · N pieza(s)"]
        D12["Llama a applyAnalysis automáticamente"]
        D1 --> D2 --> D3
        D3 -- Sí --> D4
        D3 -- No --> D5
        D5 -- Sí --> D6
        D5 -- No --> D7
        D7 -- Sí --> D8
        D7 -- No --> D9
        D9 -- Sí --> D10
        D9 -- No --> D11
        D4 --> D12
        D6 --> D12
        D8 --> D12
        D10 --> D12
        D11 --> D12
    end

    RESP --> JS_POST
```

---

## 3 · Cálculo de tiempo por subproceso  (Frontend JS espeja Backend Python)

> Ambos usan exactamente la misma lógica de prioridad.

```mermaid
flowchart TD
    IN["Entrada:\narea · subproceso · group · height · width"]

    E1{"ENSAMBLE\nsin flange\n+ width?"}
    F1["mins = (0.2545 + 0.01067 × width) × 60\ngroup = Formula(widthCm)\nextrapolado = false"]

    E2{"height AND\nregresión existe AND\nheight > maxX del catálogo?"}
    F2["mins = max(a + b × height, 0.1)\ngroup = Extrap(heightCm)\nextrapolado = true\nBadge AMARILLO"]

    E3["Busca en catálogo:\ngrupo cuya descripción tiene\naltura más cercana a height\n(si height = null → primer grupo)"]
    E4{"¿Entrada\nencontrada?"}
    F3["mins = catalog[area][sub][group].tiempo\ngroup = clave del catálogo\nextrapolado = false\nBadge NORMAL"]

    E5{"height AND\nregresión existe?"}
    F4["mins = max(a + b × height, 0.1)\ngroup = Formula(heightCm)\nextrapolado = true\nBadge AMARILLO"]

    F5["JS: mins = 0, badge '0 min'\nPython: ValueError → fila con error"]

    IN --> E1
    E1 -- Sí --> F1
    E1 -- No --> E2
    E2 -- Sí --> F2
    E2 -- No --> E3
    E3 --> E4
    E4 -- Sí --> F3
    E4 -- No --> E5
    E5 -- Sí --> F4
    E5 -- No --> F5
```

---

## 4 · Construcción de regresiones (build_regressions)

> Se ejecuta al iniciar la app y en cada request GET.  
> Solo genera regresión si hay 3+ puntos con pendiente positiva.

```mermaid
flowchart TD
    R0["Para cada (area, subproceso) en catalog"]
    R1{"¿Es ENSAMBLE\nsin flange?"}
    R_SKIP["Omitir — tiene fórmula propia"]
    R2["Para cada grupo del subproceso:\nextrae x = promedio de alturas de la descripción\n  ejemplo: '2×30-40 cm'  →  x = 35\nextrae y = tiempo (min)"]
    R3{"n ≥ 3\npares (x, y)?"}
    R4["Regresión lineal simple:\nb = (n·Σxy − Σx·Σy) ÷ (n·Σx² − (Σx)²)\na = (Σy − b·Σx) ÷ n"]
    R5{"b > 0\n(pendiente positiva)?"}
    R6["Guarda en REGRESSIONS:\n(area, sub) → { a, b, minX, maxX }"]

    R0 --> R1
    R1 -- Sí --> R_SKIP
    R1 -- No --> R2
    R2 --> R3
    R3 -- No, menos de 3 puntos --> R_SKIP
    R3 -- Sí --> R4
    R4 --> R5
    R5 -- No --> R_SKIP
    R5 -- Sí --> R6
```

---

## 5 · Fórmulas finales y cálculo de resultados

```mermaid
flowchart TD
    ROWS["Filas procesadas:\ncada fila tiene mins_por_pieza y qty"]
    PROD["production_minutes\n= Σ (mins_por_pieza × qty)\n+ secado_minutes\n+ packaging_minutes"]
    HRS["total_hours\n= production_minutes ÷ 60"]
    CUSH["total_hours_cushion\n= total_hours × (1 + cushion_pct ÷ 100)"]
    DAYS["days          = total_hours         ÷ hours_per_day\ndays_cushion  = total_hours_cushion  ÷ hours_per_day"]
    DB["Guarda Calculation en BD:\n• name · user_id · created_at\n• rows_json\n• params_json\n• results_json\n• image_url"]
    SHOW["Página /result/{id}\nMuestra tabla de tiempos\ny días estimados con y sin colchón"]

    ROWS --> PROD --> HRS --> CUSH --> DAYS --> DB --> SHOW
```

---

## 6 · Tabla de parámetros y valores por defecto

| Parámetro | Default | Dónde se usa |
|---|---|---|
| `hours_per_day` | 9 h | Divide `total_hours` para obtener días |
| `packaging_minutes` | 40 min | Se suma a `production_minutes` |
| `secado_minutes` | 120 min | Se suma a `production_minutes` |
| `cushion_pct` | 10 % | Multiplica `total_hours` por `1 + cushion/100` |
| `anchoPorPieza` | `width_cm ÷ max(pieces, 1)` | Solo para ENSAMBLE sin flange |

---

## 7 · Prioridad de cálculo (resumen ejecutivo)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PRIORIDAD  │  CONDICIÓN                               │  RESULTADO         │
├─────────────┼──────────────────────────────────────────┼────────────────────┤
│      1      │  ENSAMBLE sin flange + ancho disponible  │  Fórmula fija      │
│             │  mins = (0.2545 + 0.01067 × ancho) × 60 │                    │
├─────────────┼──────────────────────────────────────────┼────────────────────┤
│      2      │  Alto > máximo del catálogo              │  Regresión lineal  │
│             │  + regresión disponible para ese sub      │  (extrapolación)   │
├─────────────┼──────────────────────────────────────────┼────────────────────┤
│      3      │  Grupo encontrado en catálogo            │  Tiempo del        │
│             │  (altura dentro del rango conocido)      │  catálogo exacto   │
├─────────────┼──────────────────────────────────────────┼────────────────────┤
│      4      │  Grupo NO encontrado en catálogo         │  Regresión lineal  │
│             │  + alto y regresión disponibles          │  (fallback)        │
├─────────────┼──────────────────────────────────────────┼────────────────────┤
│      5      │  Sin catálogo, sin regresión, sin alto   │  Error / 0 min     │
└─────────────┴──────────────────────────────────────────┴────────────────────┘
```

---

## 8 · Fórmula de Ensamble sin flange — detalle

| Concepto | Valor |
|---|---|
| Fórmula | `mins = (0.2545 + 0.01067 × ancho_cm) × 60` |
| Origen | Regresión histórica específica del área ENSAMBLE |
| Parámetro | `widthCm` = ancho **por pieza** (no el total del letrero) |
| Aplica | Solo al subproceso exacto `"Ensamble sin flange"` |
| Excluida de `build_regressions` | Sí — se omite para no generar una regresión redundante |

---

## 9 · Flujo del envío del formulario (Submit JS)

```mermaid
flowchart TD
    S1["Usuario hace clic en Calcular"]
    S2{"¿Al menos\n1 subproceso\nmarcado?"}
    S3["Toast WARN + shake al área-grid\nNo envía el formulario"]
    S4["Sincroniza qty de cada\nsub-item desde el input visible"]
    S5["Elimina todos los .auto-hidden\nanteriores del formulario"]
    S6["Para cada item en selectedItems:\ninyecta inputs hidden:\n  areas[]\n  subprocesses[]\n  groups[]\n  quantities[]\n  widths[]  (vacío si no es ENSAMBLE sin flange)\n  heights[] (vacío si no hay medida)"]
    S7["Deshabilita botón Submit\nMuestra spinner"]
    S8["POST / con form multipart"]

    S1 --> S2
    S2 -- No --> S3
    S2 -- Sí --> S4 --> S5 --> S6 --> S7 --> S8
```
