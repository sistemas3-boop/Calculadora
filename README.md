# Calculadora de tiempo de producción (Flask)

Este proyecto es una **aplicación Flask** que ayuda a calcular el tiempo estimado de producción de un letrero en base a una **tabla de tiempos estándar** (Excel) y una **foto del proyecto**.

## ✅ Cómo funciona
1. Subes un **Excel** con los tiempos estándar por letra.
2. (Opcional) Subes una **foto** del letrero.
3. La aplicación intenta extraer el texto de la imagen (OCR) y calcula el tiempo total.
4. Se muestra el tiempo en **horas**, **días laborales (9h = 1 día)** y se agrega **un 10% de colchón**.

---

## 🛠 Requisitos
- Python 3.11+ (recomendado)
- Libreta de Python: `Flask`, `pandas`, `openpyxl`, `pillow`, `opencv-python`, `pytesseract`
- **Tesseract OCR** instalado en el sistema para que la extracción automática de texto funcione.

### 1) Instalar dependencias de Python (desde la carpeta del proyecto)
```sh
python -m pip install -r requirements.txt
```

### 2) Instalar Tesseract (solo si quieres usar OCR)
- **Windows**: descarga e instala desde https://tesseract-ocr.github.io/
- **Linux** (Debian/Ubuntu): `sudo apt install tesseract-ocr`

---

## 🚀 Ejecutar la app
Desde la carpeta del proyecto:

```sh
python app.py
```

Luego abre en el navegador:

```
http://127.0.0.1:5000
```

---

## 📄 Formato del Excel de tiempos
El Excel debe incluir al menos estas columnas (sin importar el orden):
- `letter` (letra / texto / símbolo)
- `time` (horas / minutos)

Ejemplo:

| letter | time |
|--------|------|
| A      | 0.5  |
| B      | 0.6  |
| C      | 0.55 |

---

## 🗃️ Uso sin Excel (catálogo integrado)
Si no quieres subir un archivo, la aplicación incluye un catálogo base de tiempos (el que compartiste en la imagen) en:

- `data/times.json`

Puedes seleccionar:
- Área productiva (ej. CORTE)
- Subproceso / material (ej. Acrílico blanco)
- Grupo / código (ej. CTE-A1)

El cálculo usará el tiempo (en minutos) definido para ese grupo.

Si quieres ajustar los tiempos, edita `data/times.json` y vuelve a ejecutar la app.

Puedes agregar columnas adicionales como `size` / `tamaño` para indicar el tamaño del tipo de letra.

---

## 🧠 ¿Qué pasa si no funciona el OCR?
Si la extracción automática no detecta correctamente el texto, en la página principal hay un campo donde puedes ingresar manualmente el texto del letrero.

---

## 📌 Nota
El cálculo es una estimación basada en la tabla de tiempos que proporciones. Ajusta los valores del Excel según tus estándares de producción.

### 📦 Empaque
La aplicación agrega automáticamente **40 minutos de empaque** al tiempo final. Puedes ajustar ese valor en la aplicación.
