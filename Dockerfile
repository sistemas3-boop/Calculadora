FROM python:3.11-slim

# Dependencias del sistema:
#  - libgl1/libglib2.0-0: opencv y PyMuPDF
#  - libreoffice-draw: convierte .cdr (CorelDRAW) a PDF en modo headless
#    via el filtro libcdr, para poder leer medidas y piezas del diseño
#  - tesseract-ocr/-spa: lee cotas y medidas de imágenes/PDF sin texto
#    real (ver _find_tesseract_bin en app.py) — faltaba, así que en el
#    contenedor TESSERACT_AVAILABLE quedaba en False silenciosamente y
#    todo el análisis OCR de imágenes JPG/PNG no funcionaba en producción
#    aunque sí funcionara en Windows local (ahí si está instalado aparte).
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libreoffice-draw \
    tesseract-ocr \
    tesseract-ocr-spa \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
