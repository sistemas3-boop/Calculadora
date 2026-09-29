FROM python:3.11-slim

# Dependencias del sistema:
#  - libgl1/libglib2.0-0: opencv y PyMuPDF
#  - libreoffice-draw: convierte .cdr (CorelDRAW) a PDF en modo headless
#    via el filtro libcdr, para poder leer medidas y piezas del diseño
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libreoffice-draw \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
