FROM python:3.11-slim

# ── متغيرات البيئة لإجبار headless mode قبل أي import ──
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QT_QPA_PLATFORM=offscreen \
    DISPLAY="" \
    MPLBACKEND=Agg \
    OPENCV_IO_ENABLE_OPENEXR=0

# ── مكتبات النظام المطلوبة لـ OpenCV headless ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ── تثبيت المتطلبات ──
# الخطوة 1: ثبّت كل حاجة (ultralytics هيثبّت opencv-python)
# الخطوة 2: امسح opencv-python غير الـ headless
# الخطوة 3: ثبّت opencv-python-headless (مفيش Qt، مفيش libxcb)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless --force-reinstall

COPY . .

EXPOSE 8080

CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:8080", \
     "--workers", "1", \
     "--timeout", "300", \
     "--worker-class", "sync"]
