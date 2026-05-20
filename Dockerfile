FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./
RUN pip install --upgrade pip \
    && pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-runtime.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "analysis.main:app", "--host", "0.0.0.0", "--port", "8000"]
