FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt --no-cache-dir --prefix=/install

FROM python:3.11-slim

WORKDIR /app

COPY  --from=builder /install /usr/local
COPY app/ ./app/
COPY training/ ./training/
COPY mlruns/ ./mlruns/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000"]