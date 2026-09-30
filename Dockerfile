FROM python:3.12-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Railway sets $PORT automatically; DB_PATH should point at a mounted
# persistent volume, e.g. /data/app.db, configured via a Railway Volume
# attached to this service and the DB_PATH environment variable.
ENV DB_PATH=/data/app.db

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
