FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5002

EXPOSE 5002

CMD ["python", "app.py"]
