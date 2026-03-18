# Startup Guide

Clone this repository.

Set up a .env file in the project root.

[Example of .env](./.env.txt)

Have docker started and from the project root:

```
docker compose up --build -d
```

# Infrastructure Entry Points

You can access 

## MinIO
```
http://localhost:{$MINIO_PORT}
```

# Inference API

Swagger UI

```
http://localhost:${MODELAPI_PORT}/swagger
```

Main endpoint 

```
POST /api
```

Example Request

```
curl -X POST http://localhost:8000/api \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a sample message"
  }'
```

Responce

```
{
  "confidence_score": 0.5631636113229244,
  "prediction": "offensive language",
  "text": "This is a sample message"
}
```


## Evidently

Evidently checks existance of data drift.

```
http://localhost:${EVIDENTLY_PORT}/api
```

GET to get html of last report

POST to recalculate drift and get html of that report


# Shut down

```
docker compose down
```

To shut down with data delition.

```
docker compose down -v
```