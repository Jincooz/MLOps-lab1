from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
from flask import Response, g, request
from marshmallow import Schema, fields, validate, INCLUDE
import requests
import logging
import os
import boto3
import time
import json

from metrics import REQUEST_LATENCY, REQUESTS_TOTAL, REQUEST_ERRORS, NODE_UP
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)

STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "op-store")
STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN", "")
DATABRICKS_ENDPOINT = os.environ.get("DATABRICKS_ENDPOINT", "")

HOST = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_RUN_PORT", 5002))


s3 = boto3.client(
    "s3",
    endpoint_url=STORAGE_URL,
    aws_access_key_id=STORAGE_ACCESS_KEY,
    aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY
)


PREDICTION_MAP = [
    "hate speech",
    "offensive language",
    "neither"
]

class PseudoQueue:
    def __init__(self, s3_client, bucket, file_path, max_size = 10):
        self.array = []
        self.s3 = s3_client
        self.bucket = bucket
        self.file_path = file_path
        self.max_size = max_size
    
    def append(self, new_value):
        value = dict(new_value)
        value["created_at"] = time.time()
        self.array.append(value)
        if len(self.array) >= self.max_size:
            self.flush_to_s3()
    
    def flush_to_s3(self):
        logging.info("Flushing queue into storage")
        json_bytes = json.dumps(self.array).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.file_path}/ModelServiceDump_{time.strftime('%Y%m%d_%H%M%S')}.json", Body=json_bytes)
        self.array = []


queue = PseudoQueue(s3, DATA_BUCKET, "inference-logs")

app = Flask(__name__)


# OpenAPI / Swagger configuration
app.config["API_TITLE"] = "Dynamic Table API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)

@app.before_request
def before_request():
    g.start_time = time.time()


@app.after_request
def after_request(response):
    duration = time.time() - g.start_time
    
    REQUEST_LATENCY.labels(
        "coordinator",
        request.endpoint or "unknown",
    ).observe(duration)

    REQUESTS_TOTAL.labels(
        "coordinator",
        request.endpoint or "unknown"
    ).inc()

    return response

@app.errorhandler(Exception)
def handle_error(e):
    REQUEST_ERRORS.labels(
        "coordinator",
        type(e).__name__
    ).inc()
    raise e

public_blp = Blueprint(
    "api",
    "api",
    url_prefix="/api",
    description="Public api operations"
)

class TextSchema(Schema):
    text = fields.Str(required=True)

@public_blp.route("")
class ModelUsageResource(MethodView):

    @public_blp.arguments(TextSchema)
    @public_blp.response(200)
    def post(self, text_json):
        """Use model"""
        response = requests.post(
            DATABRICKS_ENDPOINT,
            headers={"Authorization": f"Bearer {DATABRICKS_TOKEN}"},
            json={"dataframe_records": [{"comment": text_json["text"]}]}
        )

        if not response.ok:
            logging.error(f"Responce from Databricks Model Serving {response.status_code}")
            abort(404, "Service is not accesible")
        pred_class = response.json()["predictions"][0]
        prediciton = max(pred_class, key=lambda i: pred_class[i])
        result = {
            "text" : text_json["text"],
            "prediction" : prediciton,
            "confidence_score" : float(pred_class[prediciton])
        }
        queue.append(result)
        return result

api.register_blueprint(public_blp)

private_blp = Blueprint(
    "internal",
    "internal",
    url_prefix="/internal",
    description="Private api operations"
)
    
@private_blp.route("health")
class HealethResource(MethodView):
    @private_blp.response(200)
    def get(self):
        """Health check"""
        responce = {
            "status": "healthy",
        }
        return responce
    
@private_blp.route("metrics")
class MetricsAPI(MethodView):
    @private_blp.response(200)
    def get(self):
        return Response(
            generate_latest(),
            mimetype=CONTENT_TYPE_LATEST
        )
    
api.register_blueprint(private_blp)
           
if __name__ == "__main__":
    NODE_UP.labels("coordinator", None).set(1)
    app.run(debug=True, host = HOST, port = PORT)
