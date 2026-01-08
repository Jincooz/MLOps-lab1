from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
from marshmallow import Schema, fields, validate, INCLUDE
import requests
import logging
import os
import mlflow
import boto3
import numpy as np
import time
import json

logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.environ.get("MODEL_NAME", "asmm_classifier")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "latest")

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")

STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "op-store")
STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")

PREPROCESSING_URL = os.environ.get("PREPROCESSING_URL", "http://localhost:5001")

HOST = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_RUN_PORT", 5002))

model = None

def load_model(model_name = MODEL_NAME, model_version = MODEL_VERSION):
    global model
    logging.info(f"Seting mlflow configs by {MLFLOW_URL}")
    mlflow.set_tracking_uri(MLFLOW_URL)
    model_uri = f"models:/{model_name}/{model_version}"
    logging.info(f"Trying to fetch {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)

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

class TableSchema(Schema):
    name = fields.Str(required=True)
    sort_key = fields.Str(required=True)
    partition_key = fields.Str(required=True)
    data_schema = fields.Dict(required=True)

TABLES_METADATA = {}

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
        """Register a new table"""
        responce = requests.post(f"{PREPROCESSING_URL}/speed", json = text_json)
        if not responce.ok:
            logging.error(f"Responce from preprocesing service is {responce.status_code}")
            abort(404, "Service is not accesible")
        prediction = model.predict_proba([responce.json()["text"]])
        pred_class = int(np.argmax(prediction))
        result = {
            "text" : text_json["text"],
            "prediction_index" : pred_class,
            "prediction" : PREDICTION_MAP[pred_class],
            "confidence_score" : float(prediction[0][pred_class])
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

class VersionSchema(Schema):
    model_name = fields.Str(required=True)
    model_version = fields.Str(required=False)

@private_blp.route("model")
class ModelUsageResource(MethodView):

    @private_blp.arguments(VersionSchema)
    @private_blp.response(200)
    def post(self, version_info):
        """Load a model"""
        if "model_version" not in version_info:
            version_info["model_version"] = "latest"
        try:
            load_model(version_info["model_name"], version_info["model_version"])
        except Exception as e:
            logging.error(f"Error loading model {version_info['model_name']} with version {version_info['model_version']}. Error message {str(e)}")
            abort(404,"Failed to load model")
        global MODEL_NAME, MODEL_VERSION
        MODEL_NAME = version_info["model_name"]
        MODEL_VERSION = version_info["model_version"]
        return version_info
    
@private_blp.route("health")
class HealethResource(MethodView):
    @private_blp.response(200)
    def get(self):
        """Health check"""
        responce = {
            "status": "healthy",
            "model" : "Not loaded" if model is None else f"Loaded {MODEL_NAME}/{MODEL_VERSION}"
        }
        return responce

api.register_blueprint(private_blp)
           
if __name__ == "__main__":
    load_model()
    app.run(debug=True, host = HOST, port = PORT)
