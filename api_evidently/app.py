from evidently import DataDefinition
from evidently import Dataset
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

import boto3

import os

import pandas as pd
import json
import numpy as np

from flask import Flask, jsonify, request
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
from flask import Response, g, request

STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "op-store")
REFERENCE_BUCKET = os.environ.get("REFERENCE_BUCKET", "datasets")
STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")

HOST = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_RUN_PORT", 5003))

s3 = boto3.client(
    "s3",
    endpoint_url=STORAGE_URL,
    aws_access_key_id=STORAGE_ACCESS_KEY,
    aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY
)


class DriftMonitor:
    def __init__(self, s3_client, reference_bucket, inference_bucket, ref_path, log_prefix, interval=300):
        self.s3 = s3_client
        self.ref_bucket = reference_bucket
        self.inf_bucket = inference_bucket
        self.ref_path = ref_path
        self.log_prefix = log_prefix
        self.interval = interval
        self._reference = None

    def _load_reference(self):
        if self._reference is None:
            obj = self.s3.get_object(Bucket=self.ref_bucket, Key=self.ref_path)
            reference_data = pd.read_csv(obj["Body"])
            self._reference = reference_data[["tweet", "class"]]
        return self._reference
    
    def _load_recent_logs(self):
        keys = self.s3.list_objects_v2(
            Bucket=self.inf_bucket, Prefix=self.log_prefix
        ).get("Contents", [])
        frames = []
        for obj_ref in sorted(keys, key=lambda x: x["LastModified"])[-10:]:
            obj = self.s3.get_object(Bucket=self.inf_bucket, Key=obj_ref["Key"])
            frames.extend(json.loads(obj["Body"].read().decode("utf-8")))
        inference_data = pd.DataFrame(frames)
        inference_data["tweet"] = inference_data["text"]
        inference_data["class"] = np.nan
        return inference_data[["tweet", "class"]]
    
    def _save_recent_report(self, report_html):
        s3.put_object(
            Bucket=self.inf_bucket,
            Key="reports/report.html",
            Body=report_html,
            ContentType="text/html"
        )

    def load_recent_report(self):
        obj = self.s3.get_object(Bucket=self.inf_bucket, Key="reports/report.html")
        return obj["Body"].read()

    def run_report(self):
        ref = self._load_reference()
        current = self._load_recent_logs()

        definition = DataDefinition(
           text_columns=["tweet"]
        )
        ref_data = Dataset.from_pandas(
            pd.DataFrame(ref["tweet"]),
            data_definition=definition
        )
        current_data = Dataset.from_pandas(
            pd.DataFrame(current["tweet"]),
            data_definition=definition
        )

        report = Report(metrics=[
            DataSummaryPreset(),
            DataDriftPreset()
        ])
        eval = report.run(reference_data=ref_data, current_data=current_data)
        # Option 3: Best approach — save to file and read it back
        eval.save_html("report.html")
        with open("report.html", "r") as f:
            html = f.read()

        self._save_recent_report(html)

        return html

drift_monitor = DriftMonitor(
        s3_client=s3,
        reference_bucket = REFERENCE_BUCKET,
        inference_bucket = DATA_BUCKET,
        ref_path="raw/v1/data.csv",
        log_prefix="inference-logs/",
        interval=300
    )

drift_monitor.run_report()

app = Flask(__name__)

app.config["API_TITLE"] = "Dynamic Table API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

api = Api(app)

public_blp = Blueprint(
    "report",
    "report",
    url_prefix="/api",
    description="Public api operations"
)

@public_blp.route("")
class ModelUsageResource(MethodView):

    @public_blp.response(200)
    def post(self):
        html = drift_monitor.run_report()
        return Response(html, mimetype="text/html")
    
    @public_blp.response(200)
    def get(self):
        html = drift_monitor.load_recent_report()        
        return Response(html, mimetype="text/html")

api.register_blueprint(public_blp)

if __name__ == "__main__":
    app.run(debug=True, host = HOST, port = PORT)