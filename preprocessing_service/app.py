from flask import Flask
from flask.views import MethodView
from flask_smorest import Api, Blueprint, abort
from marshmallow import Schema, fields
import logging
import os

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

HOST = os.environ.get("FLASK_RUN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_RUN_PORT", 5001))

# OpenAPI / Swagger configuration
app.config["API_TITLE"] = "Dynamic Table API"
app.config["API_VERSION"] = "v1"
app.config["OPENAPI_VERSION"] = "3.0.3"
app.config["OPENAPI_URL_PREFIX"] = "/"
app.config["OPENAPI_SWAGGER_UI_PATH"] = "/swagger"
app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"

import re
USER_RE = re.compile(r"@\w+") # mentions
LINK_RE = re.compile(r"http\S+") # URLs
HTML_ENTITY_RE = re.compile(r"&#\w+;|&\w+;") # html tags
HASHTAG_RE = re.compile(r"#\w+") # hashtags
CLEAN_RE = re.compile(r"[^a-zA-Z<>|\s]") # punctuation except for <>
SPACE_RE = re.compile(r"\s+")


class TableSchema(Schema):
    name = fields.Str(required=True)
    sort_key = fields.Str(required=True)
    partition_key = fields.Str(required=True)
    data_schema = fields.Dict(required=True)

TABLES_METADATA = {}

public_blp = Blueprint(
    "preprocessing",
    "preprocessing",
    url_prefix="/",
    description="Preprocessing operations"
)

def clean_text(text):
    text = USER_RE.sub("<USER>", text)
    text = LINK_RE.sub("<LINK>", text)
    text = HTML_ENTITY_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = CLEAN_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text

api = Api(app)

class TextSchema(Schema):
    text = fields.Str(required=True)

@public_blp.route("speed")
class SpeedProcessingResource(MethodView):

    @public_blp.arguments(TextSchema)
    @public_blp.response(201, TextSchema)
    def post(self, data):
        """Register a new table"""
        text_str = data["text"]
        data["text"] = clean_text(text_str)
        return data
    

class BatchTextSchema(Schema):
    texts = fields.List(fields.Str(), required=True)

@public_blp.route("batch")
class SpeedProcessingResource(MethodView):

    @public_blp.arguments(BatchTextSchema)
    @public_blp.response(201, BatchTextSchema)
    def post(self, data):
        """Register a new table"""
        texts = data["texts"]
        data["texts"] = [clean_text(t) for t in texts]
        return data


api.register_blueprint(public_blp)
           
if __name__ == "__main__":
    app.run(debug=True, host = HOST, port = PORT)
