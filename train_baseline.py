# Databricks notebook source
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC 
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

import mlflow 
import mlflow.sklearn 

import logging

import pickle

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

logging.basicConfig(level=logging.INFO)

# COMMAND ----------

EXPERIMENT_NAME = "/Users/kopcov.vitaly@lll.kpi.ua/asmm_Lab3"
RANDOM_STATE = 96
MODEL_NAME = "asmm_classifier"

CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
train_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.train"
val_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.val"

# COMMAND ----------

train_df = spark.read.table(train_table).toPandas()
val_df = spark.read.table(val_table).toPandas()

X_train = train_df["processed_tweet"]
y_train = train_df["class"]
X_val = val_df["processed_tweet"]
y_val = val_df["class"]

# COMMAND ----------

tfidf_params = {
    "max_features": 5000,
    "ngram_range": (1,2)
}

linearsvc_params = {
    "class_weight": "balanced",
    "random_state" : RANDOM_STATE
}

CCCV_params = {
    "method": "sigmoid",
    "cv": 5
}

# COMMAND ----------

from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

input_schema = Schema([ColSpec("string", "comment")])

output_schema = Schema([
    ColSpec("double", "hate speech"),
    ColSpec("double", "offensive language"),
    ColSpec("double", "neither"),
])

signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# COMMAND ----------

import sys
import os

class SKLearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)
        sys.path.insert(0, os.path.dirname(context.artifacts["preprocessor"]))
        from preprocessing import TextPreprocessor
        self.preprocessor = TextPreprocessor()
        self.labels = ["hate speech", "offensive language", "neither"]

    def predict(self, context, model_input):
        text = self.preprocessor.transform(model_input["comment"].tolist())
        probs = self.model.predict_proba(text)
        return [{label: p for label, p in zip(self.labels, prob)} for prob in probs]

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="tfidf_svm") as run:
    mlflow.log_params(tfidf_params | linearsvc_params | CCCV_params)

    pipeline = Pipeline([ 
        ("tfidf", TfidfVectorizer(**tfidf_params)), 
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(**linearsvc_params), **CCCV_params)) ])
    
    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_val)
    y_pred = pipeline.predict(X_val)

    accuracy  = balanced_accuracy_score(y_val, y_pred)
    mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Accuracy: {accuracy}")

    confidence_threshold = 0.6 
    low_conf_mask = y_pred_proba.max(axis=1) < confidence_threshold 
    low_conf_count = low_conf_mask.sum() 
    low_conf_ratio = low_conf_count / len(y_val)
    mlflow.log_metric("low_conf_ratio", low_conf_ratio)
    logging.info(f"low_conf_ratio: {low_conf_ratio}")

    y_val_bin = np.where(y_val == 2, 1, 0)
    y_pred_bin = np.where(y_pred == 2, 1, 0)

    recall = recall_score(y_val_bin, y_pred_bin, pos_label=0)
    mlflow.log_metric("recall_bad_comments", recall)
    logging.info(f"recall_bad_comments: {recall}")

    f1 = f1_score(y_val, y_pred, average='weighted')
    mlflow.log_metric("f1_score", f1)
    logging.info(f"f1_score: {f1}")

    mlflow.set_tag("low_confidence_threshold", confidence_threshold)
    mlflow.set_tag("Training Info", "TF‑IDF vectorizer with a calibrated Linear SVM classifier")
    mlflow.set_tag("model_type", "tfidf+svm_CCCV")
    mlflow.set_tag("environment","stage")
    mlflow.set_tag("framework","sklearn")
    mlflow.set_tag("problem", "Hate Speech and Offensive Language Dataset")
    mlflow.set_tag("candidate_type", "challenger")

    with open("/tmp/sklearn_model.pkl", "wb") as file:
        pickle.dump(pipeline, file)
    
    artifacts = {"model": "/tmp/sklearn_model.pkl",
                 "preprocessor": "./preprocessing.py"}

    mlflow.pyfunc.log_model(
        python_model=SKLearnWrapper(),
        artifacts=artifacts,
        pip_requirements=["scikit-learn"],
        signature=signature,
        registered_model_name=MODEL_NAME,
        input_example=pd.DataFrame({"comment": [""]})
    )
    
    history = spark.sql(f"DESCRIBE HISTORY {CATALOG_NAME}.{SCHEMA_NAME}.data")
    latest_version = history.first()["version"]
    
    mlflow.log_param("source_table", f"{CATALOG_NAME}.{SCHEMA_NAME}.data")
    mlflow.log_param("source_table_version", latest_version)
