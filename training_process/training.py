import os
import boto3
import json
import pandas as pd
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import mlflow 
import mlflow.sklearn 
from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC 
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)

MLFLOW_URL = os.environ.get("MLFLOW_URL", "http://localhost:5000")

STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "datasets")
STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")



EXPERIMENT_NAME = "asmm_baseline"
RANDOM_STATE = 96
REGISTERED_MODEL_NAME = "asmm_classifier"

def load_data():
    logging.info("Gettting storage handle")

    s3 = boto3.client(
        "s3",
        endpoint_url=STORAGE_URL,
        aws_access_key_id=STORAGE_ACCESS_KEY,
        aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY
    )

    logging.info("Getting metadata of last processed dataset")

    processed_metadata_obj = s3.get_object(
        Bucket="datasets",
        Key="metadata/processed_tweets_latest.json"
    )
    
    processed_metadata = json.loads(processed_metadata_obj["Body"].read().decode("utf-8"))

    logging.info(f"Loading last processed dataset from {processed_metadata["path"]}")

    train_obj = s3.get_object(
        Bucket="datasets",
        Key=f"{processed_metadata["path"]}/train.parquet"
    )

    val_obj = s3.get_object(
        Bucket="datasets",
        Key=f"{processed_metadata["path"]}/val.parquet"
    )

    train_df = pd.read_parquet(BytesIO(train_obj['Body'].read()))    
    val_df = pd.read_parquet(BytesIO(val_obj['Body'].read()))

    logging.info("Last processed dataset train and val loaded")

    return train_df, val_df



def main():
    train_df, val_df = load_data()
    
    X_train = train_df["clean_text"]
    y_train = train_df["class"]
    X_val = val_df["clean_text"]
    y_val = val_df["class"]

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

    mlflow.set_tracking_uri(MLFLOW_URL)
    logging.info(f"Connected to MLFlow server")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
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
        mlflow.set_tag("environment","development")
        mlflow.set_tag("framework","sklearn")
        mlflow.set_tag("dataset", "Hate Speech and Offensive Language Dataset")

        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            name="tfidf+svm+CCCV",
            registered_model_name=REGISTERED_MODEL_NAME
        )

    

if __name__ == "__main__":
    main()