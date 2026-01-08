import os
import json
import boto3
import logging
import pandas as pd
import numpy as np
import wandb

from io import BytesIO
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score
import joblib
import tempfile

logging.basicConfig(level=logging.INFO)


STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "datasets")
STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")

WANDB_PROJECT = "asmm_baseline"
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")
RUN_NAME = "tfidf_svm_baseline"

RANDOM_STATE = 96
REGISTERED_MODEL_NAME = "asmm_classifier"


def load_data():
    logging.info("Getting storage handle")

    s3 = boto3.client(
        "s3",
        endpoint_url=STORAGE_URL,
        aws_access_key_id=STORAGE_ACCESS_KEY,
        aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY
    )

    logging.info("Getting metadata of last processed dataset")

    processed_metadata_obj = s3.get_object(
        Bucket=DATA_BUCKET,
        Key="metadata/processed_tweets_latest.json"
    )

    processed_metadata = json.loads(
        processed_metadata_obj["Body"].read().decode("utf-8")
    )

    dataset_path = processed_metadata["path"]
    logging.info(f"Loading dataset from {dataset_path}")

    train_obj = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=f"{dataset_path}/train.parquet"
    )
    val_obj = s3.get_object(
        Bucket=DATA_BUCKET,
        Key=f"{dataset_path}/val.parquet"
    )

    train_df = pd.read_parquet(BytesIO(train_obj["Body"].read()))
    val_df = pd.read_parquet(BytesIO(val_obj["Body"].read()))

    logging.info("Train and validation data loaded")

    return train_df, val_df


def main():
    train_df, val_df = load_data()

    X_train = train_df["clean_text"]
    y_train = train_df["class"]
    X_val = val_df["clean_text"]
    y_val = val_df["class"]

    tfidf_params = {
        "max_features": 5000,
        "ngram_range": (1, 2)
    }

    linearsvc_params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    }

    cccv_params = {
        "method": "sigmoid",
        "cv": 5
    }

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        config={
            **tfidf_params,
            **linearsvc_params,
            **cccv_params,
            "confidence_threshold": 0.6,
            "model_type": "tfidf+svm+CCCV",
            "framework": "sklearn",
            "dataset": "Hate Speech and Offensive Language Dataset",
            "environment": "development"
        }
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(**linearsvc_params),
            **cccv_params
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)

    accuracy = balanced_accuracy_score(y_val, y_pred)

    confidence_threshold = wandb.config.confidence_threshold
    low_conf_mask = y_pred_proba.max(axis=1) < confidence_threshold
    low_conf_ratio = low_conf_mask.sum() / len(y_val)

    y_val_bin = np.where(y_val == 2, 1, 0)
    y_pred_bin = np.where(y_pred == 2, 1, 0)

    recall_bad = recall_score(y_val_bin, y_pred_bin, pos_label=0)
    f1 = f1_score(y_val, y_pred, average="weighted")

    wandb.log({
        "balanced_accuracy": accuracy,
        "low_conf_ratio": low_conf_ratio,
        "recall_bad_comments": recall_bad,
        "f1_score": f1
    })

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Low confidence ratio: {low_conf_ratio}")
    logging.info(f"Recall (bad comments): {recall_bad}")
    logging.info(f"F1 score: {f1}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.joblib")
        joblib.dump(pipeline, model_path)

        artifact = wandb.Artifact(
            name=REGISTERED_MODEL_NAME,
            type="model",
            description="TF-IDF + Calibrated Linear SVM classifier"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
