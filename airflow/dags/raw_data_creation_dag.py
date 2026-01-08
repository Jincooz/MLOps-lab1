from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import pandas as pd
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

STORAGE_URL = os.environ.get("STORAGE_URL", "http://localhost:9000")

STORAGE_ACCESS_KEY = os.environ.get("STORAGE_USER", "minioadmin")
STORAGE_SECRET_ACCESS_KEY = os.environ.get("STORAGE_PASSWORD", "minioadmin")

OP_BUCKET = os.environ.get("OPERATION_BUCKET", "op-store")
DATA_BUCKET = os.environ.get("DATASET_BUCKET", "datasets")
PREPROCESSING_URL = os.environ.get("PREPROCESSING_URL", "http://localhost:5001")

METADATA_PREFIX = "metadata/"
RAW_PREFIX = "raw/"

def get_s3():
    return boto3.client(
        "s3",
        endpoint_url=STORAGE_URL,
        aws_access_key_id=STORAGE_ACCESS_KEY,
        aws_secret_access_key=STORAGE_SECRET_ACCESS_KEY
    )

# ------------------------
# UTIL
# ------------------------

def read_json(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def write_json(s3, bucket, key, payload):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload),
        ContentType="application/json"
    )

def list_raw_versions(s3):
    resp = s3.list_objects_v2(
        Bucket=DATA_BUCKET,
        Prefix=RAW_PREFIX,
        Delimiter="/"
    )
    return [
        p["Prefix"].replace(RAW_PREFIX, "").rstrip("/")
        for p in resp.get("CommonPrefixes", [])
    ]

def read_csv(s3, bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])

def write_csv(s3, bucket, key, df):
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=df.to_csv(index=False),
        ContentType="text/csv"
    )

# ------------------------
# TASKS
# ------------------------

def get_raw_version():
    s3 = get_s3()

    today = datetime.now().strftime("%Y-%m-%d")

    try:
        lock = read_json( 
            s3,
            DATA_BUCKET,
            f"{METADATA_PREFIX}raw_version_in_progress.json"
        )
        return lock["version"]

    except s3.exceptions.NoSuchKey:
        pass

    latest = None
    try:
        latest = read_json(
            s3,
            DATA_BUCKET,
            f"{METADATA_PREFIX}raw_tweets_latest.json"
        )["version"]
    except s3.exceptions.NoSuchKey:
        pass

    if latest != today:
        version = today
    else:
        existing = list_raw_versions(s3)
        suffixes = [
            int(v.split("_v")[1])
            for v in existing
            if v.startswith(today + "_v")
        ]
        next_v = max(suffixes, default=0) + 1
        version = f"{today}_v{next_v}"

    write_json(s3, DATA_BUCKET, 
        key = f"{METADATA_PREFIX}raw_version_in_progress.json", 
        payload= {
            "version": version,
            "created_at": datetime.now().isoformat()
        })

    return version
from datetime import timezone

def build_raw_model_logs():
    s3 = get_s3()

    lock = read_json(
        s3,
        DATA_BUCKET,
        f"{METADATA_PREFIX}raw_version_in_progress.json"
    )

    version = lock["version"]
    run_created_at = datetime.fromisoformat(lock["created_at"]).replace(tzinfo=timezone.utc)

    cursor_ts = datetime.fromisoformat(
        read_json(s3, DATA_BUCKET, f"{METADATA_PREFIX}cursor.json")["last_ingested_at"]
    ).replace(tzinfo=timezone.utc)


    resp = s3.list_objects_v2(
        Bucket=OP_BUCKET,
        Prefix="inference-logs/"
    )

    all_rows = []

    for obj in resp.get("Contents", []):
        if not obj["Key"].endswith(".json"):
            continue

        last_modified = obj["LastModified"]
        if not (cursor_ts < last_modified <= run_created_at):
            continue
        
        data = read_json(s3, OP_BUCKET, obj["Key"])
        all_rows.extend(data)

    if not all_rows:
        logging.warning("No model inference logs found")

    raw_key = f"{RAW_PREFIX}{version}/model_logs.json"

    write_json(s3, DATA_BUCKET, raw_key, all_rows)

    return raw_key

def build_raw_moderator_logs():
    s3 = get_s3()

    lock = read_json(
        s3,
        DATA_BUCKET,
        f"{METADATA_PREFIX}raw_version_in_progress.json"
    )

    version = lock["version"]
    run_created_at = datetime.fromisoformat(lock["created_at"]).replace(tzinfo=timezone.utc)

    cursor_ts = datetime.fromisoformat(
        read_json(s3, DATA_BUCKET, f"{METADATA_PREFIX}cursor.json")["last_ingested_at"]
    ).replace(tzinfo=timezone.utc)

    resp = s3.list_objects_v2(
        Bucket=OP_BUCKET,
        Prefix="moderators_logs/"
    )

    all_rows = []

    for obj in resp.get("Contents", []):
        if not obj["Key"].endswith(".json"):
            continue

        last_modified = obj["LastModified"]
        if not (cursor_ts < last_modified <= run_created_at):
            continue

        data = read_json(s3, OP_BUCKET, obj["Key"])
        all_rows.extend(data)

    raw_key = f"{RAW_PREFIX}{version}/moderator_logs.json"

    if not all_rows:
        logging.warning("No moderator logs found")

    write_json(s3, DATA_BUCKET, raw_key, all_rows)

    return raw_key

def build_raw_csv_and_metadata():
    s3 = get_s3()

    lock = read_json(
        s3,
        DATA_BUCKET,
        f"{METADATA_PREFIX}raw_version_in_progress.json"
    )
    version = lock["version"]
    run_created_at = lock["created_at"]

    raw_dir = f"{RAW_PREFIX}{version}/"
    csv_key = f"{raw_dir}data.csv"

    df = None

    try:
        latest = read_json(
            s3,
            DATA_BUCKET,
            f"{METADATA_PREFIX}raw_tweets_latest.json"
        )
        prev_csv_key = f"{latest['path']}/data.csv"
        df = read_csv(s3, DATA_BUCKET, prev_csv_key)
    except s3.exceptions.NoSuchKey:
        df = pd.DataFrame(columns=[
            "count",
            "hate_speech",
            "offensive_language",
            "neither",
            "class",
            "tweet",
            "created_at"
        ])

    if "created_at" not in df.columns:
        df["created_at"] = 0

    model_logs = read_json(
        s3,
        DATA_BUCKET,
        f"{raw_dir}model_logs.json"
    )
    #model log is {"text": "...", "prediction_index": 0, "prediction": "..,", "confidence_score": ..., "created_at": ...}

    new_entries_count = len(model_logs)

    model_rows = []
    for item in model_logs:
        model_rows.append({
            "count": 0,
            "hate_speech": 0,
            "offensive_language": 0,
            "neither": 0,
            "class": item["prediction_index"],
            "tweet": item["text"],
            "created_at": item["created_at"],
        })

    if model_rows:
        df = pd.concat([df, pd.DataFrame(model_rows)], ignore_index=True)
    
    moderator_logs = read_json(
        s3,
        DATA_BUCKET,
        f"{raw_dir}moderator_logs.json"
    )
    # moderator log is {"text": "...", "moderator_decision" : "...", "decision_index" : 0, "created_at" : ...}

    new_moderator_responses_count = len(moderator_logs)

    label_column = {
        0: "hate_speech",
        1: "offensive_language",
        2: "neither",
    }

    for item in moderator_logs:
        mask = df["tweet"] == item["text"]
        if not mask.any():
            continue

        idx = mask.idxmax()
        label_idx = item["decision_index"]
        col = label_column[label_idx]

        df.at[idx, col] += 1
        df.at[idx, "count"] += 1

        count = df.at[idx, "count"]
        if count > 0:
            if df.at[idx, col] / count > 0.5:
                df.at[idx, "class"] = label_idx

    write_csv(s3, DATA_BUCKET, csv_key, df)

    metadata = {
        "version": version,
        "path": raw_dir[:-1],
        "new_entries": new_entries_count,
        "new_moderator_responses": new_moderator_responses_count,
        "created_at": datetime.now().isoformat()
    }

    write_json(
        s3,
        DATA_BUCKET,
        f"{raw_dir}_metadata.json",
        metadata
    )

    write_json(
        s3,
        DATA_BUCKET,
        f"{METADATA_PREFIX}raw_tweets_latest.json",
        metadata
    )

    write_json(
        s3,
        DATA_BUCKET,
        f"{METADATA_PREFIX}cursor.json",
        {"last_ingested_at": run_created_at}
    )


    s3.delete_object(
        Bucket=DATA_BUCKET,
        Key=f"{METADATA_PREFIX}raw_version_in_progress.json"
    )

from datetime import datetime, timedelta

default_args = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "start_date" : datetime(2026, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ml_data_pipeline_full",
    default_args=default_args,
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:

    t1 = PythonOperator(
        task_id="get_new_raw_version",
        python_callable=get_raw_version
    )

    t2 = PythonOperator(
        task_id="download_moderator_logs",
        python_callable=build_raw_moderator_logs
    )

    t3 = PythonOperator(
        task_id="download_model_logs",
        python_callable=build_raw_model_logs
    )

    t4 = PythonOperator(
        task_id="merge_raw_data",
        python_callable=build_raw_csv_and_metadata
    )

    # Dependencies
    t1 >> [t2, t3] >> t4
