# Databricks notebook source
# MAGIC %pip install transformers torch datasets

# COMMAND ----------

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler

from sklearn.metrics import balanced_accuracy_score, recall_score, f1_score

import mlflow 
import mlflow.sklearn
import logging

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

client = mlflow.tracking.MlflowClient()

# COMMAND ----------

logging.basicConfig(level=logging.INFO)

# COMMAND ----------

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
NUM_LABELS = 3
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

confidence_threshold = 0.6 

PREDICTION_MAP = ["hate speech", "offensive language", "neither"]

# COMMAND ----------

EXPERIMENT_NAME = "/Users/kopcov.vitaly@lll.kpi.ua/asmm_Lab3"
REGISTERED_MODEL_NAME = "asmm_classifier"

CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
train_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.train"
val_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.val"

# COMMAND ----------

class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx]
        }


# COMMAND ----------

train_df = spark.read.table(train_table).toPandas()
val_df = spark.read.table(val_table).toPandas()

# COMMAND ----------

from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["class"]),
    y=train_df["class"]
)


# compute sample weights
sample_weights = np.array([class_weights[label] for label in train_df["class"]])
sampler = WeightedRandomSampler(
    weights=torch.tensor(sample_weights, dtype=torch.float),
    num_samples=len(sample_weights),
    replacement=True
)

# COMMAND ----------


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = CommentDataset(train_df["processed_tweet"], train_df["class"], tokenizer)
val_dataset   = CommentDataset(val_df["processed_tweet"],   val_df["class"],   tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# COMMAND ----------

import torch.nn.functional as F

# COMMAND ----------

def train(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    low_conf_mask = F.softmax(outputs.logits, dim=-1).cpu().numpy().max(axis=1) < confidence_threshold 
    low_conf_count = low_conf_mask.sum() 
    low_conf_ratio = low_conf_count / len(all_labels)

    y_val_bin = np.where(np.array(all_labels) == 2, 1, 0)
    y_pred_bin = np.where(np.array(all_preds) == 2, 1, 0)

    recall = recall_score(y_val_bin, y_pred_bin, pos_label=0)

    accuracy  = balanced_accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    return accuracy, f1, recall, low_conf_ratio, all_preds, all_labels


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

class BertWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        sys.path.insert(0, os.path.dirname(context.artifacts["preprocessor"]))
        from preprocessing import TextPreprocessor
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["tokenizer"])
        self.model = mlflow.pytorch.load_model(context.artifacts["model"])
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = TextPreprocessor()
        self.labels = ["hate speech", "offensive language", "neither"]

    def predict(self, context, model_input):
        import torch.nn.functional as F
        text = self.preprocessor.transform(model_input["comment"].tolist())
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits, dim=-1).cpu().numpy()
        return [{label: p for label, p in zip(self.labels, prob)} for prob in probs]

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="bert_tiny") as run:
    mlflow.set_tag("candidate_type", "challenger")
    mlflow.set_tag("Training Info", "Bert Tiny")
    mlflow.set_tag("model_type", "bert_tiny")
    mlflow.set_tag("environment","stage")
    mlflow.set_tag("framework","pytorch")
    mlflow.set_tag("problem", "Hate Speech and Offensive Language Dataset")
    
    history = spark.sql(f"DESCRIBE HISTORY {CATALOG_NAME}.{SCHEMA_NAME}.data")
    latest_version = history.first()["version"]
    
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "source_table" : f"{CATALOG_NAME}.{SCHEMA_NAME}.data",
        "source_table_version" : latest_version
    })

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, scheduler)
        accuracy,f1_score_val,recall,low_conf_ratio, preds, labels = evaluate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "accuracy" : accuracy,
            "f1_score": f1_score_val,
            "recall" : recall,
            "low_conf_ratio" : low_conf_ratio
        }, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {train_loss:.4f} — val_f1: {f1_score_val:.4f} - accuracy {accuracy} - recall {recall} - low_conf_ratio {low_conf_ratio}")
    

    tokenizer.save_pretrained("/tmp/tokenizer")
    mlflow.pytorch.save_model(model, "/tmp/bert_model")

    artifacts = {
        "tokenizer": "/tmp/tokenizer",
        "model": "/tmp/bert_model",
        "preprocessor": "./preprocessing.py"
    }
    mlflow.pyfunc.log_model(
        python_model=BertWrapper(),
        artifacts=artifacts,
        pip_requirements=["transformers", "torch"],
        signature = signature,
        registered_model_name=REGISTERED_MODEL_NAME
    )
