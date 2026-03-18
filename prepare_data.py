# Databricks notebook source
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
RANDOM_STATE = 42

# COMMAND ----------

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import TextPreprocessor

# COMMAND ----------

df = spark.read.table(f"{CATALOG_NAME}.{SCHEMA_NAME}.data").toPandas()

df["processed_tweet"] = TextPreprocessor().transform(df["tweet"])

display(df)

# COMMAND ----------

train_df, val_df = train_test_split(
    df[["processed_tweet", "class"]],
    test_size= 0.2,
    random_state=RANDOM_STATE,
    stratify=df["class"]
)

train_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.train"
val_table = f"{CATALOG_NAME}.{SCHEMA_NAME}.val"

spark.sql(f"DROP TABLE IF EXISTS {train_table};")
spark.sql(f"DROP TABLE IF EXISTS {val_table}")

spark.createDataFrame(train_df).write.saveAsTable(train_table)
spark.createDataFrame(val_df).write.saveAsTable(val_table)