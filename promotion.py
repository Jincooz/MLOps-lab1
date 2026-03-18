# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedModelInput

# COMMAND ----------

EXPERIMENT_NAME = "/Users/your-email/asmm_classifier"
REGISTERED_MODEL_NAME = "asmm_classifier"
ENDPOINT_NAME = "Lab3endpoint"
CATALOG_NAME = "workspace"
SCHEMA_NAME = "default"
ACCURACY_THRESHOLD = 0.05

client = MlflowClient()

# COMMAND ----------

def get_accuracy(model_version):
    run = client.get_run(model_version.run_id)
    return float(run.data.metrics.get("accuracy", 0))

# COMMAND ----------

def update_endpoint(new_version):
    w = WorkspaceClient()
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_models=[ServedModelInput(
            model_name=f"{CATALOG_NAME}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}",
            model_version=new_version,
            workload_size="Small",
            scale_to_zero_enabled=True
    )]
    )
    print(f"Endpoint {ENDPOINT_NAME} updated to version {new_version}")


# COMMAND ----------

versions = client.search_model_versions(f"name='{CATALOG_NAME}.{SCHEMA_NAME}.{REGISTERED_MODEL_NAME}'")
champion_version = None
challengers = []
for v in versions:
    version_data = client.get_model_version(v.name, v.version)
    tags = version_data.tags
    if not tags.get("candidate_type") or not tags.get("environment"):
        client.set_model_version_tag(v.name, v.version, "candidate_type", "challenger")
        client.set_model_version_tag(v.name, v.version, "environment", "stage")
        tags = {"candidate_type": "challenger", "environment": "stage"}

    if tags.get("candidate_type") == "challenger" and tags.get("environment") == "stage":
        challengers.append(v)
    if tags.get("candidate_type") == "champion" and tags.get("environment") == "prod":
       champion_version = v

# COMMAND ----------

def main():
    if not challengers:
        print("No challengers found in stage.")
        return
    
    if champion_version is None:
        champ_acc = -1
    else:
        champ_acc = get_accuracy(champion_version)
    
    best_challenger = None
    best_challenger_acc = -1
    for c in challengers:
        acc = get_accuracy(c)
        if acc > best_challenger_acc:
            best_challenger_acc = acc
            best_challenger = c
    
    if (1 - champ_acc) * (1-ACCURACY_THRESHOLD) > (1 - best_challenger_acc):
        if champion_version:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, champion_version.version, "candidate_type", "depr")
            client.set_model_version_tag(REGISTERED_MODEL_NAME, champion_version.version, "environment", "depr")

        client.set_model_version_tag(REGISTERED_MODEL_NAME, best_challenger.version, "candidate_type", "champion")
        client.set_model_version_tag(REGISTERED_MODEL_NAME, best_challenger.version, "environment", "prod")

        for c in challengers:
            if c.version != best_challenger.version:
                client.set_model_version_tag(REGISTERED_MODEL_NAME, c.version, "environment", "depr")
                client.set_model_version_tag(REGISTERED_MODEL_NAME, c.version, "candidate_type", "depr")
        
        update_endpoint(best_challenger.version)
    else:
        print("No new champion found.")
        for c in challengers:
            client.set_model_version_tag(REGISTERED_MODEL_NAME, c.version, "environment", "depr")
            client.set_model_version_tag(REGISTERED_MODEL_NAME, c.version, "candidate_type", "depr")


main()