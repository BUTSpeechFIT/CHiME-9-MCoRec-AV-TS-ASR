"""
Extract clustering outputs from MLflow run.
"""

import mlflow
import os
from mlflow.tracking import MlflowClient
import json
import sys
from pathlib import Path
import argparse

# =========================
# ARGS
# =========================

parser = argparse.ArgumentParser(description="Extract outputs from MLflow artifacts.")
parser.add_argument("--tracking-uri", type=str, help="MLflow Tracking URI", default="http://127.0.0.1:5000")
parser.add_argument("run_id", type=str, help="MLflow Run ID")
args = parser.parse_args()

# =========================
# CONFIG
# =========================
TRACKING_URI = args.tracking_uri
RUN_ID = args.run_id

TARGET_FILENAME = "result_table.json"
EXPORT_DIR = "./outputs"

# =========================
# SETUP
# =========================
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# =========================
# FIND ARTIFACT RECURSIVELY
# =========================
def find_artifact(run_id, target, path=""):
    for a in client.list_artifacts(run_id, path):
        if a.is_dir:
            found = find_artifact(run_id, target, a.path)
            if found:
                return found
        else:
            if a.path.endswith(target):
                return a.path
    return None


artifact_path = find_artifact(RUN_ID, TARGET_FILENAME)

if artifact_path is None:
    print(f"❌ Artifact '{TARGET_FILENAME}' not found in run {RUN_ID}")
    sys.exit(1)

print(f"✅ Found artifact at: {artifact_path}")

# =========================
# DOWNLOAD ARTIFACT
# =========================
local_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID,
    artifact_path=artifact_path,
)

local_path = Path(local_path)
print(f"📥 Downloaded to: {local_path}")

# =========================
# LOAD JSON
# =========================
with open(local_path, "r", encoding="utf-8") as f:
    data = json.load(f)["data"]

# =========================
# PROCESS DATA
# =========================

outputs = {}
for item in data:
    score = item[0]
    trace_tags = item[4]
    output = item[5]
    session_id = trace_tags.get("session_id", "unknown_session")

    outputs[session_id] = output

os.makedirs(os.path.join(EXPORT_DIR, RUN_ID), exist_ok=True)
with open(os.path.join(EXPORT_DIR, RUN_ID, "spk_to_cluster.json"), "w") as f:
    json.dump(outputs, f, indent=4)
