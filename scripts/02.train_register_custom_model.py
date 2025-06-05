# import argparse

# import mlflow
# from loguru import logger
# from pyspark.dbutils import DBUtils
# from pyspark.sql import SparkSession

# from hotel_reservations.config import ProjectConfig, Tags
# from hotel_reservations.models.custom_model import CustomModel

# # Configure tracking uri
# mlflow.set_tracking_uri("databricks")
# mlflow.set_registry_uri("databricks-uc")

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--root_path",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--env",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--git_sha",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--job_run_id",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--branch",
#     action="store",
#     default=None,
#     type=str,
#     required=True,
# )


# args = parser.parse_args()
# root_path = args.root_path
# config_path = f"{root_path}"


"""Modeling Pipeline module."""

import argparse
import os
import sys
from pathlib import Path

import mlflow

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from loguru import logger
from pyspark.dbutils import DBUtils

# Ensure project src is on path
sys.path.append(str(Path.cwd().parent / "src"))
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.custom_model import CustomModel

# Base config defaults
base_dir = str(Path(__file__).parent.parent)  # Points to project root
default_config_path = os.path.join(base_dir, "project_config.yml")

# Configure tracking URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Argument parsing with fallback defaults
try:
    parser = argparse.ArgumentParser(description="Train and register custom model")
    parser.add_argument("--root_path", default=default_config_path, type=str, required=True)
    parser.add_argument("--env", default=None, type=str, required=True)
    parser.add_argument("--git_sha", default=None, type=str, required=True)
    parser.add_argument("--job_run_id", default=None, type=str, required=True)
    parser.add_argument("--branch", default=None, type=str, required=True)
    args = parser.parse_args()
except (argparse.ArgumentError, SystemExit):
    # Fallback to safe defaults when flags are missing
    args = argparse.Namespace(
        root_path=default_config_path, env="dev", git_sha="123abc", job_run_id="job-id-01", branch="week2"
    )

root_path = args.root_path
config_path = root_path  # points to project_config.yml

# host = os.environ["DATABRICKS_HOST"].replace("https://", "")
# token = os.environ["DATABRICKS_TOKEN"]
# cluster = os.environ["DATABRICKS_CLUSTER_ID"]
# # the below url would work when code is executed from gitbash to connect to DBR
# url = f"sc://{host}:15002?token={token}&clusterId={cluster}"

# # url = "sc://${Env:DATABRICKS_HOST}:15002?token={Env:DATABRICKS_TOKEN}&clusterId=${Env:DATABRICKS_CLUSTERID}"


# # spark = SparkSession.builder.remote(url).getOrCreate()
# # trying to see if this would work
# # spark.conf.set("spark.databricks.service.address", f"https://{host}")
# # spark.conf.set("spark.databricks.service.token", token)
# # spark.conf.set("spark.databricks.clusterId", cluster)

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = DatabricksSession.builder.getOrCreate()


dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
custom_model = CustomModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[
        r"D:\MLOPS learning\maven_mlops_actual\marvelous-databricks-course-Giri-Vel\dist\hotel_reservations-0.0.1-py3-none-any.whl"
    ],
)
logger.info("Model initialized.")

# Load data and prepare features
custom_model.load_data()
custom_model.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
custom_model.train()
custom_model.log_model()
logger.info("Model training completed.")

custom_model.register_model()
logger.info("Registered model")
