# Databricks notebook source

import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from marvelous.common import is_databricks
from mlflow import MlflowClient
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModelContext
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig
from hotel_reservations.utils import adjust_predictions

# COMMAND ----------

# !uv pip install -e ..

# COMMAND ----------
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")


config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

train_set = spark.table(f"{config.catalog_name}.{config.schema_name}.train_set").toPandas()
X_train = train_set[config.num_features + config.cat_features]  # config.id_column + new_columns_created]
y_train = train_set[config.target]

# COMMAND ----------
# X_train.head()

# COMMAND ----------
# X_train.info()

# COMMAND ----------
# set the training parameters for the model

train_parameters = {
    "learning_rate": 0.1,
    "n_estimators": 1000,
    "max_depth": 8,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}


# COMMAND ----------
# Create a preprocessing and model pipeline

# set the parameters for the model

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            ColumnTransformer(
                transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), config.cat_features)],
                remainder="passthrough",
            ),
        ),
        ("classifier", LGBMClassifier(**train_parameters)),
    ]
)

pipeline.fit(X_train, y_train)

# COMMAND ----------
mlflow.set_experiment("/Shared/hotres_giridhar_log&reg")
with mlflow.start_run(
    run_name="demo-run-model-1",
    tags={"git_sha": "1234567890abcd", "branch": "week2"},
    description="demo run for model logging",
) as run:
    # Log parameters and metrics
    run_id = run.info.run_id
    mlflow.log_param("model_type", "LightGBM Classifier with preprocessing")
    mlflow.log_params(train_parameters)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="lightgbm-pipeline-model", signature=signature)

# COMMAND ----------
# Load the model using the alias and test predictions - not recommended!
# This may be working in a notebook but will fail on the endpoint
artifact_uri = mlflow.get_run(run_id=run_id).to_dictionary()["info"]["artifact_uri"]


# COMMAND ----------
model_name = f"{config.catalog_name}.{config.schema_name}.model_demo-1"
model_version = mlflow.register_model(
    model_uri=f"runs:/{run_id}/lightgbm-pipeline-model", name=model_name, tags={"git_sha": "1234567890abcd"}
)

# COMMAND ----------
# only searching by name is supported
v = mlflow.search_model_versions(filter_string=f"name='{model_name}'")
print(v[0].__dict__)

# COMMAND ----------
# not supported
mlflow.search_model_versions(filter_string=f"run_id='{run_id}'")

# COMMAND ----------
# not supported
v = mlflow.search_model_versions(filter_string="tags.git_sha='1234567890abcd'")

# COMMAND ----------
client = MlflowClient()

# COMMAND ----------
# this will fail: latest is reserved
client.set_registered_model_alias(name=model_name, alias="latest", version=model_version.version)

# COMMAND ----------
# loading latest also fails
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@latest")

# COMMAND ----------
# let's set latest-model alias instead

client.set_registered_model_alias(name=model_name, alias="latest-model", version=model_version.version)

# COMMAND ----------
model_uri = f"models:/{model_name}@latest-model"
sklearn_pipeline = mlflow.sklearn.load_model(model_uri)
predictions = sklearn_pipeline.predict(X_train[0:1])
print(predictions)

# COMMAND ----------
# A better way, also explained here
#  will work in a later version of mlflow:
# https://docs.databricks.com/aws/en/machine-learning/model-serving/model-serving-debug
# https://www.databricksters.com/p/pyfunc-it-well-do-it-live

# mlflow.models.predict(model_uri, X_train[0:1])

# COMMAND ----------
# check the datatype of predictions
# print(type(predictions))

# COMMAND ----------
# Let's wrap it around a custom model


class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    """A custom model wrapper for the house price prediction model."""

    def __init__(self, model: object) -> None:
        """Initialize the model wrapper with a given model."""
        self.model = model

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> dict:
        """Predict house prices using the wrapped model."""
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
wrapped_model = HousePriceModelWrapper(sklearn_pipeline)  # we pass the loaded model to the wrapper

mlflow.set_experiment(experiment_name="/Shared/hotres_giridhar_pyfunc1")
with mlflow.start_run(tags={"branch": "week2", "git_sha": "1234567890efgh"}) as run:
    run_id = run.info.run_id
    # signature = infer_signature(model_input=X_train, model_output={'Prediction': 1000.0})
    signature = infer_signature(model_input=X_train, model_output=pipeline.predict(X_train))
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        # additional_pip_deps=[f"../dist/hotel_reservations-{__version__}-py3-none-any.whl",
        #                      ],
        # additional_pip_deps=[f"hotel_reservations=={__version__}"],
        # hard coding for now
        additional_pip_deps=["hotel_reservations==0.0.1"],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-hotel-reservations-model",
        # code_paths = [f"../dist/hotel_reservations-{__version__}-py3-none-any.whl"],
        # hard coding for now
        code_paths=["../dist/hotel_reservations-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
# Another way of doing the same thing:


class HousePriceModelWrapper2(mlflow.pyfunc.PythonModel):
    """A custom model wrapper for the house price prediction model.

    This version loads the model from artifacts.
    """

    def load_context(self, context: PythonModelContext) -> None:
        """Load the model from the context artifacts."""
        self.model = mlflow.sklearn.load_model(context.artifacts["lightgbm-pipeline-model"])

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> dict:
        """Predict house prices using the wrapped model."""
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")


# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/hotres_giridhar_pyfunc2")
with mlflow.start_run(tags={"branch": "week2", "git_sha": "1234567890ijkl"}) as run:
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={"Prediction": 1000.0})
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        # additional_pip_deps=[f"../dist/hotel_reservations-{__version__}-py3-none-any.whl",
        #                      ],
        # additional_pip_deps=[f"hotel_reservations=={__version__}"],
        # hard coding for now
        additional_pip_deps=["hotel_reservations==0.0.1"],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=HousePriceModelWrapper2(),
        artifact_path="pyfunc-hotel-reservations-model",
        artifacts={"lightgbm-pipeline-model": f"models:/{model_name}@latest-model"},
        # code_paths = [f"../dist/hotel_reservations-{__version__}-py3-none-any.whl"],
        # hard coding for now
        code_paths=["../dist/hotel_reservations-0.0.1-py3-none-any.whl"],
        signature=signature,
    )

# COMMAND ----------
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-hotel-reservations-model")
# COMMAND ----------
pyfunc_model.predict(X_train[0:1])
# COMMAND ----------
