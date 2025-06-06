"""Unit tests for CustomModel."""

import os

# from databricks.connect import DatabricksSession
import mlflow
import pandas as pd
import pytest
from conftest import CATALOG_DIR, TRACKING_URI

try:
    from databricks.connect import DatabricksSession
except ImportError:

    class DatabricksSession:
        """Dummy DatabricksSession stub for tests."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize dummy session."""
            pass

        def sql(self, *args: object, **kwargs: object) -> None:
            """Stub .sql(...) to return None."""
            return None


from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.entities.model_registry.registered_model import RegisteredModel
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from hotel_reservations import PROJECT_DIR
from hotel_reservations.config import ProjectConfig, Tags
from hotel_reservations.models.custom_model import CustomModel

# Make your local SparkSession appear as DatabricksSession
CustomModel.DatabricksSession = DatabricksSession

# @pytest.fixture(scope="session")
# def config() -> ProjectConfig:
#     yaml_path = PROJECT_DIR / "project_config.yml"
#     return ProjectConfig.from_yaml(str(yaml_path), env="dev")

# @pytest.fixture(scope="session")
# def tags() -> Tags:
#     return Tags(git_sha="wxyz", branch="week2_test", job_run_id="9")
# @pytest.fixture(scope="session")
# def spark_session() -> SparkSession:
#     # 1) Remove *any* Connect‐mode or Databricks‐Connect env vars:
#     for var in [
#         "SPARK_REMOTE",
#         "SPARK_MASTER",
#         "SPARK_CONNECT_MODE_ENABLED",
#         "SPARK_LOCAL_REMOTE",
#         "DATABRICKS_HOST",
#         "DATABRICKS_TOKEN",
#         "DATABRICKS_CLUSTER_ID",
#         "PYSPARK_SUBMIT_ARGS",
#     ]:
#         os.environ.pop(var, None)

#     # 2) Create a plain local SparkContext and wrap it in a SparkSession:
#     sc = SparkContext(master="local[*]", appName="pytest")
#     spark = SparkSession(sc)

#     return spark


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    """Fake SparkSession that returns Pandas DataFrames when `.table(...)` is called.

    This matches the reference’s pattern of `spark.table(...).toPandas()`.
    """
    # 1) Unset all Databricks‐Connect / Spark‐remote flags so nothing tries to open a real JVM.
    for var in [
        "SPARK_REMOTE",
        "SPARK_MASTER",
        "SPARK_CONNECT_MODE_ENABLED",
        "SPARK_LOCAL_REMOTE",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLUSTER_ID",
        "PYSPARK_SUBMIT_ARGS",
    ]:
        os.environ.pop(var, None)

    # 2) Preload the CSVs from tests/catalog into Pandas DataFrames:
    from conftest import CATALOG_DIR  # this points to tests/catalog

    train_pdf = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    test_pdf = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())

    # 3) Create a “dummy” SparkSession‐like object:
    class DummyDF:
        """A dummy DataFrame that mimics the Spark DataFrame API."""

        def __init__(self, pdf: pd.DataFrame) -> None:
            """Initialize with a Pandas DataFrame."""
            self._pdf = pdf

        def toPandas(self) -> pd.DataFrame:
            """Return the underlying Pandas DataFrame."""
            return self._pdf

    class DummySpark:
        def table(self, full_table_name: str) -> DummyDF:
            """CustomModel.load_data() calls spark.table(f"{catalog}.{schema}.train_set") and spark.table(f"{catalog}.{schema}.test_set").

            We will ignore catalog/schema and just return the right PDF.
            """
            if full_table_name.endswith(".train_set"):
                return DummyDF(train_pdf)
            elif full_table_name.endswith(".test_set"):
                return DummyDF(test_pdf)
            else:
                raise ValueError(f"No stub for table {full_table_name}")

    # 4) Return an actual DummySpark instance for all tests to consume:
    return DummySpark()


@pytest.fixture(scope="session")
def config() -> ProjectConfig:
    """Fixture to load the project configuration from a YAML file."""
    yaml_path = PROJECT_DIR / "project_config.yml"
    return ProjectConfig.from_yaml(str(yaml_path), env="dev")


@pytest.fixture(scope="session")
def tags() -> Tags:
    """Fixture to create a Tags instance with predefined values."""
    return Tags(git_sha="wxyz", branch="week2_test", job_run_id="9")


mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Test the initialization of CustomModel.

    This function creates a CustomModel instance and asserts that its attributes are of the correct types.

    :param config: Configuration for the project
    :param tags: Tags associated with the model
    :param spark_session: Spark session object
    """
    model = CustomModel(config=config, tags=tags, spark=spark_session, code_paths=[])
    assert isinstance(model, CustomModel)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    # assert isinstance(model.spark, SparkSession)
    assert hasattr(model.spark, "table"), "spark must implement a .table(...) method"
    assert isinstance(model.code_paths, list)
    assert not model.code_paths


def test_load_data_validate_df_assignment(mock_custom_model: CustomModel) -> None:
    """Validate correct assignment of train and test DataFrames from CSV files.

    :param mock_custom_model: Mocked CustomModel instance for testing.
    """
    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())

    # Execute
    mock_custom_model.load_data()

    # Validate DataFrame assignments
    pd.testing.assert_frame_equal(mock_custom_model.train_set, train_data)
    pd.testing.assert_frame_equal(mock_custom_model.test_set, test_data)


def test_load_data_validate_splits(mock_custom_model: CustomModel) -> None:
    """Verify correct feature/target splits in training and test data.

    :param mock_custom_model: Mocked CustomModel instance for testing.
    """
    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())

    # Execute
    mock_custom_model.load_data()

    # Verify feature/target splits
    expected_features = mock_custom_model.num_features + mock_custom_model.cat_features
    pd.testing.assert_frame_equal(mock_custom_model.X_train, train_data[expected_features])
    pd.testing.assert_series_equal(mock_custom_model.y_train, train_data[mock_custom_model.target])
    pd.testing.assert_frame_equal(mock_custom_model.X_test, test_data[expected_features])
    pd.testing.assert_series_equal(mock_custom_model.y_test, test_data[mock_custom_model.target])


def test_prepare_features(mock_custom_model: CustomModel) -> None:
    """Test that prepare_features method initializes pipeline components correctly.

    Verifies the preprocessor is a ColumnTransformer and pipeline contains expected
    ColumnTransformer and LGBMRegressor steps in sequence.

    :param mock_custom_model: Mocked CustomModel instance for testing
    """
    mock_custom_model.prepare_features()

    assert isinstance(mock_custom_model.preprocessor, ColumnTransformer)
    assert isinstance(mock_custom_model.pipeline, Pipeline)
    assert isinstance(mock_custom_model.pipeline.steps, list)
    assert isinstance(mock_custom_model.pipeline.steps[0][1], ColumnTransformer)
    assert isinstance(mock_custom_model.pipeline.steps[1][1], LGBMClassifier)


def test_train(mock_custom_model: CustomModel) -> None:
    """Test that train method configures pipeline with correct feature handling.

    Validates feature count matches configuration and feature names align with
    numerical/categorical features defined in model config.

    :param mock_custom_model: Mocked CustomModel instance for testing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    # new_columns_created = mock_custom_model.pipeline.named_steps['preprocessor'].get_feature_names_out()
    # new_columns_created = new_columns_created.tolist()
    # expected_feature_names = mock_custom_model.config.num_features + mock_custom_model.config.cat_features + mock_custom_model.config.id_column + new_columns_created

    # assert mock_custom_model.pipeline.n_features_in_ == len(expected_feature_names)
    # assert sorted(expected_feature_names) == sorted(mock_custom_model.pipeline.feature_names_in_)

    expected_input_features = mock_custom_model.config.num_features + mock_custom_model.config.cat_features

    assert mock_custom_model.pipeline.n_features_in_ == len(expected_input_features)
    assert sorted(expected_input_features) == sorted(mock_custom_model.pipeline.feature_names_in_)


def test_log_model_with_PandasDataset(mock_custom_model: CustomModel) -> None:
    """Test model logging with PandasDataset validation.

    Verifies that the model's pipeline captures correct feature dimensions and names,
    then checks proper dataset type handling during model logging.

    :param mock_custom_model: Mocked CustomModel instance for testing
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    expected_feature_names = mock_custom_model.config.num_features + mock_custom_model.config.cat_features

    assert mock_custom_model.pipeline.n_features_in_ == len(expected_feature_names)
    assert sorted(expected_feature_names) == sorted(mock_custom_model.pipeline.feature_names_in_)

    mock_custom_model.log_model(dataset_type="PandasDataset")

    # Split the following part
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_custom_model.experiment_name)
    assert experiment.name == mock_custom_model.experiment_name

    experiment_id = experiment.experiment_id
    assert experiment_id

    runs = client.search_runs(experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1
    latest_run = runs[0]

    model_uri = f"runs:/{latest_run.info.run_id}/giridhar_tests_model"
    logger.info(f"{model_uri= }")

    assert model_uri


def test_register_model(mock_custom_model: CustomModel) -> None:
    """Test the registration of a custom MLflow model.

    This function performs several operations on the mock custom model, including loading data,
    preparing features, training, and logging the model. It then registers the model and verifies
    its existence in the MLflow model registry.

    :param mock_custom_model: A mocked instance of the CustomModel class.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    mock_custom_model.register_model()

    client = MlflowClient()
    # model_name = f"{mock_custom_model.catalog_name}.{mock_custom_model.schema_name}.house_prices_test_model_custom"
    model_name = f"{mock_custom_model.catalog_name}.{mock_custom_model.schema_name}.giridhar_model_custom"

    try:
        model = client.get_registered_model(model_name)
        logger.info(f"Model '{model_name}' is registered.")
        logger.info(f"Latest version: {model.latest_versions[-1].version}")
        logger.info(f"{model.name = }")
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.error(f"Model '{model_name}' is not registered.")
        else:
            raise e

    assert isinstance(model, RegisteredModel)
    alias, version = model.aliases.popitem()
    assert alias == "latest-model"


def test_retrieve_current_run_metadata(mock_custom_model: CustomModel) -> None:
    """Test retrieving the current run metadata from a mock custom model.

    This function verifies that the `retrieve_current_run_metadata` method
    of the `CustomModel` class returns metrics and parameters as dictionaries.

    :param mock_custom_model: A mocked instance of the CustomModel class.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    metrics, params = mock_custom_model.retrieve_current_run_metadata()
    assert isinstance(metrics, dict)
    assert metrics
    assert isinstance(params, dict)

    expected = {"model_type": "LightGBM classifier with preprocessing"}
    expected.update(mock_custom_model.parameters)
    # assert params


def test_load_latest_model_and_predict(mock_custom_model: CustomModel) -> None:
    """Test the process of loading the latest model and making predictions.

    This function performs the following steps:
    - Loads data using the provided custom model.
    - Prepares features and trains the model.
    - Logs and registers the trained model.
    - Extracts input data from the test set and makes predictions using the latest model.

    :param mock_custom_model: Instance of a custom machine learning model with methods for data
                              loading, feature preparation, training, logging, and prediction.
    """
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")
    mock_custom_model.register_model()

    # input_data = mock_custom_model.test_set.drop(columns=[mock_custom_model.config.target])
    # input_data = input_data.where(input_data.notna(), None)  # noqa

    # for row in input_data.itertuples(index=False):
    #     row_df = pd.DataFrame([row._asdict()])
    #     print(row_df.to_dict(orient="split"))
    #     predictions = mock_custom_model.load_latest_model_and_predict(input_data=row_df)

    #     assert len(predictions) == 1

    # Drop target and fill NaNs once
    input_df = mock_custom_model.test_set.drop(columns=[mock_custom_model.config.target]).where(
        lambda df: df.notna(), None
    )

    # One MLflow load + predict for all rows
    preds = mock_custom_model.load_latest_model_and_predict(input_data=input_df)
    assert len(preds) == len(input_df)
