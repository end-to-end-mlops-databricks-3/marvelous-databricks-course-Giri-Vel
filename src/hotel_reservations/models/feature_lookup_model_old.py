"""FeatureLookUp model implementation."""

from datetime import datetime

import mlflow
import mlflow.pyfunc
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMClassifier
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hotel_reservations.config import ProjectConfig, Tags

from pyspark.sql.functions import expr, col, when, pandas_udf, concat, lit
from pyspark.sql.types import IntegerType, ArrayType, DoubleType, StringType



class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def _dedupe_columns(self, df: DataFrame) -> DataFrame:
        """Remove duplicate column names, preserving first occurrence."""
        
        seen = set()
        unique_cols = []
        for c in df.columns:
            if c not in seen:
                unique_cols.append(c)
                seen.add(c)
        return df.select(*unique_cols)

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        # self.parameters = self.config.parameters
        self.parameters = {
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "max_depth": 8,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.giridhar_package_hotres_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.giridhar_package_calculate_adultschilds"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the house_features table and populate it.

        This table stores features related to houses.
        """

        self.train_set = self._dedupe_columns(self.train_set)
        self.test_set  = self._dedupe_columns(self.test_set)

    
        all_feats = cfg.num_features + cfg.cat_features

    
        id_col   = cfg.id_column[0]
        cols_defs = []
        for f in all_feats:
            t = "INT" if f != "arrival_date" else "STRING"
            cols_defs.append(f"{f} {t}")

        schema = ",\n    ".join([f"{id_col} STRING NOT NULL"] + cols_defs)

        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name} (
            Booking_ID STRING NOT NULL,
            type_of_meal_plan STRING,
            required_car_parking_space INT,
            room_type_reserved STRING,
            market_segment_type STRING,
            no_of_adults INT,
            no_of_children INT,
            CONSTRAINT hotelres_pk PRIMARY KEY (Booking_ID)
        )
        USING DELTA
        TBLPROPERTIES (
            delta.enableChangeDataFeed = true
        )
        """)

        # self.spark.sql(f"""
        #         CREATE OR REPLACE TABLE {self.feature_table_name}
        #         (Booking_ID STRING NOT NULL, type_of_meal_plan STRING, required_car_parking_space INT, room_type_reserved STRING, market_segment_type STRING, no_of_adults INT, no_of_children INT);
        #         """)
        # primary key on Databricks is not enforced!
        # try:
        #     self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT hotres_pk_demo PRIMARY KEY(Booking_ID);")
        # except Exception:
        #     pass
        # self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")



        # self.spark.sql(f"""
        #         INSERT INTO {self.feature_table_name}
        #         SELECT Booking_ID, type_of_meal_plan, required_car_parking_space, room_type_reserved, market_segment_type, no_of_adults, no_of_children
        #         FROM {self.catalog_name}.{self.schema_name}.train_set
        #         """)
        # self.spark.sql(f"""
        #         INSERT INTO {self.feature_table_name}
        #         SELECT Booking_ID, type_of_meal_plan, required_car_parking_space, room_type_reserved, market_segment_type, no_of_adults, no_of_children
        #         FROM {self.catalog_name}.{self.schema_name}.test_set
        #         """)


        select_csv = ", ".join([id_col] + all_feats)
        for part in ["train_set", "test_set"]:
            self.spark.sql(f"""
                INSERT INTO {self.feature_table_name}
                SELECT {select_csv}
                FROM {cfg.catalog_name}.{cfg.schema_name}.{part}
            """)
        
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the house's age.

        This function subtracts the year built from the current year.
        """

        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(no_of_adults INT, no_of_children INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return no_of_adults + no_of_children
        $$
        """)

        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns and casts 'YearBuilt' to integer type.
        """
        
        """Load train & test sets, cast types, keep only relevant cols."""

        cfg = self.config
        id_col = cfg.id_column[0]
        num_feats = cfg.num_features
        cat_feats = cfg.cat_features
        all_cols = [id_col] + num_feats + cat_feats + [cfg.target]

        # read and select only the columns we need
        self.train_set = (
            self.spark
                .table(f"{cfg.catalog_name}.{cfg.schema_name}.train_set")
                .select(*all_cols)
                .withColumn(id_col, col(id_col).cast("string"))
        )
        for f in num_feats:
            # cast all numeric features to int, except arrival_date if you prefer string
            if f != "arrival_date":
                self.train_set = self.train_set.withColumn(f, col(f).cast("int"))

        self.test_set = (
            self.spark
                .table(f"{cfg.catalog_name}.{cfg.schema_name}.test_set")
                .select(*all_cols)
                .withColumn(id_col, col(id_col).cast("string"))
        )
        for f in num_feats:
            if f != "arrival_date":
                self.test_set = self.test_set.withColumn(f, col(f).cast("int"))

        logger.info(f"Loaded train cols → {self.train_set.columns}")
        logger.info(f"Loaded test  cols → {self.test_set.columns}")



        # """Load train & test sets from Delta, cast types, drop unused cols."""

        # logger.info(f"Loading tables: train_set cols → {self.spark.table(f'{self.config.catalog_name}.{self.config.schema_name}.train_set').columns}")
        # logger.info(f"               test_set cols → {self.spark.table(f'{self.config.catalog_name}.{self.config.schema_name}.test_set').columns}")

        # self.train_set: DataFrame = self.spark.table(
        #     f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        # )
        # self.test_set: DataFrame = self.spark.table(
        #     f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        # )

        # self.train_set = (
        #     self.train_set
        #     .withColumn("Booking_ID", col("Booking_ID").cast("string"))
        #     .withColumn("required_car_parking_space", 
        #                 col("required_car_parking_space").cast("int"))
        #     .withColumn("no_of_adults", col("no_of_adults").cast("int"))
        #     .withColumn("no_of_children", col("no_of_children").cast("int"))
        # )

        # self.test_set = (
        #     self.test_set
        #     .withColumn("Booking_ID", col("Booking_ID").cast("string"))
        #     .withColumn("required_car_parking_space", 
        #                 col("required_car_parking_space").cast("int"))
        #     .withColumn("no_of_adults", col("no_of_adults").cast("int"))
        #     .withColumn("no_of_children", col("no_of_children").cast("int"))
        # )

        # logger.info(f"✅ Loaded train ({self.train_set.count()} rows) "
        #             f"and test ({self.test_set.count()} rows) datasets.")


    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """

        cfg = self.config
        id_col = cfg.id_column[0]
        # from your notebook
        lookup_cols = [
            "type_of_meal_plan",
            "required_car_parking_space",
            "room_type_reserved",
            "market_segment_type",
            "no_of_adults",
            "no_of_children"
        ]
        
        lookup_cols = list(dict.fromkeys(self.num_features + self.cat_features))
        logger.info(f"Feature lookups on → {lookup_cols}")
        logger.info(f"FeatureFunction inputs → {{'no_of_adults','no_of_children'}}")
        
        # lookup_cols = self.num_features + self.cat_features
        df_for_fe = self.train_set.drop(*self.num_features, *self.cat_features)

        # 1. Build the training set via FeatureEngineeringClient
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=lookup_cols,  #self.num_features + self.cat_features,
                    lookup_key="Booking_ID",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="guest_count",
                    input_bindings={
                        "no_of_adults": "no_of_adults",
                        "no_of_children": "no_of_children",
                    },
                ),
            ],
            exclude_columns=lookup_cols, #[],
        )

        # 2. Load to Pandas for downstream training
        self.training_df = self.training_set.load_df().toPandas()

        # 3. Compute the same feature on test set
        self.test_set = self.test_set.withColumn(
            "guest_count",
            self.test_set["no_of_adults"] + self.test_set["no_of_children"]
        ).toPandas()

        # 4. Split into X/y
        feat_cols = lookup_cols + ["guest_count"]  #self.num_features + self.cat_features + ["guest_count"]
        self.X_train = self.training_df[feat_cols]
        self.y_train = self.training_df[self.target]
        self.X_test  = self.test_set[feat_cols]
        self.y_test  = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")
        logger.info(f"Training on X_train cols → {list(self.X_train.columns)}; y_train name → {self.y_train.name}")

        # 1. Preprocessing + LightGBM classifier pipeline
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)],
            remainder="passthrough",
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LGBMClassifier(**self.parameters)),
        ])

        # 2. MLflow run
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            # 3. Fit & predict
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            # 4. Classification metrics
            acc = accuracy_score(self.y_test, y_pred)
            f1  = f1_score(self.y_test, y_pred, average="weighted")
            prec = precision_score(self.y_test, y_pred, average="weighted")
            rec  = recall_score(self.y_test, y_pred, average="weighted")
            logger.info(f"Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")

            # 5. Log params & metrics
            mlflow.log_param("model_type", "LightGBMClassifier w/ preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metrics({"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec})

            # 6. Infer signature & log model
            signature = infer_signature(self.X_train, y_pred)
            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-packaged-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

        logger.info("🚀 training ended under mlflow...")



    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """

        logger.info(f"Registering model run_id → {self.run_id}")

        model_name = f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.model_name}_packaged_fe"
        model_uri = f"runs:/{self.run_id}/lightgbm-pipeline-packaged-model-fe"
        registered = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=self.tags,
        )

        # 2. Alias the latest version to “latest-model”
        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=registered.version,
        )

        logger.info(f"✅ Registered model {model_name} v{registered.version} as ‘latest-model’")
        return str(registered.version)



    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """

        logger.info(f"Scoring DataFrame cols → {list(X.columns)}")

        model_name = f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.model_name}_packaged_fe"
        model_uri = f"models:/{model_name}@latest-model"

        # Score the batch via the FE client
        predictions = self.fe.score_batch(
            model_uri=model_uri,
            df=X
        )

        logger.info(f"✅ Scored batch with model {model_uri}")
        return predictions
    


