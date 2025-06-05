# Databricks notebook source
from databricks.connect import DatabricksSession

# Re-create the remote Spark session
spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------
# spark = SparkSession.builder.getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
print("Spark version:", spark.version)

# COMMAND ----------
spark.stop()


# COMMAND ----------
spark.range(1).count()
