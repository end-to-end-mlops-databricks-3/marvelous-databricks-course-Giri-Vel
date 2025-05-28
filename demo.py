# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)

# COMMAND ----------

print("Spark version:", spark.version)

# COMMAND ----------
# spark.stop()
# COMMAND ----------
spark.range(1).count()
# COMMAND ----------
