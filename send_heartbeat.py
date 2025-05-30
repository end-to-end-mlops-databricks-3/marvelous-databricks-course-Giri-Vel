from pyspark.sql import SparkSession
# keep the Spark Connect session alive across retries
spark = SparkSession.builder \
    .config("spark.databricks.session.keepAliveEnabled", "true") \
    .getOrCreate()

print(spark.range(3).collect())
