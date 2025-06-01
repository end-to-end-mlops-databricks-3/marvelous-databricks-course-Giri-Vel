from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.getOrCreate()
df = spark.read.csv("/FileStore/tables/Hotel_Reservations.csv", header=True, inferSchema=True)
df.show()
