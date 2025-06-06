from databricks.sdk.runtime import dbutils  # explicitly list what you need
from pyspark.sql.context import SQLContext
from pyspark.sql.functions import udf as U
from pyspark.sql.session import SparkSession

udf = U
spark: SparkSession
sc = spark.sparkContext
sqlContext: SQLContext
sql = sqlContext.sql
table = sqlContext.table
getArgument = dbutils.widgets.getArgument

def displayHTML(html: str) -> None: ...
def display(input: object = None, *args: object, **kwargs: object) -> None: ...
