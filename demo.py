# Databricks notebook source
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)

# # COMMAND ----------
# df = spark.read.table("samples.nyctaxi.trips")
# df.show(5)

# # print("Spark version:", spark.version)

# # # COMMAND ----------
# # # spark.stop()
# # COMMAND ----------
# spark.range(1).count()
# # COMMAND ----------


# from pyspark.sql import SparkSession
# import os

# # --- Step 1: (Optional but recommended) Stop the existing Spark session safely ---
# try:
#     if 'spark' in globals() and spark is not None:
#         spark.stop()
#         print("Existing Spark session stopped.")
# except Exception as e:
#     print(f"No active Spark session to stop or error stopping: {e}")

# # --- Step 2: Get the Spark Connect URL ---
# # In Databricks Connect, the Spark Connect URL is typically
# # derived from your environment variables or the Databricks extension.
# # The 'SPARK_REMOTE' environment variable is commonly used for this.
# # If you don't have it set as an environment variable, you might need
# # to manually construct it based on your workspace and cluster ID.
# # Format: sc://<databricks-workspace-host>:443/;x-databricks-cluster-id=<cluster-id>;user_id=<user-id> (for older versions)
# # Or simpler: sc://<databricks-workspace-host>:443/;x-databricks-cluster-id=<cluster-id>
# # Or even simpler: sc://<databricks-workspace-host>:443/
# # The 'SPARK_REMOTE' environment variable is the most robust way.

# spark_remote_url = os.environ.get("SPARK_REMOTE")

# if not spark_remote_url:
#     print("WARNING: SPARK_REMOTE environment variable not set. Attempting to get from Databricks extension.")
#     # Fallback: If SPARK_REMOTE is not set, the Databricks extension in VS Code
#     # usually handles this implicitly when you first connect.
#     # For programmatic re-creation, it's best to have SPARK_REMOTE set.
#     # You might need to manually set it here if you're not getting it from env.
#     # Example (replace with your actual host and cluster ID):
#     # spark_remote_url = "sc://<your-workspace-host>:443/;x-databricks-cluster-id=<your-cluster-id>"
#     # If the Databricks VS Code extension is active, it *should* manage this.
#     # The error suggests it's not being picked up automatically after spark.stop().
#     # The safest bet is to ensure SPARK_REMOTE is set.
#     # For now, let's assume it should be available.

#     # If you still get this error, you need to ensure SPARK_REMOTE is truly set
#     # in your VS Code environment or explicitly define it here.
#     # For the purposes of this example, let's raise an error if it's not found
#     # to highlight its importance.
#     raise ValueError("SPARK_REMOTE environment variable is not set. Please ensure your Databricks Connect setup is correct or set it explicitly.")


# # --- Step 3: Re-initialize SparkSession with the remote URL ---
# try:
#     spark = SparkSession.builder.remote(spark_remote_url).getOrCreate()
#     print("New Spark session created successfully.")
# except Exception as e:
#     print(f"Error creating new Spark session: {e}")
#     print("Please ensure your Databricks cluster is running and your Databricks Connect configuration (especially SPARK_REMOTE) is correct.")
#     # Re-raise to show the original error if needed
#     raise

# # --- Step 4: Test the new session ---
# try:
#     result = spark.range(1).count()
#     print(f"Spark command executed successfully: {result}")
# except Exception as e:
#     print(f"Error running Spark command after re-initialization: {e}")
# # COMMAND ----------

# test_connect.py

# import os
# os.environ["SPARK_CONNECT_URL"] = "sc://dbc-c2e8445d-159d.cloud.databricks.com:15001"
# os.environ["DATABRICKS_TOKEN"]   = "eyJraWQiOiJkZmJjOWVmMThjZTQ2ZTlhMDg2NWZmYzlkODkxYzJmMjg2NmFjMDM3MWZiNDlmOTdhMDg1MzBjNWYyODU3ZTg4IiwidHlwIjoiYXQrand0IiwiYWxnIjoiUlMyNTYifQ.eyJjbGllbnRfaWQiOiJkYXRhYnJpY2tzLWNsaSIsInNjb3BlIjoiYWxsLWFwaXMgb2ZmbGluZV9hY2Nlc3MiLCJpc3MiOiJodHRwczovL2RiYy1jMmU4NDQ1ZC0xNTlkLmNsb3VkLmRhdGFicmlja3MuY29tL29pZGMiLCJhdWQiOiI2ODk5OTA1Njc3MzU4MzIiLCJzdWIiOiJnaXJpZGhhcmFudmVsQGdtYWlsLmNvbSIsImlhdCI6MTc0ODYyOTg2NCwiZXhwIjoxNzQ4NjMzNDY0LCJqdGkiOiIzZTI0YzM5ZC1mZmRjLTQ0ZDItYTA2Zi00MWFhMjcwYjUwZTMifQ.be8EOB3GdeZqylSQq1jhCcpF6C7ZqcvMiIDcWd7hqxMFV1FSdU9lfNMZtnkG3MUDTxEOsxdsU7kMEz4ipi1fCfC9nurNhOxP8EG18oGXDInSllCeB64W5vwq5ftnxuqPKeGNyt20qEM8XwsMufECN88WCJHYaClPc-U5hrcL6TNjA3-wcNZU1lbre51rIYtV1u5pb-z6MAWCErbsykfW8WJrwku1kxtCFK7cpTOobGNPuVEWA2H4g5iyFcAzavII7GsrnjfKxhoPm8e7ubVT9SJ5rB7ebi7i8Zgkbiv8WkRpJIaEbPO1o7tPjSY0QhfJS0PuRmukBjCkhSjIrI9jHQ"

# from databricks.connect import DatabricksSession
# spark = DatabricksSession.builder.getOrCreate()
# print(spark.range(3).collect())
