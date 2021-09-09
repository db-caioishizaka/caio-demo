# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Demo
# MAGIC 
# MAGIC In this demo we will demonstrate how to:
# MAGIC 1. Get data from its source
# MAGIC 2. Write that data into Delta Bronze Tables
# MAGIC 3. Transform the data from Bronze and write it to Silver
# MAGIC 4. Use data from Silver to train a ML model
# MAGIC 5. **Apply the ML model to a different Delta Table and store the results in a Silver table**
# MAGIC 6. **Aggregate data from Silver Table and store it in Gold Table**
# MAGIC 7. Explore Gold Table using Databricks SQL
# MAGIC 
# MAGIC In this notebook we will cover 5-6

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLFLow
# MAGIC 
# MAGIC ![Managed MLFlow](https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg)

# COMMAND ----------

#Let's grab the model
import mlflow

selected_model = mlflow.pyfunc.load_model("models:/caio_demo_airbnb_pricing/production")
apply_model_udf = mlflow.pyfunc.spark_udf(spark, "models:/caio_demo_airbnb_pricing/production", result_type = "float") #always a good idea explicitly telling what the outcome of the prediction will be

# COMMAND ----------

# Now let's include the predicted price to our silver table

listings_silver_df = spark.table("caio_demo_airbnb.listings_silver").filter()

# COMMAND ----------

display(listings_silver_df)

# COMMAND ----------

#Now let's apply the model
from pyspark.sql.functions import struct

#If not struct is defined, data will be passed to the udf as a dataframe with column names as 0,1,2... Hence the importance of naming the columns like this
udf_inputs = struct(*(listings_silver_df.columns))

listings_silver_df = listings_silver_df.withColumn("predicted_price", apply_model_udf(udf_inputs))

# COMMAND ----------

display(listings_silver_df)

# COMMAND ----------

listings_silver_df.createOrReplaceTempView("listings_silver_update")

# COMMAND ----------

# MAGIC %sql
# MAGIC ALTER TABLE caio_demo_airbnb.listings_silver
# MAGIC ADD COLUMN predicted_price DOUBLE

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO caio_demo_airbnb.listings_silver
# MAGIC USING listings_silver_update
# MAGIC ON listings_silver.id = listings_silver_update.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET listings_silver.predicted_price = listings_silver_update.predicted_price

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM caio_demo_airbnb.listings_silver

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now let's create a gold table
# MAGIC DROP TABLE IF EXISTS caio_demo_airbnb.listings_gold;
# MAGIC CREATE TABLE IF NOT EXISTS caio_demo_airbnb.listings_gold
# MAGIC USING DELTA
# MAGIC AS SELECT host_is_superhost, cancellation_policy, instant_bookable, neighbourhood_cleansed, property_type, room_type, accommodates, bathrooms, bedrooms, SUM(price) sum_price, SUM(predicted_price) sum_predicted_price, COUNT(1) AS num_listings
# MAGIC    FROM caio_demo_airbnb.listings_silver
# MAGIC    GROUP BY host_is_superhost, cancellation_policy, instant_bookable, neighbourhood_cleansed, property_type, room_type, accommodates, bathrooms, bedrooms
