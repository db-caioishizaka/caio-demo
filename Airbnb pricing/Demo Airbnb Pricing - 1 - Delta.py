# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Demo
# MAGIC 
# MAGIC In this demo we will demonstrate how to:
# MAGIC 1. **Get data from its source**
# MAGIC 2. **Write that data into Delta Bronze Tables**
# MAGIC 3. **Transform the data from Bronze and write it to Silver**
# MAGIC 4. Use data from Silver to train a ML model
# MAGIC 5. Apply the ML model to a different Delta Table and store the results in a Silver table
# MAGIC 6. Aggregate data from Silver Table and store it in Gold Table
# MAGIC 7. Explore Gold Table using Databricks SQL
# MAGIC 
# MAGIC In this notebook we will cover 1-3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta Lake
# MAGIC 
# MAGIC ![Delta Lake base architecture](https://live-delta-io.pantheonsite.io/wp-content/uploads/2019/04/Delta-Lake-marketecture-0423c.png)

# COMMAND ----------

# Reading the data 
# File location and type
file_location = "dbfs:/FileStore/caio/airbnb/original"
file_type = "parquet"

df = spark.read\
          .format(file_type) \
          .load(file_location)

display(df)

# COMMAND ----------

#Creating a temp view so the table is accessible via SQL
#The temp view is available only in this session. People using the same cluster at the same time will not be able to see this.
df.createOrReplaceTempView("airbnb")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Creating the table if it not exists. In case table already exists, it will go to the next step
# MAGIC CREATE DATABASE IF NOT EXISTS caio_demo_airbnb;
# MAGIC CREATE TABLE IF NOT EXISTS caio_demo_airbnb.listings_bronze
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (host_is_superhost)
# MAGIC AS SELECT * FROM airbnb

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Once table is created, it is available for everyone to see
# MAGIC select 
# MAGIC   host_is_superhost, 
# MAGIC   COUNT(1) 
# MAGIC from 
# MAGIC   caio_demo_airbnb.listings_bronze 
# MAGIC GROUP BY 
# MAGIC   host_is_superhost

# COMMAND ----------

# Now let's grab some new data that has arrived to update the listings_bronze table

file_location_new = "dbfs:/FileStore/caio/airbnb/updated"
file_type_new = "parquet"

df_new = spark.read\
              .format(file_type_new) \
              .load(file_location_new)

display(df_new)

df_new.createOrReplaceTempView("airbnb_new")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's merge the new data to Bronze Table using sql
# MAGIC MERGE INTO caio_demo_airbnb.listings_bronze
# MAGIC USING airbnb_new
# MAGIC ON listings_bronze.id = airbnb_new.id
# MAGIC WHEN MATCHED THEN
# MAGIC   UPDATE SET *
# MAGIC WHEN NOT MATCHED
# MAGIC   THEN INSERT *

# COMMAND ----------

# Let's create a size score, based on how many people it accommodates, number of bathrooms, bedrooms and beds
# Using pyspark instead of sql for a change
from pyspark.sql.functions import col
airbnb_listings_bronze = spark.table("caio_demo_airbnb.listings_bronze")

airbnb_listings_bronze_processed = airbnb_listings_bronze.withColumn("size_score", col("accommodates") + col("bathrooms") + col("bedrooms") + col("beds"))
airbnb_listings_bronze_processed.createOrReplaceTempView("airbnb_listings_bronze_processed")
display(airbnb_listings_bronze_processed)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Creating the table if it not exists. In case table already exists, it will go to the next step
# MAGIC CREATE TABLE IF NOT EXISTS caio_demo_airbnb.listings_silver
# MAGIC USING DELTA
# MAGIC PARTITIONED BY (host_is_superhost)
# MAGIC AS SELECT * FROM airbnb_listings_bronze_processed

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's deep dive in file location and time travel

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's explore the delta table and where it is located
# MAGIC DESCRIBE EXTENDED caio_demo_airbnb.listings_bronze

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse/caio_demo_airbnb.db/listings_bronze

# COMMAND ----------

# MAGIC %fs ls dbfs:/user/hive/warehouse/caio_demo_airbnb.db/listings_bronze/_delta_log

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's see the table history now
# MAGIC DESCRIBE HISTORY caio_demo_airbnb.listings_bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- let's compare #records on each version
# MAGIC SELECT 'last_version' as version, COUNT(1) AS num_records FROM caio_demo_airbnb.listings_bronze
# MAGIC UNION ALL
# MAGIC SELECT '0' as version, COUNT(1) AS num_records FROM caio_demo_airbnb.listings_bronze VERSION AS OF 0

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Now giving a timestamp
# MAGIC SELECT '2021-07-05 15:10' as timestamp, COUNT(1) AS num_records FROM caio_demo_airbnb.listings_bronze TIMESTAMP AS OF "2021-07-05 15:10"
