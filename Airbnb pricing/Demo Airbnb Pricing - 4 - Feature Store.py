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
# MAGIC 5. Apply the ML model to a different Delta Table and store the results in a Silver table
# MAGIC 6. Aggregate data from Silver Table and store it in Gold Table
# MAGIC 7. Explore Gold Table using Databricks SQL
# MAGIC 8. **Feature store**
# MAGIC 
# MAGIC In this notebook we will cover 5-6

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Store
# MAGIC 
# MAGIC ![Databricks Feature Store](https://databricks.com/wp-content/uploads/2021/05/FS-Graphic-1-min.png)

# COMMAND ----------

#Let's grab our data

listings_df = spark.table("caio_demo_airbnb.listings_silver")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Let's create the database
# MAGIC CREATE DATABASE IF NOT EXISTS caio_demo_airbnb_feature_store

# COMMAND ----------

# Now let's include the predicted price to our silver table

from databricks.feature_store import feature_table

def compute_features(data):
  ''' Feature computation code returns a DataFrame with 'customer_id' as primary key'''
  return(data)

# create feature table keyed by customer_id
# take schema from DataFrame output by compute_customer_features
from databricks.feature_store import FeatureStoreClient

listings_features_df = compute_features(listings_df)

fs = FeatureStoreClient()

listings_feature_table = fs.create_feature_table(
  name = 'caio_demo_airbnb_feature_store.listings_features',
  keys = 'id',
  features_df = listings_features_df,
  description = 'Airbnb listings features'
)

# COMMAND ----------

# Now let's read from the feature table
from databricks.feature_store import FeatureStoreClient
fs = FeatureStoreClient()

listings_features_df = fs.read_table(
  name='caio_demo_airbnb_feature_store.listings_features'
)

display(listings_features_df)

# COMMAND ----------

# Now we will start with a dataset with only id and price, and make a feature lookup to create our training set
# First we list all features that we are interested in
from databricks.feature_store import FeatureLookup

listings_features = [x for x in listings_features_df.columns if x not in ['íd','price', 'predicted_price']]

# Now our origin df will be very simple

base_df = spark.sql("""SELECT id AS listing_id, price FROM caio_demo_airbnb.listings_silver""")

feature_lookups = [
    FeatureLookup(
      table_name = 'caio_demo_airbnb_feature_store.listings_features',
      feature_name = x,
      lookup_key = 'listing_id'
    ) for x in listings_features]

training_set = fs.create_training_set(
  df = base_df,
  feature_lookups = feature_lookups,
  label = 'price',
  exclude_columns = ['listing_id', 'id']
)

training_df = training_set.load_df()

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/caio.ishizaka@databricks.com/databricks_automl/caio_demo_airbnbprice_listings_silver")

df_loaded = training_df.toPandas()

target_col = "price"

transformers = []

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

bool_pipeline = Pipeline(steps=[
    ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
    ("imputer", SimpleImputer(missing_values=None, strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

transformers.append(("boolean", bool_pipeline, ['host_is_superhost', 'instant_bookable']))

from sklearn.impute import SimpleImputer

transformers.append(("numerical", SimpleImputer(strategy="mean"), ['bathrooms_na', 'bedrooms_na', 'beds_na', 'review_scores_accuracy_na', 'review_scores_checkin_na', 'review_scores_cleanliness_na', 'review_scores_communication_na', 'review_scores_location_na', 'review_scores_rating_na', 'review_scores_value_na', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'host_total_listings_count', 'latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'review_scores_accuracy', 'review_scores_checkin', 'review_scores_cleanliness', 'review_scores_communication', 'review_scores_location', 'review_scores_rating', 'review_scores_value', 'size_score']))

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

one_hot_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values=None, strategy="constant", fill_value="")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers.append(("onehot", one_hot_pipeline, ['bed_type', 'cancellation_policy', 'neighbourhood_cleansed', 'property_type', 'room_type']))

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

from sklearn.preprocessing import StandardScaler

standardizer = StandardScaler()

from sklearn.model_selection import train_test_split

split_X = df_loaded.drop([target_col], axis=1)
split_y = df_loaded[target_col]

X_train, X_val, y_train, y_val = train_test_split(split_X, split_y, random_state=105547341)

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

set_config(display='diagram')

skrf_regressor = RandomForestRegressor(
  bootstrap=False,
  criterion="mae",
  max_features=0.5592881260127998,
  min_samples_leaf=5,
  min_samples_split=10,
  random_state=56176716,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("standardizer", standardizer),
    ("regressor", skrf_regressor),
])

model

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

import pandas as pd

with mlflow.start_run(run_name="random_forest_regressor_feature_store") as mlflow_run:
    model.fit(X_train, y_train)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    skrf_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val,
                                                                prefix="val_")
    display(pd.DataFrame(skrf_val_metrics, index=[0]))

# COMMAND ----------

# Now we log the model that uses feature store
fs.log_model(
    model,
    "caio_demo_airbnb_pricing_fs",
    flavor=mlflow.sklearn,
    training_set=training_set,
    registered_model_name="caio_demo_airbnb_pricing_fs"
  )

# COMMAND ----------

# Now let's see how to score using the model we registered

model_uri = "models:/caio_demo_airbnb_pricing_fs/production"

predictions = fs.score_batch(
    model_uri,
    base_df.select("listing_id")
)

display(predictions)
