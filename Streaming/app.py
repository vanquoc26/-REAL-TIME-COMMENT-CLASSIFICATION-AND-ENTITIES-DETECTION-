import os
os.environ['HADOOP_HOME'] = "C:\\hadoop"
os.environ['PATH'] += ";C:\\hadoop\\bin"

import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import StringType
import json
import joblib
import streamlit as st
from time import sleep
from IPython.display import clear_output

# Streamlit title
st.title('Real-time Comment Classification and Entities Detection')

# Kafka and Spark configuration
scala_version = '2.12'
spark_version = '3.5.1'
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:3.3.1'
]

spark = SparkSession \
    .builder \
    .appName("BigSale") \
    .master("local") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .config("spark.python.worker.reuse", "true") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.execution.arrow.maxRecordsPerBatch", "16") \
    .config("spark.jars.packages", ",".join(packages)) \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Kafka topic and server configuration
topic_name = 'doanBD_3'
kafka_server = 'localhost:9092'
streamRawDf = spark.readStream.format("kafka").option("kafka.bootstrap.servers", kafka_server).option("subscribe", topic_name).option("startingOffsets", "latest").load()

# Load the saved model components
dct_objects = joblib.load('model/all_objects.pkl')
dct_model = joblib.load('model/multi_target_lr_model.pkl')
dct_vectorizer = joblib.load('model/vectorizer.pkl')

clf_model = joblib.load('model/random_forest_model.pkl')
clf_vectorizer = joblib.load('model/rf_vectorizer.pkl')

# Function to predict entities in a comment
def predict_entities(comment, model, vectorizer, all_objects):
    X_new = vectorizer.transform([comment])
    Y_new_pred = model.predict(X_new)
    entities = [all_objects[i] for i in range(len(all_objects)) if Y_new_pred[0, i] == 1]
    return ', '.join(entities)

def classify_comment(comment, model, vectorizer):    
    X_new = vectorizer.transform([comment])
    Y_new_pred = model.predict(X_new)
    predict = int(Y_new_pred[0])
    return predict

# Register UDF
predict_entities_udf = f.udf(lambda comment: predict_entities(comment, dct_model, dct_vectorizer, dct_objects), StringType())
classify_comment_udf = f.udf(lambda comment: classify_comment(comment, clf_model, clf_vectorizer), StringType())

# Decode JSON column
def decode_json_column(col):
    return f.udf(lambda x: json.loads(x), StringType())(col)

# Process the streaming data
df = (streamRawDf
      .selectExpr("CAST(value AS STRING) as value")
      .withColumn("Comment", decode_json_column("value")))

# Write to memory table
stream_writer = (df.writeStream
                 .trigger(processingTime="5 seconds")
                 .outputMode("append")
                 .format("memory")
                 .queryName("Table"))

query = stream_writer.start()

# Placeholder for clearing and updating content
placeholder = st.empty()

# Live view loop
x = 0
while True:
    try:
        # Query the in-memory table
        result_df = spark.sql("SELECT * FROM Table")
        result_df = result_df.withColumn("Entities", predict_entities_udf(f.col("Comment")))
        result_df = result_df.withColumn("Classified", classify_comment_udf(f.col("Comment")))
        result_df = result_df.select("Comment", "Entities", "Classified")
        all_rows = result_df.collect()

        last_20_rows = all_rows[-20:]

        tail_df = spark.createDataFrame(all_rows, result_df.schema)
        tail_pdf = tail_df.toPandas()

        # Update the placeholder content
        with placeholder.container():
            st.write('Showing live view update every 30 seconds')
            st.dataframe(tail_pdf, width=800)

        sleep(5)
    except KeyboardInterrupt:
        st.write("Streaming query interrupted.")
        break
