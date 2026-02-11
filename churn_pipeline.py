"""
Lab 6 — Spark ML Pipeline on Amazon EMR
Customer Churn Prediction
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import time

# ── 1. Spark Session ──────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("CustomerChurnPipeline") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("Spark version:", spark.version)

# ── 2. Load Data from HDFS ────────────────────────────────────────────────────
data = spark.read.csv(
    "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
    header=True,
    inferSchema=True
)

print("Dataset schema:")
data.printSchema()
print("Total rows:", data.count())
data.show(5)

# ── 3. Drop Unnecessary Columns ───────────────────────────────────────────────
data = data.drop("RowNumber", "CustomerId", "Surname")

# ── 4. Train / Test Split ─────────────────────────────────────────────────────
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
print(f"Train rows: {train_data.count()} | Test rows: {test_data.count()}")

# ── 5. Pipeline Stages ────────────────────────────────────────────────────────
geo_indexer    = StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
gender_indexer = StringIndexer(inputCol="Gender",    outputCol="GenderIndex")

encoder = OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec",   "GenderVec"]
)

assembler = VectorAssembler(
    inputCols=[
        "CreditScore", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary",
        "GeographyVec", "GenderVec"
    ],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=False,
    withStd=True
)

lr = LogisticRegression(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    maxIter=20
)

pipeline_lr = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    lr
])

# ── 6. Train Logistic Regression ──────────────────────────────────────────────
print("\n=== Training Logistic Regression ===")
start = time.time()
model_lr = pipeline_lr.fit(train_data)
lr_train_time = time.time() - start
print(f"LR training time: {lr_train_time:.2f}s")

predictions_lr = model_lr.transform(test_data)
predictions_lr.select("Exited", "prediction", "probability").show(10)

# ── 7. Evaluate LR ───────────────────────────────────────────────────────────
acc_eval = MulticlassClassificationEvaluator(
    labelCol="Exited", predictionCol="prediction", metricName="accuracy"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="Exited", predictionCol="prediction", metricName="f1"
)
auc_eval = BinaryClassificationEvaluator(
    labelCol="Exited", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

lr_accuracy = acc_eval.evaluate(predictions_lr)
lr_f1       = f1_eval.evaluate(predictions_lr)
lr_auc      = auc_eval.evaluate(predictions_lr)

print(f"\n--- Logistic Regression Results ---")
print(f"Accuracy : {lr_accuracy:.4f}")
print(f"F1 Score : {lr_f1:.4f}")
print(f"AUC-ROC  : {lr_auc:.4f}")
print(f"Train Time: {lr_train_time:.2f}s")

# ── 8. Experiment — Option C: Random Forest ───────────────────────────────────
print("\n=== Training Random Forest ===")
rf = RandomForestClassifier(
    labelCol="Exited",
    featuresCol="scaledFeatures",
    numTrees=50,
    seed=42
)

pipeline_rf = Pipeline(stages=[
    geo_indexer,
    gender_indexer,
    encoder,
    assembler,
    scaler,
    rf
])

start = time.time()
model_rf = pipeline_rf.fit(train_data)
rf_train_time = time.time() - start
print(f"RF training time: {rf_train_time:.2f}s")

predictions_rf = model_rf.transform(test_data)

rf_accuracy = acc_eval.evaluate(predictions_rf)
rf_f1       = f1_eval.evaluate(predictions_rf)
rf_auc      = auc_eval.evaluate(predictions_rf)

print(f"\n--- Random Forest Results ---")
print(f"Accuracy : {rf_accuracy:.4f}")
print(f"F1 Score : {rf_f1:.4f}")
print(f"AUC-ROC  : {rf_auc:.4f}")
print(f"Train Time: {rf_train_time:.2f}s")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n========== EXPERIMENT SUMMARY ==========")
print(f"{'Metric':<20} {'Logistic Regression':<25} {'Random Forest':<20}")
print("-" * 65)
print(f"{'Accuracy':<20} {lr_accuracy:<25.4f} {rf_accuracy:<20.4f}")
print(f"{'F1 Score':<20} {lr_f1:<25.4f} {rf_f1:<20.4f}")
print(f"{'AUC-ROC':<20} {lr_auc:<25.4f} {rf_auc:<20.4f}")
print(f"{'Train Time (s)':<20} {lr_train_time:<25.2f} {rf_train_time:<20.2f}")
print("========================================")

spark.stop()
