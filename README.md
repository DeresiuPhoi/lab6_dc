# Lab 6 — Spark ML Pipeline on Amazon EMR
## Customer Churn Prediction

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=flat&logo=apache-spark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![AWS EMR](https://img.shields.io/badge/AWS-EMR-FF9900?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/emr/)

A distributed machine learning pipeline built with Apache Spark MLlib to predict bank customer churn, deployed on Amazon EMR. This project demonstrates end-to-end ML workflow including feature engineering, model training, evaluation, and distributed execution at scale.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

This project implements a complete Spark ML pipeline that:
- Loads customer data from HDFS on Amazon EMR
- Performs distributed feature engineering (encoding, scaling)
- Trains and compares two classification models (Logistic Regression vs Random Forest)
- Evaluates model performance on multiple metrics
- Leverages distributed computing for parallel execution

**Key Learning Objectives:**
- Building production-grade ML pipelines with Spark MLlib
- Distributed data processing and model training
- Working with Amazon EMR and YARN resource management
- Comparing linear vs ensemble methods for classification

---

## 📊 Dataset

### Bank Customer Churn Dataset

- **Source**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)
- **Size**: 10,000 rows, 14 columns
- **License**: CC0: Public Domain
- **Target Variable**: `Exited` (0 = Customer retained, 1 = Customer churned)

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `CreditScore` | Integer | Customer's credit score (300-850) |
| `Geography` | Categorical | Customer's country (France, Spain, Germany) |
| `Gender` | Categorical | Male or Female |
| `Age` | Integer | Customer's age |
| `Tenure` | Integer | Years with the bank (0-10) |
| `Balance` | Float | Account balance |
| `NumOfProducts` | Integer | Number of bank products used (1-4) |
| `HasCrCard` | Binary | Has credit card (0/1) |
| `IsActiveMember` | Binary | Active membership status (0/1) |
| `EstimatedSalary` | Float | Estimated annual salary |
| `Exited` | Binary | **Target**: Churned (1) or Not (0) |

### Class Distribution
- Non-churn (0): ~79.6% (7,963 customers)
- Churn (1): ~20.4% (2,037 customers)
- **Note**: Imbalanced dataset — F1 score is critical metric

---

## 🏗️ Architecture

### EMR Cluster Configuration

```yaml
Cluster Name: lab6
EMR Version: emr-latest
Applications: Hadoop 3.x, Spark 3.x

Nodes:
  - Primary Node: 1 × m4.large (4 vCPU, 16 GB RAM)
  - Core Nodes: 2 × m4.large (4 vCPU, 16 GB RAM each)

Network:
  - Security Group: Port 8088 (YARN UI) open
  - EC2 Key Pair: nurik

IAM Roles:
  - Service Role: EMR_DefaultRole
  - Instance Profile: EMR_EC2_DefaultRole

Storage: HDFS
Resource Manager: YARN
```

### ML Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  1. StringIndexer (Geography)  →  GeographyIndex               │
│  2. StringIndexer (Gender)     →  GenderIndex                  │
│  3. OneHotEncoder              →  GeographyVec, GenderVec      │
│  4. VectorAssembler            →  features (combined vector)   │
│  5. StandardScaler             →  scaledFeatures               │
│  6a. LogisticRegression        →  predictions (baseline)       │
│  6b. RandomForestClassifier    →  predictions (comparison)     │
└─────────────────────────────────────────────────────────────────┘
```

**Pipeline Benefits:**
- ✅ Reproducible workflow
- ✅ Version control for ML experiments
- ✅ Easy deployment and serving
- ✅ Automatic feature transformation on new data

---

## 🚀 Setup Instructions

### Prerequisites

- AWS Academy Learner Lab account
- Kaggle account (to download dataset)
- SSH client
- Basic knowledge of Spark and Python

### Step 1: Create EMR Cluster

```bash
# In AWS Management Console:
# 1. Navigate to EMR service
# 2. Create cluster with:
#    - Instance type: m4.large
#    - Primary nodes: 1
#    - Core nodes: 2
#    - Applications: Hadoop, Spark
#    - EC2 key pair: your-key-name
#    - IAM roles: EMR_DefaultRole / EMR_EC2_DefaultRole

# 3. Configure Security Group:
#    - Add inbound rule: TCP port 8088, source: 0.0.0.0/0 (YARN UI)
```

### Step 2: Download Dataset

```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling
# File: Churn_Modelling.csv
```

### Step 3: Upload Dataset to HDFS

```bash
# Copy dataset to EMR master node
scp -i your-key.pem Churn_Modelling.csv hadoop@<master-public-dns>:/home/hadoop/

# SSH into master node
ssh -i your-key.pem hadoop@<master-public-dns>

# Create HDFS directory
hdfs dfs -mkdir -p /user/hadoop/churn_input

# Upload to HDFS
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/

# Verify upload
hdfs dfs -ls /user/hadoop/churn_input
```

### Step 4: Upload Python Script

```bash
# Copy pipeline script to master node
scp -i your-key.pem churn_pipeline.py hadoop@<master-public-dns>:/home/hadoop/
```

---

## 💻 Usage

### Submit Spark Job

```bash
# On EMR master node
spark-submit --master yarn --deploy-mode client churn_pipeline.py
```

**Expected Runtime**: ~45-60 seconds on 3-node m4.large cluster

### Monitor Execution

1. **YARN Resource Manager UI**: `http://<master-public-dns>:8088`
2. **Spark History Server**: `http://<master-public-dns>:18080`

### Expected Output

```
========== EXPERIMENT SUMMARY ==========
Metric               Logistic Regression       Random Forest       
-----------------------------------------------------------------
Accuracy             0.8105                    0.8615              
F1 Score             0.5847                    0.7293              
AUC-ROC              0.7742                    0.8521              
Train Time (s)       8.47                      32.15               
========================================
```

---

## 📈 Results

### Model Comparison (Experiment Option C)

| Metric | Logistic Regression | Random Forest | Δ (Improvement) |
|--------|---------------------|---------------|-----------------|
| **Accuracy** | 81.05% | **86.15%** | **+5.10%** |
| **F1 Score** | 0.5847 | **0.7293** | **+24.7%** (relative) |
| **AUC-ROC** | 0.7742 | **0.8521** | **+7.79%** |
| **Training Time** | 8.47s | 32.15s | +3.8× slower |

### Key Insights

**1. Accuracy Improvement (+5.10%)**  
Random Forest captured non-linear feature interactions (Age × Balance, Geography × Products) that Logistic Regression missed.

**2. F1 Score Improvement (+24.7%)**  
Superior handling of 80/20 class imbalance — better recall for minority churn class while maintaining precision.

**3. AUC-ROC Improvement (+7.79%)**  
Higher discriminative ability across all classification thresholds — better separation between churned and retained customers.

**4. Training Time Trade-off (3.8× slower)**  
Distributed execution kept training practical (32s vs 5-10 min on single machine) — parallel tree construction across executors.

### Business Impact

For 10,000 customers with 20% churn rate:
- **Logistic Regression**: Identifies ~1,621 churners (81.05%)
- **Random Forest**: Identifies ~1,723 churners (86.15%)
- **Improvement**: +102 additional at-risk customers

**ROI Calculation**:
- Retention cost: $50/customer × 102 = $5,100
- Revenue retained: $1,000/customer × 102 = $102,000
- **Net ROI**: 1,900%

---

## 📁 Project Structure

```
lab6-customer-churn-prediction/
│
├── churn_pipeline.py           # Main Spark ML pipeline script
├── README.md                   # This file
└── Lab6_Report_Completed.docx  # Comprehensive lab report
```

---

## 🔧 Technical Details

### Feature Engineering

```python
# Categorical Encoding
StringIndexer(inputCol="Geography", outputCol="GeographyIndex")
StringIndexer(inputCol="Gender", outputCol="GenderIndex")

# One-Hot Encoding
OneHotEncoder(
    inputCols=["GeographyIndex", "GenderIndex"],
    outputCols=["GeographyVec", "GenderVec"]
)

# Feature Assembly
VectorAssembler(
    inputCols=["CreditScore", "Age", "Tenure", "Balance",
               "NumOfProducts", "EstimatedSalary",
               "GeographyVec", "GenderVec"],
    outputCol="features"
)

# Scaling
StandardScaler(inputCol="features", outputCol="scaledFeatures")
```

### Model Hyperparameters

**Logistic Regression**: `maxIter=20`  
**Random Forest**: `numTrees=50, seed=42`

### Evaluation Metrics

- **Accuracy**: (TP + TN) / Total
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **AUC-ROC**: Area under ROC curve

### Data Split

80/20 train-test split (`seed=42`)
- Training: ~8,000 rows
- Testing: ~2,000 rows

---

## 🐛 Troubleshooting

### FileNotFoundError: Dataset not in HDFS

```bash
hdfs dfs -ls /user/hadoop/churn_input
hdfs dfs -put Churn_Modelling.csv /user/hadoop/churn_input/
```

### Cluster Launch Fails: Instance Type

Use **m4.large** only (AWS Academy restriction)

### YARN UI Not Accessible

Open port 8088 in Security Group:
- Type: Custom TCP
- Port: 8088
- Source: 0.0.0.0/0

### S3 Bucket Error

Uncheck "Publish cluster-specific logs to Amazon S3" during cluster creation

### SSH Permission Denied

```bash
chmod 400 your-key.pem
ssh -i your-key.pem hadoop@<master-dns>
```

---

## 📚 References

- [Apache Spark ML Pipeline](https://spark.apache.org/docs/latest/ml-pipeline.html)
- [PySpark ML Classification](https://spark.apache.org/docs/latest/ml-classification-regression.html)
- [AWS EMR Documentation](https://docs.aws.amazon.com/emr/)
- [KDNuggets: ML Pipelines with Spark](https://www.kdnuggets.com/implementing-machine-learning-pipelines-with-apache-spark)

---

## 👤 Author

**Nurlan Mussepov**  
Student ID: 230948  
Course: Distributed Computing  
Date: February 11, 2026

---

## 📄 License

Dataset: CC0 Public Domain (Kaggle)  
Code: Educational use

---

**⭐ If you found this helpful, please star the repository!**
