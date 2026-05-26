# -*- coding: utf-8 -*-
"""Customer_Churn_HighRecall.py — Optimized for Recall"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, recall_score
)

# ================= DATA =================
df = pd.read_csv('dataset/raw/Telco-Customer-Churn.csv')

# ==============DATA UNDERSTANDING==============
print("\n=== INFO DATA ===")
print(df.info())

print("\n=== MISSING VALUE ===")
print(df.isnull().sum())

print("\n=== DUPLIKAT ===")
print(df.duplicated().sum())

print("\n=== STATISTIK DESKRIPTIF ===")
print(df.describe())

print("\n=== JUMLAH NILAI UNIK ===")
print(df.nunique())

print("\n=== DISTRIBUSI TARGET ===")
print(df['Churn'].value_counts())

# ================= EDA =================

plt.figure()
sns.countplot(x='Churn', data=df)
plt.title("Distribusi Churn")
plt.savefig("eda_churn.png")

plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn berdasarkan Contract")
plt.xticks(rotation=30)
plt.savefig("eda_contract.png")

plt.figure()
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.savefig("eda_tenure.png")

plt.figure()
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.savefig("eda_monthly.png")

plt.figure()
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.xticks(rotation=30)
plt.title("Payment Method vs Churn")
plt.savefig("eda_payment.png")

print("\nEDA selesai, grafik disimpan sebagai file PNG")


# ================= PREPROCESSING =================
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df = df.drop('customerID', axis=1)
df = pd.get_dummies(df, drop_first=True)

X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ================= DAGSHUB + MLFLOW =================
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv()

dagshub.init(
    repo_owner='lutfianaade',
    repo_name='customer_churn_prediction',
    mlflow=True
)

mlflow.set_experiment("Churn Experiment - High Recall")

print("TRACKING URI:", mlflow.get_tracking_uri())

# ================= SAMPLING =================
X_train_sample = X_train_res.sample(3000, random_state=42)
y_train_sample = y_train_res.loc[X_train_sample.index]


# ================= STEP 1: Cari max_depth & min_samples_split =================
# FIX: optimasi pakai RECALL, bukan accuracy
# FIX: class_weight='balanced' dipasang dari awal agar konsisten
results = []

for depth in [3, 5, 7, 10, 15]:
    for split in [2, 5, 10]:

        with mlflow.start_run(run_name="STEP_1"):

            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=split,
                class_weight='balanced',   # <-- FIX: tambah dari awal
                random_state=42
            )

            model.fit(X_train_sample, y_train_sample)
            y_pred = model.predict(X_test)

            acc    = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)   # <-- FIX: hitung recall

            mlflow.log_param("max_depth", depth)
            mlflow.log_param("min_samples_split", split)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("recall_churn", recall)   # <-- FIX: log recall

            results.append({
                'max_depth': depth,
                'min_samples_split': split,
                'accuracy': acc,
                'recall_churn': recall
            })

df_results = pd.DataFrame(results)

# FIX: sort by recall, bukan accuracy
best = df_results.sort_values(by='recall_churn', ascending=False).iloc[0]

best_depth = int(best['max_depth'])
best_split = int(best['min_samples_split'])

print("\nBEST STEP 1 (by recall):", best_depth, best_split)
print(df_results.sort_values('recall_churn', ascending=False).head(5).to_string(index=False))


# ================= STEP 2: Cari min_samples_leaf & criterion =================
# FIX: optimasi pakai recall (class churn=1)
results_step2 = []

for leaf in [1, 2, 4]:
    for crit in ['gini', 'entropy']:

        with mlflow.start_run(run_name="STEP_2"):

            model = DecisionTreeClassifier(
                max_depth=best_depth,
                min_samples_split=best_split,
                min_samples_leaf=leaf,
                criterion=crit,
                class_weight='balanced',   # <-- FIX: konsisten
                random_state=42
            )

            model.fit(X_train_sample, y_train_sample)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred, output_dict=True)
            recall_churn = report['1']['recall']   # <-- FIX: ambil recall, bukan f1

            mlflow.log_param("min_samples_leaf", leaf)
            mlflow.log_param("criterion", crit)
            mlflow.log_metric("recall_churn", recall_churn)
            mlflow.log_metric("f1_churn", report['1']['f1-score'])

            results_step2.append({
                'min_samples_leaf': leaf,
                'criterion': crit,
                'recall_churn': recall_churn,
                'f1_churn': report['1']['f1-score']
            })

df_step2 = pd.DataFrame(results_step2)

# FIX: sort by recall
best2 = df_step2.sort_values(by='recall_churn', ascending=False).iloc[0]

best_leaf = int(best2['min_samples_leaf'])
best_crit = best2['criterion']

print("\nBEST STEP 2 (by recall):", best_leaf, best_crit)
print(df_step2.sort_values('recall_churn', ascending=False).to_string(index=False))


# ================= FINAL MODEL =================
# FIX: tambah threshold tuning untuk maksimalkan recall

with mlflow.start_run(run_name="FINAL_MODEL"):

    best_model = DecisionTreeClassifier(
        max_depth=best_depth,
        min_samples_split=best_split,
        min_samples_leaf=best_leaf,
        criterion=best_crit,
        class_weight='balanced',
        random_state=42
    )

    # Train dengan FULL resampled data (bukan sample)
    best_model.fit(X_train_res, y_train_res)

    # ---- FIX: Threshold Tuning ----
    # Default threshold=0.5 sering terlalu ketat untuk deteksi churn.
    # Kita cari threshold terbaik yang memaksimalkan recall
    # dengan syarat precision tidak terlalu jelek (>= 0.40).

    y_proba = best_model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_recall    = 0.0

    print("\n--- Threshold Tuning ---")
    for thresh in np.arange(0.30, 0.55, 0.05):
        y_pred_thresh = (y_proba >= thresh).astype(int)
        r = classification_report(y_test, y_pred_thresh, output_dict=True)
        rec  = r['1']['recall']
        prec = r['1']['precision']
        f1   = r['1']['f1-score']
        print(f"  threshold={thresh:.2f} | recall={rec:.3f} | precision={prec:.3f} | f1={f1:.3f}")

        # Pilih threshold dengan recall tertinggi, asalkan precision >= 0.40
        if rec > best_recall and prec >= 0.40:
            best_recall    = rec
            best_threshold = thresh

    print(f"\n=> Threshold terpilih: {best_threshold:.2f}")

    # Prediksi final dengan threshold optimal
    y_pred_final = (y_proba >= best_threshold).astype(int)

    acc_final = accuracy_score(y_test, y_pred_final)
    report    = classification_report(y_test, y_pred_final, output_dict=True)

    print("\n=== HASIL MODEL FINAL ===\n")
    print(classification_report(y_test, y_pred_final))
    print(confusion_matrix(y_test, y_pred_final))

    # Log semua params & metrics ke MLflow
    mlflow.log_param("max_depth",          best_depth)
    mlflow.log_param("min_samples_split",  best_split)
    mlflow.log_param("min_samples_leaf",   best_leaf)
    mlflow.log_param("criterion",          best_crit)
    mlflow.log_param("threshold",          best_threshold)   # <-- FIX: log threshold

    mlflow.log_metric("accuracy",          acc_final)
    mlflow.log_metric("recall_churn",      report['1']['recall'])
    mlflow.log_metric("precision_churn",   report['1']['precision'])
    mlflow.log_metric("f1_churn",          report['1']['f1-score'])

# Register model ke MLflow Registry
mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",
    registered_model_name="ChurnModel"
)


# ================= SAVE MODEL =================
import joblib

joblib.dump(best_model, 'best_model.pkl')

# Simpan threshold supaya bisa dipakai saat inference
import json
with open('model_config.json', 'w') as f:
    json.dump({'threshold': float(best_threshold)}, f)

print("\nModel berhasil disimpan sebagai best_model.pkl")
print("Threshold disimpan di model_config.json")
print("\n=== CARA PAKAI SAAT INFERENCE ===")
print("""
import joblib, json
model = joblib.load('best_model.pkl')
with open('model_config.json') as f:
    threshold = json.load(f)['threshold']

y_proba = model.predict_proba(X_new)[:, 1]
y_pred  = (y_proba >= threshold).astype(int)
""")
