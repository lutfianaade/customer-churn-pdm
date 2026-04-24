import pandas as pd
from imblearn.over_sampling import SMOTE

# load data
df = pd.read_csv('dataset/raw/Telco-Customer-Churn.csv')

# ubah tipe data
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# hapus missing value
df = df.dropna()

# hapus kolom tidak penting
df = df.drop('customerID', axis=1)

# encoding
df = pd.get_dummies(df, drop_first=True)

# pisah fitur & target
X = df.drop('Churn_Yes', axis=1)
y = df['Churn_Yes'].astype(int)

# SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# gabung lagi
df_clean = pd.concat(
    [pd.DataFrame(X_res, columns=X.columns),
     pd.DataFrame(y_res, columns=['Churn_Yes'])],
    axis=1
)

# simpan
df_clean.to_csv('clean_data.csv', index=False)
