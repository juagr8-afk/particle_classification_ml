import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

RND = 42
TEST_SIZE = 0.2

print("Upload zip file with CSV files")

try:
    from google.colab import files
    uploaded = files.upload()
    zip_filename = list(uploaded.keys())[0]
    print(f"Uploaded file: {zip_filename}")
except:
    zip_filename = input("Enter file directory: ")

df_list = []

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {zip_filename}")

    # Load each CSV file
    for i, csv_file in enumerate(csv_files):
        with zip_ref.open(csv_file) as f:
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)
            print(f"Loaded ({i+1}/{len(csv_files)}): {csv_file} - Shape: {df_temp.shape}")

df = pd.concat(df_list, ignore_index=True)
print(f"\nAll files loaded. Total shape: {df.shape}")


# Create binary target using the 'class' column
# Class 1: Zee, Class 0: Zmumu
class_mapping = {'Zee': 1, 'Zmumu': 0}
df['target_class'] = df['class'].map(class_mapping)

# Separate features and target
y = df['target_class'].values
X = df.drop(columns=['class', 'target_class'])  # Remove unused columns


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RND, stratify=y
)
print(f"\nTrain/test split: Train {X_train.shape}, Test {X_test.shape}")

# Imputation of missing values
num_cols = X.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train[num_cols])
X_test_imputed = imputer.transform(X_test[num_cols])

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Save imputer and scaler
joblib.dump(imputer, "imputer_zee_zmumu.joblib")
joblib.dump(scaler, "scaler_zee_zmumu.joblib")


results = {}

# 1) Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RND)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
results['LogisticRegression'] = (accuracy_score(y_test, pred_lr), f1_score(y_test, pred_lr))

# 2) Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
pred_rf = rf.predict(X_test_scaled)
results['RandomForest'] = (accuracy_score(y_test, pred_rf), f1_score(y_test, pred_rf))

# results
res_df = pd.DataFrame(results, index=['Accuracy','F1']).T
print("\nRESULTS:")
print(res_df)
res_df.to_csv("classification_metrics_zee_zmumu.csv")

# Confusion matrices
for model_name, pred in zip(['LogisticRegression','RandomForest'],
                            [pred_lr, pred_rf]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Zmumu (0)','Zee (1)'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} - Zee vs Zmumu")
    plt.savefig(f"{model_name}_confusion_zee_zmumu.png", dpi=150)
    plt.close()

print("\nCreated files:")
print("- classification_metrics_zee_zmumu.csv")
print("- LogisticRegression_confusion_zee_zmumu.png")
print("- RandomForest_confusion_zee_zmumu.png")
print("- imputer_zee_zmumu.joblib")
print("- scaler_zee_zmumu.joblib")
