import numpy as np
import pandas as pd
import glob
import joblib
import matplotlib.pyplot as plt
import zipfile
import os
from io import StringIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------
RND = 42
TEST_SIZE = 0.2

# ---------------------------------------

# -------- Subir archivo ZIP --------
print("=== SUBIR ARCHIVO ZIP CON CSVs ===")
print("Por favor, sube tu archivo ZIP que contiene los CSVs")

try:
    from google.colab import files
    print("Entorno: Google Colab")
    uploaded = files.upload()
    zip_filename = list(uploaded.keys())[0]
    print(f"Archivo subido: {zip_filename}")
except:
    print("Entorno: Local")
    zip_filename = input("Introduce la ruta completa del archivo ZIP: ")

# -------- Cargar desde ZIP --------
if not os.path.exists(zip_filename):
    raise FileNotFoundError(f"No se encontró el archivo {zip_filename}")

df_list = []

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    # Listar todos los archivos CSV en el ZIP
    csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {zip_filename}")
    
    print(f"Encontrados {len(csv_files)} archivos CSV en el ZIP")
    
    # Cargar cada CSV
    for i, csv_file in enumerate(csv_files):
        with zip_ref.open(csv_file) as f:
            # Leer el contenido y cargarlo como DataFrame
            df_temp = pd.read_csv(f)
            df_list.append(df_temp)
            print(f"Cargado ({i+1}/{len(csv_files)}): {csv_file} - Forma: {df_temp.shape}")

df = pd.concat(df_list, ignore_index=True)
print(f"\nTodos los archivos cargados, forma total: {df.shape}")

# -------- Crear target binario --------
# Clase 1: Z (80 <= M <= 100), Clase 0: No-Z
df = df[~df["M"].isna()].copy()  # eliminar NaNs en M
df['Z_class'] = np.where((df['M'] >= 80) & (df['M'] <= 100), 1, 0)

print(f"Distribución de clases: {df['Z_class'].value_counts()}")

y = df['Z_class'].values
X = df.drop(columns=['M','Z_class'])

# -------- Imputación de NaNs --------
num_cols = X.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
X[num_cols] = imputer.fit_transform(X[num_cols])

# -------- Escalado --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[num_cols])

# Guardar scaler e imputador
joblib.dump(imputer, "imputer_z_csv.joblib")
joblib.dump(scaler, "scaler_z_csv.joblib")

# -------- Split train/test --------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RND, stratify=y
)
print("Split train/test:", X_train.shape, X_test.shape)

# -------- Modelos --------
results = {}

# 1) Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=RND)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results['LogisticRegression'] = (accuracy_score(y_test, pred_lr), f1_score(y_test, pred_lr))

# 2) Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=RND, n_jobs=-1)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results['RandomForest'] = (accuracy_score(y_test, pred_rf), f1_score(y_test, pred_rf))

# -------- Resultados --------
res_df = pd.DataFrame(results, index=['Accuracy','F1']).T
print("\n" + "="*50)
print("RESULTADOS:")
print("="*50)
print(res_df)
res_df.to_csv("classification_metrics_csv.csv")

# -------- Matriz de Confusión --------
for model_name, pred in zip(['LogisticRegression','RandomForest'],
                            [pred_lr, pred_rf]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(model_name)
    plt.savefig(f"{model_name}_confusion_csv.png", dpi=150)
    plt.close()

print("\n" + "="*50)
print("Pipeline completo finalizado!")
print("Archivos generados:")
print("- classification_metrics_csv.csv")
print("- LogisticRegression_confusion_csv.png") 
print("- RandomForest_confusion_csv.png")
print("- imputer_z_csv.joblib")
print("- scaler_z_csv.joblib")
print("="*50)
