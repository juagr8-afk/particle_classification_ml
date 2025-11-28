import numpy as np
import pandas as pd
import glob
import joblib
import matplotlib.pyplot as plt

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
DATA_FOLDER = "pkl_data"  # Carpeta con todos los .pkl
# ---------------------------------------

# -------- Cargar todos los PKL --------
all_files = glob.glob(f"{DATA_FOLDER}/*.pkl")
if len(all_files) == 0:
    raise FileNotFoundError(f"No se encontraron archivos .pkl en {DATA_FOLDER}")

df_list = [pd.read_pickle(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)
print(f"Archivos cargados, forma total: {df.shape}")

# -------- Crear target binario --------
# Clase 1: Z (80 <= M <= 100), Clase 0: No-Z
df = df[~df["M"].isna()].copy()  # eliminar NaNs en M
df['Z_class'] = np.where((df['M'] >= 80) & (df['M'] <= 100), 1, 0)

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
joblib.dump(imputer, "imputer_z_pkl.joblib")
joblib.dump(scaler, "scaler_z_pkl.joblib")

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
print(res_df)
res_df.to_csv("classification_metrics_pkl.csv")

# -------- Matriz de Confusión --------
for model_name, pred in zip(['LogisticRegression','RandomForest'],
                            [pred_lr, pred_rf]):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(model_name)
    plt.savefig(f"{model_name}_confusion_pkl.png", dpi=150)
    plt.close()

print("Pipeline completo finalizado. Métricas y matrices guardadas.")
