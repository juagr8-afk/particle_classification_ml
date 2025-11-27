# particle_classification_ml

---

## README.md for Z vs Non-Z Classification

```markdown
# Z vs Non-Z Classification in Dielectron Events (CERN)

This project implements a **binary classification pipeline** to identify events that may correspond to a Z boson in dielectron collisions.

- **Class 1 (Z)**: 80 ≤ M ≤ 100 GeV  
- **Class 0 (Non-Z)**: all other events

## Dataset

- `.pkl` files of particle collisions.  
- Main columns:
  - `E1`, `E2`, `px1, py1, pz1, px2, py2, pz2`, `pt1, pt2`, `eta1, eta2`, `phi1, phi2`, `Q1, Q2`  
  - `M`: invariant mass (GeV) — used to create binary target

## Preprocessing

1. Load all `.pkl` files and concatenate into a single DataFrame.  
2. Remove rows with `NaN` in `M`.  
3. Create binary target `Z_class` (1 = Z, 0 = Non-Z).  
4. Impute NaNs in numeric features using the **mean**.  
5. Scale features using `StandardScaler`.  
6. Train/test split 80/20 with stratification.

## Trained Models

| Model               | Accuracy | F1 Score |
|--------------------|---------|---------|
| Logistic Regression | –       | –       |
| Random Forest       | –       | –       |


> Metrics are saved in `classification_metrics_pkl.csv`, and confusion matrices in `*_confusion_pkl.png`.

## Usage

1. Place `.pkl` files in the folder `pkl_data/`.  
2. Run:

```bash
python classify_cern_z_pkl.py
