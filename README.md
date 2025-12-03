# particle_classification_ml

# Z Boson Decay Classification events (CERN)

This project implements a **binary classification pipeline** to identify events where two electron candidates with invariant mass near the mass of Z-boson were observed:

- **Class 1 (Zee)**: Events where a Z boson decayed into an electron (e⁻) and a positron (e⁺) 
- **Class 0 (Zmumu)**: Events where a Z boson decayed into a muon (μ⁻) and an anti-muon (μ⁺)


## Dataset

- Taken from Kaggle: Z boson DataSet. CSV file.
- Main columns:


## Preprocessing

1. Load all `.csv` files and concatenate into a single DataFrame  
2. Remove rows with `NaN`.  
3. Create binary target `Z_class` (1 = Zee, 0 = Zmumu)  
4. Impute NaNs in numeric features using the **mean**  
5. Scale features using `StandardScaler`  
6. Train/test split 80/20 

## Trained Models

| Model               | Accuracy | F1 Score |
|--------------------|---------|---------|
| Logistic Regression |  0.59775  |  0.62327  |
| Random Forest       |  0.92925  |  0.93062  |

> Metrics are saved once the code is executed.


