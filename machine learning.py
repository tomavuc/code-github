import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report

from parameter_grid import models_dict

#Data Loading (After generation)
data = pd.read_csv('final_feature_matrix.csv')
data.head()

missing_per_column = data.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')

X = data.drop(columns=['GenBankID','UniProtID','pIspG','organism']) 
y = data['pIspG']
y = LabelEncoder().fit_transform(y)
print(y)

outer_cv = KFold(n_splits = 3, shuffle=True, random_state=42)
inner_cv = KFold(n_splits = 3, shuffle=True, random_state=42)
results = {}

#Full pipeline (first try)
for name, (model, params) in models_dict.items():
    pipe = Pipeline([('imputer', SimpleImputer(strategy = "mean")), ('scaler', StandardScaler()), ("classifier", model)])

    print(f"Grid search started for {name}")
    grid = GridSearchCV(pipe, param_grid = params, cv = inner_cv, scoring = "accuracy", n_jobs = -1)

    scores = cross_val_score(grid, X, y, cv = outer_cv, scoring = "accuracy", n_jobs = -1)

    results[name] = {"mean_accuracy": np.mean(scores), "std_accuracy": np.std(scores)}
    print(f"Model: {name}, Mean Accuracy: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")


print("\nNested CV Results:")
for model_name, metrics in results.items():
    print(model_name, metrics)

# Generate predictions using cross_val_predict (outer CV)
print(grid)
y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)

print("\nClassification Report:")
print(classification_report(y, y_pred))

