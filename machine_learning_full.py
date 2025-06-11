import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from parameter_grid import models_dict

#Data Loading (After generation)
print("Machine learning with all features, with the changed deeprank!")
data = pd.read_csv('all_merged_v3.csv')
data.head()
data['pIspG'] = data['pIspG'].replace({'+/-': '+'})
data['pIspG'] = data['pIspG'].replace({'+': 1, '-': 0})

missing_per_column = data.isna().sum()
if missing_per_column.any():
    print('There are missing points in the data we have to fill!')
else:
    print('Data is full!')

#top_feats = ['length', 'molecular_weight', 'pid_total', 'irc_negative_positive_sum', 'aa_S', 'res_type_7_pct', 'isoelectric_point', 'aa_L', 'res_charge_sum', 'CL_FLDA_B', 'CL_FLDA_A', 'GO:0046872', 'GO:0005506', 'RMSE', 'aa_Q', 'polarity_2_pct', 'penalty_top50', 'res_type_9_pct', 'aa_R', 'res_charge_negative_sum']
#X = X[top_feats]
X = data.drop(columns=['id','UniProtID','pIspG','organism'])
y = data['pIspG']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
var_thresh = 0.1
keep_cols  = X_scaled.var(axis=0) > var_thresh
X_filtered = X_scaled.loc[:, keep_cols]

dropped = keep_cols[~keep_cols].index.tolist()   # names of low-variance cols
print("\nDropped features (var <", var_thresh, "):")
print(dropped)

print(f"Kept {keep_cols.sum()} / {len(keep_cols)} features "
      f"(var > {var_thresh})")

outer_cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits = 4, shuffle=True, random_state=42)
results = {}
best_models = {}
y_pred_all = np.empty_like(y, dtype=int)

#Full pipeline (first try)
for name, (model, params) in models_dict.items():

    print(f"Grid search started for {name}")
    #grid = GridSearchCV(model, param_grid = params, cv = inner_cv, scoring = "balanced_accuracy", n_jobs = -1)

    #cv_res = cross_validate(grid, X_filtered, y, cv=outer_cv, scoring="balanced_accuracy", return_estimator=True, n_jobs=-1)
    #scores = cross_val_score(grid, X, y, cv = outer_cv, scoring = "balanced_accuracy", n_jobs = -1)

    #scores = cv_res["test_score"] 
    #results[name] = {"mean_accuracy": np.mean(scores), "std_accuracy": np.std(scores)}
    #print(f"Model: {name}, Average Balanced Accuracy: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")

    #y_pred = cross_val_predict(grid, X, y, cv=outer_cv, n_jobs=-1)
    #for (train_idx, test_idx), est in zip(outer_cv.split(X_filtered, y), cv_res["estimator"]):
    #    y_pred_all[test_idx] = est.predict(X_filtered.iloc[test_idx])

    #print("\nClassification Report:")
    #print(classification_report(y, y_pred_all, zero_division=0))

    grid_refit = GridSearchCV(model, param_grid=params,
                              cv=inner_cv,
                              scoring='balanced_accuracy',
                              n_jobs=-1)
    
    grid_refit.fit(X_filtered, y)
    best_models[name] = grid_refit.best_estimator_   # store for SHAP
    print(f"Stored best {name} model for SHAP")

#best_name = max(results, key=lambda k: results[k]["mean_accuracy"])
#print(f"Best - performing model: {best_name} - ({results[best_name]['mean_accuracy']:.4f} +- {results[best_name]['std_accuracy']:.4f})")

#print("\nNested CV Results:")
#for model_name, metrics in results.items():
#    print(model_name, metrics)

print(best_models)
joblib.dump(best_models, "best_models_v2.pkl")
print("Saved one best estimator per model class to artifacts/best_models_v2.pkl")