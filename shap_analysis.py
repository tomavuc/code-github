import os
os.environ["OMP_NUM_THREADS"] = "1" 

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from utils import balance_and_split_data

data_path = "all_merged_v2.csv"
models_pkl = "best_models.pkl"
out_dir = "shap_reports"
max_display = 10

def pick_background(X, n=100):
    if len(X) <= n:
        return X
    return shap.kmeans(X, n)

def choose_explainer(model, X_bg, feature_names):
    if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier, XGBClassifier)):
        return shap.TreeExplainer(model) 
    else:
        def to_df(d):
            return pd.DataFrame(np.asarray(d), columns=feature_names)
        if hasattr(model, "predict_proba"):
            predict_fn = lambda d: model.predict_proba(to_df(d))[:, 1]
        else:
            predict_fn = lambda d: model.predict(to_df(d))
        return shap.KernelExplainer(predict_fn, X_bg)

def run_one(model_name: str, pipe, X: pd.DataFrame, out_dir: str, max_display=max_display):
    print(f"SHAP for {model_name}")
    os.makedirs(out_dir, exist_ok=True)

    X_bg = pick_background(X, n=100)
    explainer = choose_explainer(pipe, X_bg, X.columns)
    shap_vals = explainer.shap_values(X)
    
    shap_vals = shap_vals[:,:,1] if shap_vals.ndim==3 else shap_vals
    np.save(os.path.join(out_dir, f"{model_name}_shap_values.npy"), shap_vals)

    vals = np.abs(shap_vals).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance.to_csv(os.path.join(out_dir, f"{model_name}_importance.csv"))
    print(f"Top-5 features for {model_name}:\n{feature_importance.head(5)}\n")

    plt.tight_layout()
    shap.summary_plot(shap_vals, X, feature_names=X.columns,
                    show=True, max_display=max_display, title = f"{model_name} SHAP values", plot_type = 'violin',
                    plot_size=(8, 6))
    shap.savefig(os.path.join(out_dir, f"{model_name}_shap_violin_plot.png"))
    shap.summary_plot(shap_vals, X, plot_type='bar', feature_names=X.columns, max_display=max_display,
                  color="lightgrey",show=True)
    plt.savefig(os.path.join(out_dir, f"{model_name}_shap_bar_plot.png"))

if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df['pIspG'] = df['pIspG'].replace({'+/-': '+'})
    df['pIspG'] = df['pIspG'].replace({'+': 1, '-': 0}).astype(int)
    X = df.drop(columns=['id', 'UniProtID', 'pIspG', 'organism'])
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    var_thresh = 0.1
    keep_cols  = X_scaled.var(axis=0) > var_thresh
    X_filtered = X_scaled.loc[:, keep_cols]

    best = joblib.load(models_pkl)
    print("Loaded", len(best), "pipelines")
    print(best)
    for name, pipe in best.items():
        run_one(name, pipe, X_filtered, out_dir)
