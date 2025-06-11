from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats

def _load_df(csv_or_df):
    if isinstance(csv_or_df, pd.DataFrame):
        return csv_or_df.copy(deep=True)
    return pd.read_csv(csv_or_df)

def welch_t_test_feature_selection(csv_path, label_col= "pIspG", pos_label = "+", neg_label = "-", id_col = "id", exclude_ids = None, exclude_cols = None, top_k = 10):
    df = _load_df(csv_path)

    if exclude_ids is not None:
        df = df[~df[id_col].isin(exclude_ids)]

    df[label_col] = df[label_col].replace({"+/-": pos_label})
    df[label_col] = df[label_col].replace({pos_label: 1, neg_label: 0})
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]

    if exclude_cols is None:
        exclude_cols = []

    feature_cols = df.select_dtypes(include=[np.number]).columns.difference([label_col] + list(exclude_cols))
    t_stats = []
    p_values = []

    for col in feature_cols:
        t, p = stats.ttest_ind(pos_df[col].dropna(), neg_df[col].dropna(), equal_var=False)
        t_stats.append(t)
        p_values.append(p)

    results = pd.DataFrame({"feature": feature_cols, "t_stat": t_stats, "p_value": p_values}).assign(abs_t=lambda d: d["t_stat"].abs())
    results = results.sort_values("abs_t", ascending=False).reset_index(drop=True)

    top_features = []
    if top_k is not None and top_k > 0:
        top_features = results.head(top_k)["feature"].tolist()

    return results, top_features

stats_df, top = welch_t_test_feature_selection("all_merged_v3.csv", id_col="id", top_k=10)
print("Top 10 probes:", top)
