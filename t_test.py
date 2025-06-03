import pandas as pd
import numpy as np
import scipy.stats as st

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path


def welch_t_test_feature_selection(
    csv_path: str | Path,
    label_col: str = "pIspG",
    pos_label: str = "+",
    neg_label: str = "-",
    top_k: int = 10,
    exclude_cols: list[str] | None = None):

    # ---------- Load & tidy labels ----------
    df = pd.read_csv(csv_path)
    df[label_col] = (
        df[label_col]
        .replace({"+/-": pos_label})
        .replace({pos_label: 1, neg_label: 0})
        .astype(int))

    # ---------- Determine candidate feature columns ----------
    if exclude_cols is None:
        exclude_cols = ["id", "UniProtID", "organism", label_col]
    else:
        exclude_cols = list(set(exclude_cols + [label_col]))

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Keep only numeric columns (Welch’s t requires continuous vars)
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    rows: list[tuple[str, float, float]] = []
    y = df[label_col]
    for col in feature_cols:
        g0 = df.loc[y == 0, col].dropna()
        g1 = df.loc[y == 1, col].dropna()
        # Skip degenerate cases where one class has <2 samples
        if min(len(g0), len(g1)) < 2:
            continue
        t_stat, p_val = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        rows.append((col, t_stat, p_val))

    stats_df = (
        pd.DataFrame(rows, columns=["feature", "t", "p"])  # type: ignore[arg-type]
        .assign(abs_t=lambda d: d["t"].abs())
        .sort_values("abs_t", ascending=False)
        .reset_index(drop=True)
    )

    return stats_df, stats_df.head(top_k)

all_stats, top_stats = welch_t_test_feature_selection("all_merged.csv", top_k=10)
print("Top 2 features by |t|:\n", list(top_stats['feature']))
