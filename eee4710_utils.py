"""
eee4710_utils.py — helper functions for the EEE 4710 AI & ML Lab (IUT).

Import in Colab:
    !git clone -q https://github.com/<INSTRUCTOR_USERNAME>/eee4710-lab-utils.git
    import sys; sys.path.append("eee4710-lab-utils")
    import eee4710_utils as eu

Every function has a docstring: run help(eu.<function_name>) to read it.
"""

import io
import os
import random
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

__version__ = "1.0"

GRID_URL = ("https://archive.ics.uci.edu/static/public/471/"
            "electrical+grid+stability+simulated+data.zip")
DATASETS = ("grid_stability", "digits", "breast_cancer")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed=42):
    """Fix the random seeds of Python and NumPy so results are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


# --------------------------------------------------------------------------
# Loading data
# --------------------------------------------------------------------------
def load_dataset(name="grid_stability", path=None, verbose=True):
    """Load one of the lab datasets and return it as a pandas DataFrame.

    Parameters
    ----------
    name : str
        "grid_stability" : UCI Electrical Grid Stability Simulated Data (10,000 rows).
                           Targets: 'stab' (regression) and 'stabf' (classification).
        "digits"         : 8x8 handwritten digits from scikit-learn (1,797 rows), target 'target'.
        "breast_cancer"  : Wisconsin breast cancer data from scikit-learn, target 'target'.
    path : str, optional
        Path to a CSV you already downloaded (e.g. on Google Drive). Skips the download.
    verbose : bool
        Print the shape of the loaded data.
    """
    name = name.lower()
    if path is not None:
        df = pd.read_csv(path)
    elif name in ("grid_stability", "grid"):
        df = _load_grid_stability()
    elif name == "digits":
        from sklearn.datasets import load_digits
        df = load_digits(as_frame=True).frame
    elif name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        df = load_breast_cancer(as_frame=True).frame
    else:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of {DATASETS}.")
    if verbose:
        print(f"Loaded '{name}': {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def _load_grid_stability():
    """Try the ucimlrepo package first; fall back to downloading the zip from UCI."""
    try:
        from ucimlrepo import fetch_ucirepo
        return fetch_ucirepo(id=471).data.original.copy()
    except Exception as err:  # package missing or UCI API unavailable
        print(f"ucimlrepo did not work ({type(err).__name__}); downloading the zip from UCI instead.")
    with urllib.request.urlopen(GRID_URL, timeout=60) as response:
        raw = response.read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
        return pd.read_csv(z.open(csv_name))


# --------------------------------------------------------------------------
# Exploratory data analysis
# --------------------------------------------------------------------------
def eda_summary(df, target=None):
    """Return a one-row-per-column summary: dtype, missing values, unique values, min, max.

    Also prints the number of duplicate rows and, if `target` is given, the class balance.
    """
    numeric = df.select_dtypes("number")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "min": numeric.min().reindex(df.columns),
        "max": numeric.max().reindex(df.columns),
    })
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]} | Duplicate rows: {df.duplicated().sum():,}")
    if target is not None:
        counts = df[target].value_counts()
        share = (counts / counts.sum() * 100).round(1)
        print("\nClass balance of", repr(target))
        print(pd.DataFrame({"count": counts, "%": share}).to_string())
    return summary


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_data(X, y, test_size=0.2, val_size=0.2, stratify=True, seed=42):
    """Split into train / validation / test sets.

    test_size and val_size are fractions of the WHOLE dataset, so the defaults give 60/20/20.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    strat = y if stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=seed)
    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_size / (1 - test_size), stratify=strat_tmp, random_state=seed)
    print(f"train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_classifier(model, X, y, name=None):
    """Score a fitted classifier. Binary targets use the positive class 1; multi-class uses macro averages."""
    y_pred = model.predict(X)
    binary = len(np.unique(y)) == 2
    avg = "binary" if binary else "macro"
    scores = {
        "model": name or type(model).__name__,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y, y_pred, average=avg, zero_division=0),
    }
    if binary and hasattr(model, "predict_proba"):
        scores["roc_auc"] = roc_auc_score(y, model.predict_proba(X)[:, 1])
    return scores


def evaluate_regressor(model, X, y, name=None):
    """Score a fitted regressor with MAE, RMSE and R²."""
    y_pred = model.predict(X)
    return {
        "model": name or type(model).__name__,
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
        "R2": r2_score(y, y_pred),
    }


def compare_models(models, X_train, y_train, X_eval, y_eval, task="classification"):
    """Fit every model in a {name: model} dict and score it on (X_eval, y_eval).

    The models you pass in are cloned, so they stay unfitted.
    Returns (results DataFrame sorted best-first, {name: fitted model}).
    """
    rows, fitted = [], {}
    for name, model in models.items():
        m = clone(model)
        start = time.perf_counter()
        m.fit(X_train, y_train)
        fit_time = time.perf_counter() - start
        if task == "classification":
            row = evaluate_classifier(m, X_eval, y_eval, name)
        else:
            row = evaluate_regressor(m, X_eval, y_eval, name)
        row["fit_time_s"] = round(fit_time, 3)
        rows.append(row)
        fitted[name] = m
        print(f"  done: {name:<22} ({fit_time:.2f} s)")
    sort_by = "f1" if task == "classification" else "R2"
    results = pd.DataFrame(rows).set_index("model").sort_values(sort_by, ascending=False)
    return results.round(4), fitted


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion(model, X, y, labels=None, normalize=None, title=None):
    """Plot the confusion matrix of a fitted classifier. normalize: None, 'true', 'pred' or 'all'."""
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=labels, normalize=normalize, cmap="Blues", colorbar=False)
    disp.ax_.set_title(title or f"Confusion matrix: {type(model).__name__}")
    plt.show()


def plot_feature_importance(model, feature_names, top=15):
    """Bar chart of feature importances (tree models) or absolute coefficients (linear models)."""
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "importance"
    elif hasattr(model, "coef_"):
        values, label = np.abs(np.ravel(model.coef_)), "|coefficient|"
    else:
        print(f"{type(model).__name__} has no built-in importances. "
              "Try sklearn.inspection.permutation_importance instead.")
        return None
    imp = pd.Series(values, index=feature_names).sort_values().tail(top)
    imp.plot.barh(figsize=(6, 0.35 * len(imp) + 1), color="#4C72B0")
    plt.xlabel(label)
    plt.title(f"Feature importance: {type(model).__name__}")
    plt.tight_layout()
    plt.show()
    return imp.sort_values(ascending=False)
