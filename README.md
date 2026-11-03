# eee4710-lab-utils

Helper functions for **EEE 4710: Artificial Intelligence and Machine Learning Lab** (IUT).

## Use in Google Colab

```python
!git clone -q https://github.com/<INSTRUCTOR_USERNAME>/eee4710-lab-utils.git
import sys; sys.path.append("eee4710-lab-utils")
import eee4710_utils as eu
help(eu)
```

Only need the one file? Download it directly:

```python
!wget -q https://raw.githubusercontent.com/<INSTRUCTOR_USERNAME>/eee4710-lab-utils/main/eee4710_utils.py
import eee4710_utils as eu
```

## What's inside

| Function | What it does |
|---|---|
| `set_seed(seed)` | Fixes Python and NumPy random seeds |
| `load_dataset(name, path=None)` | Loads `grid_stability` (UCI), `digits` or `breast_cancer` as a DataFrame |
| `eda_summary(df, target=None)` | Per-column dtype, missing values, unique values, min/max; class balance |
| `split_data(X, y, test_size, val_size)` | Stratified train / validation / test split (60/20/20 by default) |
| `evaluate_classifier(model, X, y)` | Accuracy, precision, recall, F1 and ROC-AUC |
| `evaluate_regressor(model, X, y)` | MAE, RMSE and R² |
| `compare_models(models, X_tr, y_tr, X_ev, y_ev, task)` | Fits and scores a dict of models, returns a sorted table |
| `plot_confusion(model, X, y)` | Confusion matrix plot |
| `plot_feature_importance(model, feature_names)` | Tree importances or linear-model coefficients |

Requires: numpy, pandas, matplotlib, scikit-learn (all preinstalled in Colab). `ucimlrepo` is optional.
