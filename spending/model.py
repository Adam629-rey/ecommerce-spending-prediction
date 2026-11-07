"""Validated loading, reproducible training and holdout evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/customers.csv"
FEATURES = ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]
TARGET = "Yearly Amount Spent"

def load_data(path=DATA):
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[required].apply(pd.to_numeric, errors="raise")
    if len(frame) < 20:
        raise ValueError("At least 20 observations are required.")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or infinite values are not allowed.")
    if (frame < 0).any().any():
        raise ValueError("Features and spending must be non-negative.")
    if frame.duplicated(subset=FEATURES).any():
        raise ValueError("Duplicate feature rows must be reviewed before splitting.")
    if frame[TARGET].nunique() < 2:
        raise ValueError("Target must vary.")
    return frame

def metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return dict(mae=float(mean_absolute_error(actual, predicted)), mse=float(mse),
                rmse=float(np.sqrt(mse)), r2=float(r2_score(actual, predicted)))

def train(frame):
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame[TARGET], test_size=0.3, random_state=101)
    cv = cross_validate(LinearRegression(), X_train, y_train,
        cv=KFold(n_splits=5, shuffle=True, random_state=101),
        scoring={"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error", "r2": "r2"})
    model = LinearRegression().fit(X_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    pred = model.predict(X_test)
    report = {"rows": len(frame), "train_rows": len(X_train), "test_rows": len(X_test),
              "seed": 101, "test_fraction": 0.3,
              "holdout": metrics(y_test, pred), "baseline": metrics(y_test, baseline.predict(X_test)),
              "cross_validation": {}}
    for name, scores in {"mae": -cv["test_mae"], "rmse": np.sqrt(-cv["test_mse"]), "r2": cv["test_r2"]}.items():
        report["cross_validation"][name] = {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores.tolist()}
    return model, report, X_train, pd.DataFrame({"actual": y_test, "predicted": pred, "residual": y_test - pred})

def predict(model, frame):
    values = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any().any():
        raise ValueError("Inputs must be finite, non-negative numbers.")
    return model.predict(values)
