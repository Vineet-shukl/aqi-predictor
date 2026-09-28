"""Models that are not sklearn estimators but must remain joblib-loadable."""

from __future__ import annotations

import pandas as pd


class PersistenceRegressor:
    """Predict today's PM2.5 with yesterday's PM2.5 (the lag-1 feature)."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn-style argument name
        return self

    def predict(self, X):  # noqa: N803
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PersistenceRegressor expects a pandas DataFrame")
        if "pm25_lag_1" not in X.columns:
            raise KeyError("pm25_lag_1 is required")
        return X["pm25_lag_1"].to_numpy(dtype=float)
