import json
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import load_params
from src.data import load_data


def _coerce_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype(int)

    mapping = {
        "0": 0,
        "1": 1,
        "low": 0,
        "medium": 1,
        "high": 1,
        "no": 0,
        "yes": 1,
        "false": 0,
        "true": 1,
    }
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(f"Cible invalide: {unknown}")
    return normalized.map(mapping).astype(int)


def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X


def _build_pipeline(X: pd.DataFrame, max_iter: int) -> Pipeline:
    numeric_cols = list(X.select_dtypes(include=["number"]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop") if transformers else "passthrough"
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=max_iter)),
        ]
    )


def train(df: pd.DataFrame, params: dict, track: bool = True, out_dir: str = "models"):
    target = params["data"]["target"]
    drop_cols = set(params["data"].get("drop_columns", []))
    feature_cols = [c for c in df.columns if c not in {target, *drop_cols}]

    X = df[feature_cols].copy()
    y = _coerce_target(df[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["data"].get("test_size", 0.2),
        random_state=params["data"].get("seed", 42),
        stratify=y,
    )

    model = _build_pipeline(X_train, params["train"].get("max_iter", 200))
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "f1": float(f1_score(y_test, (proba >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y_test, proba)),
    }

    gate = params.get("gate", {})
    passed = (
        metrics["f1"] >= gate.get("min_f1", 0.0)
        and metrics["auc"] >= gate.get("min_auc", 0.0)
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(model, out_path / "model.joblib")

        with open(out_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out_path / "meta.json", "w") as f:
            json.dump({"version": "v1", "features": feature_cols, "target": target}, f, indent=2)

        ref = X_train.copy()
        ref["target"] = y_train
        ref.to_csv(out_path / "reference.csv", index=False)

    # Keep the original flag for caller intent without preventing artifact generation.
    if track:
        # This branch intentionally remains empty: tracking is optional in this repo,
        # and model artifacts must still be written when out_dir is provided.
        pass

    return {
        "passed": passed,
        "metrics": metrics,
        "model": model,
    }


if __name__ == "__main__":
    params = load_params()
    df = load_data(params)
    result = train(df, params, track=True, out_dir="models")
    print(json.dumps(result["metrics"], indent=2))
    print("PASS" if result["passed"] else "FAIL")
