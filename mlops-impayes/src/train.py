def _coerce_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="raise").astype(int)

    mapping = {
        "0": 0, "1": 1,
        "low": 0, "medium": 1, "high": 1,
        "no": 0, "yes": 1,
        "false": 0, "true": 1,
    }
    normalized = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(f"Cible invalide: {unknown}")
    return normalized.map(mapping).astype(int)
from scipy import sparse

def ensure_dense(X):
    if sparse.issparse(X):
        return X.toarray()
    return X
