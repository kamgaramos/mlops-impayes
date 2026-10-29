from pathlib import Path
import pandas as pd
from src.config import load_params


def normalize_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        raise KeyError(f"Colonne cible absente: {target}")

    s = df[target].copy()
    if pd.api.types.is_numeric_dtype(s):
        converted = pd.to_numeric(s, errors="raise").astype(int)
        unknown = sorted(set(converted.unique()) - {0, 1})
        if unknown:
            raise ValueError(
                f"Valeurs numériques inattendues pour la cible '{target}': {unknown}. "
                "La cible doit être binaire (0 ou 1)."
            )
        df[target] = converted
        return df

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
    normalized = s.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(
            f"Valeurs de cible non supportées pour '{target}': {unknown}. "
            "Attendu: 0/1 ou low/medium/high/yes/no/true/false."
        )
    df[target] = normalized.map(mapping).astype(int)
    return df


def download(p: dict) -> pd.DataFrame:
    from datasets import load_dataset
    ds = load_dataset(p["data"]["hf_dataset"])
    split = "train" if "train" in ds else list(ds.keys())[0]
    return ds[split].to_pandas()


def validate(df: pd.DataFrame, target: str) -> None:
    assert target in df.columns, f"Colonne cible absente: {target}"
    assert df[target].notna().all(), "Valeurs nulles dans la cible"
    assert len(df) > 0, "Jeu de données vide"
    assert set(df[target].unique()) <= {0, 1}, (
        f"La cible '{target}' doit être binaire après normalisation; valeurs observées: {sorted(df[target].unique())}"
    )
    if "tenure_months" in df.columns:
        assert (df["tenure_months"].dropna() >= 0).all(), "tenure_months négatif"


def load_data(p: dict) -> pd.DataFrame:
    raw = Path(p["data"]["raw_path"])
    if raw.exists():
        df = pd.read_csv(raw)
    else:
        df = download(p)
        raw.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw, index=False)

    df = normalize_target(df, p["data"]["target"])
    validate(df, p["data"]["target"])
    return df


if __name__ == "__main__":
    d = load_data(load_params())
    print(d.shape)
    print(d.dtypes)
