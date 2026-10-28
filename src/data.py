import pandas as pd

from config import DATA_PATH, TARGET


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan pada {path}")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def split_features_target(df):
    return df.drop(columns=[TARGET]), df[TARGET]
