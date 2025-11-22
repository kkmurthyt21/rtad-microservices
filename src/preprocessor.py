import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple z-score normalization column-wise."""
    df = df.copy()
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            # avoid division by zero, leave column unchanged
            continue
        df[col] = (df[col] - mean) / std
    return df
