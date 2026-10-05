from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import config as C
except ImportError:
    import config as C


def normalize_categories(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Прибирає пробіли та зводить категорії до нижнього регістру.

    У сирих даних кожна категорія має двійника з ведучим пробілом і великими
    літерами. Застосовую .str напряму, а не через astype(str) — інакше NaN
    перетворився б на рядок 'nan' і перестав бути пропуском.
    """
    df = df.copy()
    cols = cols if cols is not None else [c for c in C.CAT_COLS if c in df.columns]
    for c in cols:
        df[c] = df[c].str.strip().str.lower()
    return df


def build_category_dtypes(frames: list[pd.DataFrame]) -> dict[str, pd.CategoricalDtype]:
    """Фіксує рівні категорій один раз для всіх наборів даних.

    Pandas зберігає категорії цілими кодами, і якщо нумерувати train і test
    окремо, коди можуть зсунутися — модель тихо почне плутати категорії.
    Таргет тут не задіяний, тож витоку немає.
    """
    dtypes = {}
    for c in C.CAT_COLS:
        levels = set()
        for df in frames:
            if c in df.columns:
                levels |= set(df[c].dropna().unique())
        dtypes[c] = pd.CategoricalDtype(categories=sorted(levels), ordered=False)
    return dtypes


def apply_category_dtypes(df: pd.DataFrame, dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    df = df.copy()
    for c, dt in dtypes.items():
        if c in df.columns:
            df[c] = df[c].astype(dt)
    return df


def prepare(verbose: bool = True) -> None:
    """Готує дані: нормалізує категорії, фіксує dtypes, відрізає holdout.

    Holdout відрізаю тут, до будь-якого аналізу, щоб він лишався незалежною
    оцінкою. Запускати один раз: python src/data.py
    """
    log = print if verbose else (lambda *a, **k: None)

    log("Читання сирих CSV...")
    train = pd.read_csv(C.RAW_TRAIN)
    test = pd.read_csv(C.RAW_TEST)
    unlabeled = pd.read_csv(C.RAW_UNLABELED)
    log(f"  train {train.shape} | test {test.shape} | unlabeled {unlabeled.shape}")

    log("Нормалізація категорій...")
    train = normalize_categories(train)
    test = normalize_categories(test)
    unlabeled = normalize_categories(unlabeled)

    log("Фіксація рівнів категорій...")
    dtypes = build_category_dtypes([train, test, unlabeled])
    for c, dt in dtypes.items():
        log(f"  {c:20s} {len(dt.categories):2d} рівнів: {list(dt.categories)}")
    train = apply_category_dtypes(train, dtypes)
    test = apply_category_dtypes(test, dtypes)
    unlabeled = apply_category_dtypes(unlabeled, dtypes)

    log(f"\nРозрізання holdout {C.HOLDOUT_FRAC:.0%} (seed={C.SEED})...")
    train_dev, holdout = train_test_split(
        train, test_size=C.HOLDOUT_FRAC, random_state=C.SEED, shuffle=True
    )
    log(f"  train_dev {train_dev.shape} | holdout {holdout.shape}")
    log(f"  mean target: train_dev {train_dev[C.TARGET].mean():.4f} | "
        f"holdout {holdout[C.TARGET].mean():.4f} "
        f"(SE ~ {train[C.TARGET].std()/np.sqrt(len(holdout)):.3f})")

    log("\nЗапис parquet...")
    train_dev.reset_index(drop=True).to_parquet(C.TRAIN_DEV, index=False)
    holdout.reset_index(drop=True).to_parquet(C.HOLDOUT, index=False)
    test.to_parquet(C.TEST, index=False)
    unlabeled.to_parquet(C.UNLABELED, index=False)
    for p in (C.TRAIN_DEV, C.HOLDOUT, C.TEST, C.UNLABELED):
        log(f"  {p.name:22s} {p.stat().st_size/1e6:6.1f} MB")

    validate_preparation(verbose=verbose)


def validate_preparation(verbose: bool = True) -> None:
    log = print if verbose else (lambda *a, **k: None)
    train_dev = pd.read_parquet(C.TRAIN_DEV)
    holdout = pd.read_parquet(C.HOLDOUT)
    test = pd.read_parquet(C.TEST)

    assert len(train_dev) + len(holdout) == 441_000, "втрачено або продубльовано рядки"
    assert set(train_dev[C.ID_COL]).isdisjoint(holdout[C.ID_COL]), "train_dev і holdout перетинаються"
    assert train_dev[C.ID_COL].is_unique and holdout[C.ID_COL].is_unique

    for c in C.CAT_COLS:
        s = train_dev[c].dropna().astype(str)
        assert (s == s.str.strip()).all(), f"{c}: залишились пробіли"
        assert (s == s.str.lower()).all(), f"{c}: залишився верхній регістр"

    assert train_dev[C.CAT_COLS].isna().sum().min() > 5_000, "пропуски в категоріях зникли"
    for c in C.CAT_COLS:
        assert "nan" not in set(train_dev[c].cat.categories), f"{c}: 'nan' став категорією"

    for c in C.CAT_COLS:
        assert list(train_dev[c].cat.categories) == list(test[c].cat.categories), f"{c}: рівні різні"
        assert list(train_dev[c].cat.categories) == list(holdout[c].cat.categories), f"{c}: рівні різні"

    assert C.ID_COL not in C.FEATURES_V0 and C.TARGET not in C.FEATURES_V0
    assert all(f in train_dev.columns for f in C.FEATURES_V0)

    log("\n[OK] Усі перевірки кроку 0 пройдено.")


def load_train_dev() -> pd.DataFrame:
    return pd.read_parquet(C.TRAIN_DEV)


def load_test() -> pd.DataFrame:
    return pd.read_parquet(C.TEST)


def load_unlabeled() -> pd.DataFrame:
    return pd.read_parquet(C.UNLABELED)


def load_holdout() -> pd.DataFrame:
    """Читає holdout. Викликаю тільки з 05_final_submission.ipynb.

    Кожне звернення до цих даних до фіналу знижує цінність holdout як
    незалежної оцінки.
    """
    return pd.read_parquet(C.HOLDOUT)


def xy(df: pd.DataFrame, features: list[str] | None = None):
    features = features if features is not None else C.FEATURES_V0
    return df[features].copy(), df[C.TARGET].copy()


if __name__ == "__main__":
    prepare()
