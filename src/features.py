from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from . import config as C
except ImportError:
    import config as C


MISSING_TOKEN = "__missing__"

ORDINAL_MAPS = {
    "rest_quality": {"poor": 0, "average": 1, "good": 2},
    "campus_resources": {"low": 0, "medium": 1, "high": 2},
}

STUDY_TIME_CAP = 9.2
STUDY_SCALE = 4.0


def make_preprocessor(scale: bool = True,
                      num_cols: list[str] | None = None,
                      cat_cols: list[str] | None = None,
                      cat_missing: str = "most_frequent") -> ColumnTransformer:
    """Збирає препроцесор: пропуски заповнюються, категорії кодуються one-hot.

    Скейлінг вмикаю лише для лінійних моделей — деревам він нічого не дає.
    cat_missing='category' замість заповнення модою робить пропуск окремим
    значенням; мода тут не нейтральна і зміщує прогноз.
    """
    num_cols = num_cols if num_cols is not None else C.NUM_COLS
    cat_cols = cat_cols if cat_cols is not None else C.CAT_COLS

    if cat_missing == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif cat_missing == "category":
        cat_imputer = SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)
    else:
        raise ValueError(f"cat_missing має бути 'most_frequent' або 'category', не {cat_missing!r}")

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline([
                ("impute", cat_imputer),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_features(
    df,
    fix_study_time: bool = False,
    mutual_fill: bool = False,
    add_consensus: bool = False,
    drop: tuple = (),
    missing_indicators: bool = False,
    ordinal: bool = False,
):
    """Повертає (X, num_cols, cat_cols) для заданої комбінації змін фіч.

    Усе тут обробляє рядок незалежно від інших, тому застосовую до розрізання
    на фолди. Прапорці відповідають експериментам FE-1…FE-4 з ноутбука 03.
    """
    import numpy as np
    import pandas as pd

    X = df.copy()
    num_cols = list(C.NUM_COLS)
    cat_cols = list(C.CAT_COLS)

    if missing_indicators:
        for c in C.NUM_COLS:
            X[f"{c}_isna"] = X[c].isna().astype(np.int8)
            num_cols.append(f"{c}_isna")
        X["n_missing"] = df[C.FEATURES_V0].isna().sum(axis=1).astype(np.int8)
        num_cols.append("n_missing")

    if fix_study_time:
        X["study_time"] = np.where(X["study_time"] > STUDY_TIME_CAP,
                                   X["study_time"] / STUDY_SCALE, X["study_time"])
    if mutual_fill:
        X["study_minutes"] = X["study_minutes"].fillna(X["study_time"] * 60)
        X["study_time"] = X["study_time"].fillna(X["study_minutes"] / 60)
    if add_consensus:
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                X["study_consensus"] = np.nanmean(
                    np.c_[X["study_time"].values, X["study_minutes"].values / 60], axis=1)
        num_cols.append("study_consensus")

    if ordinal:
        for c, mapping in ORDINAL_MAPS.items():
            if c in cat_cols:
                X[f"{c}_ord"] = X[c].astype(object).map(mapping).astype(float)
                num_cols.append(f"{c}_ord")
                cat_cols.remove(c)

    for c in drop:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    return X[num_cols + cat_cols], num_cols, cat_cols
