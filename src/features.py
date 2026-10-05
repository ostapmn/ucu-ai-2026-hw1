"""Побудова фіч і препроцесорів.

Усі НАВЧЕНІ перетворення (імпутація, скейлінг, енкодинг) повертаються як
sklearn-трансформери, щоб фітуватися всередині кожного фолду. Pipeline тут —
не стилістика, а інструмент коректності: якби медіана для імпутації
рахувалася по всьому train_dev до розрізання на фолди, у навчання потрапила б
інформація з валідаційних рядків.
"""
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


def make_preprocessor(scale: bool = True,
                      num_cols: list[str] | None = None,
                      cat_cols: list[str] | None = None,
                      cat_missing: str = "most_frequent") -> ColumnTransformer:
    """Базовий препроцесор: median-імпутація + (опційно) скейлінг для числових,
    most_frequent-імпутація + one-hot для категоріальних.

    scale=True  — для лінійних моделей (коефіцієнти стають порівнюваними).
    scale=False — для дерев: вони інваріантні до монотонних перетворень, тому
                  скейлінг нічого не змінює, лише витрачає час.

    cat_missing='most_frequent' — заповнити пропуски модою (базовий варіант v0).
    cat_missing='category'      — зробити пропуск окремою категорією (FE-5).

        Мотивація варіанту 'category': мода не є нейтральним значенням.
        Для rest_quality мода — 'poor' із середнім балом 57.00 при глобальному
        62.51, для learning_routine — 'coaching' із 69.26. Тобто заповнення модою
        вносить систематичний зсув −5.51 і +6.74 бала відповідно на ~8 000 рядках
        кожне, хоча EDA (§3 ноутбука 01) показав, що пропуски MCAR і їхній
        середній бал дорівнює глобальному. Окрема категорія дозволяє моделі
        вивчити для цих рядків саме ≈ 62.5.

    handle_unknown='ignore' обов'язковий: у навчальному фолді може не
    трапитись рідкісна категорія, і без цього predict на валідації впаде.
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


# ---------------------------------------------------------------------------
# Конструктор фіче-сетів для експериментів ноутбука 03
# ---------------------------------------------------------------------------
# Усі перетворення нижче ДЕТЕРМІНОВАНІ: кожен рядок обробляється незалежно від
# інших, жодна статистика з даних не оцінюється. Тому їх можна застосовувати до
# розрізання на фолди без ризику витоку. Навчені перетворення (імпутація,
# скейлінг, енкодинг) лишаються у make_preprocessor всередині фолду.

ORDINAL_MAPS = {
    "rest_quality": {"poor": 0, "average": 1, "good": 2},
    "campus_resources": {"low": 0, "medium": 1, "high": 2},
}

STUDY_TIME_CAP = 9.2      # фізична межа: study_minutes має стелю 528 хв = 8.8 год
STUDY_SCALE = 4.0         # виміряний множник пошкодження (EDA §6.1, медіана 3.9991)


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

    fix_study_time     — поділити пошкоджені study_time (> 9.2 год) на 4 (FE-1)
    mutual_fill        — взаємно заповнити пропуски study_time <-> study_minutes (FE-1)
    add_consensus      — додати усереднення двох вимірів часу навчання (FE-1)
    drop               — колонки, які прибрати (FE-2)
    missing_indicators — додати *_isna для числових і n_missing (FE-3)
    ordinal            — замінити one-hot на порядковий код для впорядкованих категорій (FE-4)
    """
    import numpy as np
    import pandas as pd

    X = df.copy()
    num_cols = list(C.NUM_COLS)
    cat_cols = list(C.CAT_COLS)

    # --- FE-3: індикатори рахуються ДО будь-яких заповнень ---
    if missing_indicators:
        for c in C.NUM_COLS:
            X[f"{c}_isna"] = X[c].isna().astype(np.int8)
            num_cols.append(f"{c}_isna")
        X["n_missing"] = df[C.FEATURES_V0].isna().sum(axis=1).astype(np.int8)
        num_cols.append("n_missing")

    # --- FE-1: виправлення пошкодження study_time ---
    if fix_study_time:
        X["study_time"] = np.where(X["study_time"] > STUDY_TIME_CAP,
                                   X["study_time"] / STUDY_SCALE, X["study_time"])
    if mutual_fill:
        X["study_minutes"] = X["study_minutes"].fillna(X["study_time"] * 60)
        X["study_time"] = X["study_time"].fillna(X["study_minutes"] / 60)
    if add_consensus:
        # nanmean по рядку, де обидва виміри NaN, законно дає NaN (імпутер потім
        # його заповнить) — глушимо лише супутнє попередження numpy
        with np.errstate(invalid="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                X["study_consensus"] = np.nanmean(
                    np.c_[X["study_time"].values, X["study_minutes"].values / 60], axis=1)
        num_cols.append("study_consensus")

    # --- FE-4: порядкове кодування впорядкованих категорій ---
    if ordinal:
        for c, mapping in ORDINAL_MAPS.items():
            if c in cat_cols:
                X[f"{c}_ord"] = X[c].astype(object).map(mapping).astype(float)
                num_cols.append(f"{c}_ord")
                cat_cols.remove(c)

    # --- FE-2: видалення фіч ---
    for c in drop:
        if c in num_cols:
            num_cols.remove(c)
        if c in cat_cols:
            cat_cols.remove(c)

    return X[num_cols + cat_cols], num_cols, cat_cols
