"""Завантаження та підготовка даних.

Запуск як скрипт готує parquet-файли й відрізає holdout:
    python src/data.py

ВАЖЛИВО щодо порядку операцій:
  1. Нормалізація категорій (strip + lower) робиться ДО розрізання на
     train_dev / holdout. Це детермінована операція: кожен рядок обробляється
     незалежно від інших, жодна статистика з даних не вчиться, тому витоку немає.
  2. Імпутація, скейлінг, енкодинг — НЕ тут. Вони навчені (fit на даних), тому
     живуть у sklearn.Pipeline всередині кожного фолду (див. cv.py).
  3. Holdout відрізається до будь-якого аналізу даних, включно з EDA, щоб
     залишатися повністю незалежною оцінкою.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from . import config as C
except ImportError:  # запуск як скрипт або з ноутбука
    import config as C


# --------------------------------------------------------------------------
# Нормалізація категорій
# --------------------------------------------------------------------------
def normalize_categories(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Прибирає пробіли та зводить до нижнього регістру в категоріальних колонках.

    EDA показав, що кожна категорія в program_track / rest_quality /
    learning_routine має "брудного двійника" з ведучим пробілом і UPPERCASE
    (напр. ' B.TECH' поряд з 'b.tech', ~1.5% рядків). Без нормалізації
    OneHotEncoder створить дублікати колонок, а дерево розщепить одну й ту саму
    категорію на дві гілки.

    ПАСТКА: .astype(str) перетворює NaN на літеральний рядок 'nan', після чого
    .isna() повертає False, і ~8800 пропусків у кожній колонці тихо стають
    повноцінною категорією. Тому .str застосовується напряму до object-колонки —
    акцесор .str сам пропускає NaN.
    """
    df = df.copy()
    cols = cols if cols is not None else [c for c in C.CAT_COLS if c in df.columns]
    for c in cols:
        df[c] = df[c].str.strip().str.lower()
    return df


def build_category_dtypes(frames: list[pd.DataFrame]) -> dict[str, pd.CategoricalDtype]:
    """Будує ФІКСОВАНІ рівні категорій з об'єднання всіх наборів даних.

    Навіщо фіксувати: pd.Categorical зберігає дані як цілочисельні коди плюс
    список категорій, і коди призначаються за порядком у цьому списку. Якщо
    зробити .astype('category') окремо на train і окремо на test, pandas
    побудує списки незалежно, і якщо в одному наборі якоїсь категорії немає —
    усі наступні коди зсунуться. Модель, навчена на train-кодах, отримає
    test-коди, що означають інші категорії. Виключення не буде, прогнози тихо
    стануть гіршими.

    Чи це витік? Ні. Таргет не задіяний — ми дивимося лише на те, які значення
    категорій існують у файлах ФІЧЕЙ, а test.csv з фічами виданий легально.
    Це transductive setting; у звіті він описаний явно.
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


# --------------------------------------------------------------------------
# Підготовка (запускається один раз)
# --------------------------------------------------------------------------
def prepare(verbose: bool = True) -> None:
    """Читає сирі CSV, нормалізує, фіксує dtypes, відрізає holdout, пише parquet."""
    log = print if verbose else (lambda *a, **k: None)

    log("Читання сирих CSV...")
    train = pd.read_csv(C.RAW_TRAIN)
    test = pd.read_csv(C.RAW_TEST)
    unlabeled = pd.read_csv(C.RAW_UNLABELED)
    log(f"  train {train.shape} | test {test.shape} | unlabeled {unlabeled.shape}")

    log("Нормалізація категорій (strip + lower)...")
    train = normalize_categories(train)
    test = normalize_categories(test)
    unlabeled = normalize_categories(unlabeled)

    log("Фіксація рівнів категорій з об'єднання train+test+unlabeled...")
    dtypes = build_category_dtypes([train, test, unlabeled])
    for c, dt in dtypes.items():
        log(f"  {c:20s} {len(dt.categories):2d} рівнів: {list(dt.categories)}")
    train = apply_category_dtypes(train, dtypes)
    test = apply_category_dtypes(test, dtypes)
    unlabeled = apply_category_dtypes(unlabeled, dtypes)

    log(f"\nРозрізання holdout {C.HOLDOUT_FRAC:.0%} (seed={C.SEED}) ДО будь-якого аналізу...")
    train_dev, holdout = train_test_split(
        train, test_size=C.HOLDOUT_FRAC, random_state=C.SEED, shuffle=True
    )
    log(f"  train_dev {train_dev.shape} | holdout {holdout.shape}")

    # Стратифікація по корзинах таргета не застосована свідомо:
    # SE середнього на holdout = sigma/sqrt(n) = 18.9/sqrt(44100) ~ 0.09 бала,
    # що є шумом третього знаку при RMSE ~ 8.9. Див. звіт.
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
    """Чек-ліст кроку 0. Падає з AssertionError, якщо щось не так."""
    log = print if verbose else (lambda *a, **k: None)
    train_dev = pd.read_parquet(C.TRAIN_DEV)
    holdout = pd.read_parquet(C.HOLDOUT)
    test = pd.read_parquet(C.TEST)

    # цілісність розрізу
    assert len(train_dev) + len(holdout) == 441_000, "втрачено або продубльовано рядки"
    assert set(train_dev[C.ID_COL]).isdisjoint(holdout[C.ID_COL]), "train_dev і holdout перетинаються"
    assert train_dev[C.ID_COL].is_unique and holdout[C.ID_COL].is_unique

    # нормалізація справді відбулась
    for c in C.CAT_COLS:
        s = train_dev[c].dropna().astype(str)
        assert (s == s.str.strip()).all(), f"{c}: залишились пробіли"
        assert (s == s.str.lower()).all(), f"{c}: залишився верхній регістр"

    # пропуски НЕ зникли (перевірка на пастку .astype(str) -> 'nan')
    assert train_dev[C.CAT_COLS].isna().sum().min() > 5_000, "пропуски в категоріях зникли"
    # перевіряємо РІВНІ категорій, а не astype(str) усієї колонки: astype(str)
    # сам перетворив би справжні NaN на рядок 'nan' і зробив перевірку безглуздою
    for c in C.CAT_COLS:
        assert "nan" not in set(train_dev[c].cat.categories), f"{c}: 'nan' став категорією"

    # рівні категорій однакові в усіх файлах (перевірка на зсув кодів)
    for c in C.CAT_COLS:
        assert list(train_dev[c].cat.categories) == list(test[c].cat.categories), f"{c}: рівні різні"
        assert list(train_dev[c].cat.categories) == list(holdout[c].cat.categories), f"{c}: рівні різні"

    # фіче-сет чистий
    assert C.ID_COL not in C.FEATURES_V0 and C.TARGET not in C.FEATURES_V0
    assert all(f in train_dev.columns for f in C.FEATURES_V0)

    log("\n[OK] Усі перевірки кроку 0 пройдено.")


# --------------------------------------------------------------------------
# Завантаження (використовується в ноутбуках)
# --------------------------------------------------------------------------
def load_train_dev() -> pd.DataFrame:
    """90% train. Єдине джерело даних для ноутбуків 01-04."""
    return pd.read_parquet(C.TRAIN_DEV)


def load_test() -> pd.DataFrame:
    return pd.read_parquet(C.TEST)


def load_unlabeled() -> pd.DataFrame:
    return pd.read_parquet(C.UNLABELED)


def load_holdout() -> pd.DataFrame:
    """UNTOUCHED HOLDOUT.

    Викликати ТІЛЬКИ з 05_final_submission.ipynb, після того як усі рішення
    щодо моделі, гіперпараметрів і фічей уже прийняті за CV. Кожне звернення
    до цих даних до фіналу знижує цінність holdout як незалежної оцінки.
    """
    return pd.read_parquet(C.HOLDOUT)


def xy(df: pd.DataFrame, features: list[str] | None = None):
    """Розділяє датафрейм на (X, y) за явним списком фіч."""
    features = features if features is not None else C.FEATURES_V0
    return df[features].copy(), df[C.TARGET].copy()


if __name__ == "__main__":
    prepare()
