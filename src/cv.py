"""Єдина точка оцінювання моделей.

Усі числа, що потрапляють у звіт, отримані через run_cv(). Це гарантує, що
порівняння між моделями коректні: ті самі фолди, та сама метрика, той самий
спосіб агрегації.

Два протоколи:
  A (BENCHMARK) — 5 фолдів на повних train_dev. Усі числа для звіту.
  B (SEARCH)    — 3 фолди на підвибірці 150k. Лише для РАНЖУВАННЯ
                  гіперпараметрів; переможець завжди переміряється за A.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

try:
    from . import config as C
except ImportError:
    import config as C


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def get_folds(protocol: str = "A") -> KFold:
    """Повертає спліттер. Створюється в одному місці, щоб усі експерименти
    використовували ідентичне розбиття."""
    n = C.N_FOLDS if protocol == "A" else C.SEARCH_N_FOLDS
    return KFold(n_splits=n, shuffle=True, random_state=C.SEED)


def run_cv(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    name: str,
    protocol: str = "A",
    feature_set: str = "v0",
    params: dict | None = None,
    fit_kwargs_fn=None,
    log: bool = True,
    verbose: bool = True,
    save_oof: bool = False,
):
    """Прогоняє K-fold CV і повертає (oof, fold_rmses).

    Parameters
    ----------
    model_factory : callable
        Callable БЕЗ аргументів, що повертає НОВИЙ незафічений estimator або
        Pipeline. Саме callable, а не готовий об'єкт: якби передавався об'єкт,
        на другому фолді він би дофітувався поверх стану з першого, і
        валідаційні рядки першого фолду вже брали б участь у навчанні —
        тихий витік без жодного повідомлення про помилку.
    fit_kwargs_fn : callable(X_trn, y_trn) -> dict, optional
        Для early stopping: будує eval_set з ВНУТРІШНЬОГО спліту навчального
        фолду, щоб зовнішній (оцінюваний) фолд не брав участі у виборі
        кількості раундів.
    """
    folds = get_folds(protocol)
    oof = np.full(len(X), np.nan)
    fold_rmses, train_rmses, fit_times = [], [], []

    for k, (trn, val) in enumerate(folds.split(X), 1):
        X_trn, y_trn = X.iloc[trn], y.iloc[trn]
        X_val, y_val = X.iloc[val], y.iloc[val]

        model = model_factory()          # новий об'єкт на кожному фолді
        fit_kwargs = fit_kwargs_fn(X_trn, y_trn) if fit_kwargs_fn else {}

        t0 = time.time()
        model.fit(X_trn, y_trn, **fit_kwargs)
        fit_times.append(time.time() - t0)

        oof[val] = model.predict(X_val)
        fold_rmses.append(rmse(y_val, oof[val]))
        train_rmses.append(rmse(y_trn, model.predict(X_trn)))

        if verbose:
            print(f"  fold {k}/{folds.get_n_splits()}  "
                  f"val {fold_rmses[-1]:.4f}  train {train_rmses[-1]:.4f}  "
                  f"{fit_times[-1]:.1f}s")

    mask = ~np.isnan(oof)
    row = {
        "name": name,
        "protocol": protocol,
        "feature_set": feature_set,
        "n_features": X.shape[1],
        "n_rows": len(X),
        "n_folds": folds.get_n_splits(),
        # mean/std по фолдах: std показує, чи різниця між моделями значуща
        "mean_rmse": float(np.mean(fold_rmses)),
        "std_rmse": float(np.std(fold_rmses)),
        # RMSE по всьому OOF-вектору: одне число на всіх рядках, менш шумне.
        # Не зобов'язане дорівнювати mean_rmse — RMSE не лінійна функція.
        "oof_rmse": rmse(y[mask], oof[mask]),
        "train_rmse": float(np.mean(train_rmses)),
        "overfit_gap": float(np.mean(fold_rmses) - np.mean(train_rmses)),
        "fit_seconds": float(np.sum(fit_times)),
        "params": json.dumps(params or {}, default=str, ensure_ascii=False),
    }

    if verbose:
        print(f"[{name}]  OOF {row['oof_rmse']:.4f}  |  "
              f"mean {row['mean_rmse']:.4f} ± {row['std_rmse']:.4f}  |  "
              f"train {row['train_rmse']:.4f}  (gap {row['overfit_gap']:+.4f})  |  "
              f"{row['fit_seconds']:.0f}s")

    if log:
        _append_result(row)
    if save_oof:
        # OOF-вектор потрібен для блендингу та стекінгу: мета-модель МУСИТЬ
        # вчитися саме на прогнозах для рядків, яких базова модель не бачила
        C.OOF_DIR.mkdir(parents=True, exist_ok=True)
        np.save(C.OOF_DIR / f"{name}.npy", oof)
    return oof, fold_rmses


def load_oof(names: list[str]) -> pd.DataFrame:
    """Збирає матрицю OOF-прогнозів (рядки × моделі) для ансамблю."""
    return pd.DataFrame({n: np.load(C.OOF_DIR / f"{n}.npy") for n in names})


def _append_result(row: dict) -> None:
    """Дописує рядок у журнал експериментів. Журнал ведеться з першого дня,
    щоб розділ 'Результати' у звіті збирався з таблиці, а не з пам'яті."""
    C.EXPERIMENTS_DIR.mkdir(exist_ok=True)
    row = {"timestamp": pd.Timestamp.now().isoformat(timespec="seconds"), **row}
    df = pd.DataFrame([row])
    header = not C.RESULTS_CSV.exists()
    df.to_csv(C.RESULTS_CSV, mode="a", header=header, index=False)


def results_table(protocol: str | None = None) -> pd.DataFrame:
    """Читає журнал експериментів для звіту."""
    if not C.RESULTS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(C.RESULTS_CSV)
    if protocol:
        df = df[df.protocol == protocol]
    return df


def make_submission(pred: np.ndarray, test_ids: pd.Series, name: str,
                    clip: tuple[float, float] | None = None) -> str:
    """Записує сабмішн з вирівнюванням по sample_submission і перевіркою формату."""
    if clip is not None:
        pred = np.clip(pred, *clip)

    sample = pd.read_csv(C.SAMPLE_SUB)
    preds = pd.DataFrame({C.ID_COL: test_ids.values, C.TARGET: pred})
    # Вирівнювання по record_id, а не покладання на порядок рядків
    sub = sample[[C.ID_COL]].merge(preds, on=C.ID_COL, how="left")

    assert list(sub.columns) == [C.ID_COL, C.TARGET]
    assert len(sub) == 189_000, len(sub)
    assert sub[C.ID_COL].is_unique
    assert set(sub[C.ID_COL]) == set(sample[C.ID_COL])
    assert (sub[C.ID_COL].values == sample[C.ID_COL].values).all(), "порядок не збігається"
    assert sub[C.TARGET].notna().all(), "є пропуски в прогнозах"
    assert np.isfinite(sub[C.TARGET]).all(), "є нескінченні значення"

    C.SUBMISSIONS_DIR.mkdir(exist_ok=True)
    path = C.SUBMISSIONS_DIR / f"sub_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"[OK] {path.name}  n={len(sub)}  "
          f"min={sub[C.TARGET].min():.2f}  mean={sub[C.TARGET].mean():.2f}  max={sub[C.TARGET].max():.2f}")
    return str(path)
