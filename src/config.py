"""Єдине джерело правди для констант проєкту.

Усі ноутбуки та модулі імпортують константи звідси. Причина: якщо, наприклад,
random_state для KFold буде продубльований у двох ноутбуках і в одному з них
випадково зміниться, половина порівняльної таблиці у звіті буде отримана на
інших фолдах — числа виглядатимуть нормально, але порівняння стане недійсним.
"""
from pathlib import Path

# --- Відтворюваність -------------------------------------------------------
SEED = 42
N_FOLDS = 5
HOLDOUT_FRAC = 0.10

# Protocol B (пошук гіперпараметрів): підвибірка + менше фолдів, щоб вкластися
# в бюджет CPU. Переможець завжди переміряється за Protocol A (повні дані, 5 фолдів).
SEARCH_N_FOLDS = 3
SEARCH_SAMPLE = 150_000

# --- Шляхи -----------------------------------------------------------------
# Будуються від розташування цього файлу, а не від cwd: інакше код працює
# з ноутбука, але падає при запуску скриптом.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RAW_TRAIN = DATA_DIR / "train.csv"
RAW_TEST = DATA_DIR / "test.csv"
RAW_UNLABELED = DATA_DIR / "unlabeled.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

# Підготовлені файли (parquet зберігає dtypes, зокрема category)
TRAIN_DEV = DATA_DIR / "train_dev.parquet"   # 90% train — тут вся робота
HOLDOUT = DATA_DIR / "holdout.parquet"       # 10% train — НЕ чіпати до фіналу
TEST = DATA_DIR / "test.parquet"
UNLABELED = DATA_DIR / "unlabeled.parquet"

EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
OOF_DIR = EXPERIMENTS_DIR / "oof"   # збережені OOF-вектори для ансамблів
SUBMISSIONS_DIR = ROOT / "submissions"
FIGURES_DIR = ROOT / "figures"

# --- Колонки ---------------------------------------------------------------
ID_COL = "record_id"
TARGET = "assessment_score"

NUM_COLS = [
    "learner_age",
    "study_time",
    "attendance_rate",
    "sleep_duration",
    "study_minutes",
    "engagement_index",
    "intake_marker",
]

CAT_COLS = [
    "gender_group",
    "program_track",
    "home_internet",
    "rest_quality",
    "learning_routine",
    "campus_resources",
    "assessment_level",
    "registration_group",
]

# Базовий фіче-сет (v0) для обов'язкових порівнянь моделей.
# Задано ЯВНИМ списком, а не як "усі колонки крім двох": при додаванні фіч у
# §10 це змушує свідомо вирішувати, що входить у фіче-сет, і унеможливлює
# випадкове протягування record_id або проміжних колонок у модель.
FEATURES_V0 = NUM_COLS + CAT_COLS

assert ID_COL not in FEATURES_V0
assert TARGET not in FEATURES_V0

# --- Фізичні межі з предметної області -------------------------------------
# Використовуються для ПІДРАХУНКУ порушень в EDA. Межі взяті з логіки домену
# (відвідуваність — відсоток, отже <= 100), а не з даних, тому застосування
# clip() на їх основі не є навченою операцією і не створює витоку.
PHYSICAL_BOUNDS = {
    "learner_age": (10.0, 100.0),    # вік студента; спостережений діапазон 17-24
    "attendance_rate": (0.0, 100.0),
    "engagement_index": (0.0, 100.0),
    "study_time": (0.0, 9.0),        # study_minutes має стелю 528 хв = 8.8 год
    "study_minutes": (0.0, 540.0),
    "sleep_duration": (0.0, 24.0),
    "intake_marker": (0.0, 1.0),
}

# З опису даних таргет у балах 0-100, але EDA (§2 ноутбука 01) показав цензурування:
# спостережений діапазон train — [19.599, 100.0] із точковими масами на обох межах.
# Клампінг робимо за спостереженими межами: прогноз нижче 19.599 гарантовано гірший
# за саме 19.599, бо таких значень у даних не існує.
TARGET_BOUNDS = (19.599, 100.0)
