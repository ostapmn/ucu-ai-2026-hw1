from pathlib import Path

SEED = 42
N_FOLDS = 5
HOLDOUT_FRAC = 0.10

SEARCH_N_FOLDS = 3
SEARCH_SAMPLE = 150_000

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

RAW_TRAIN = DATA_DIR / "train.csv"
RAW_TEST = DATA_DIR / "test.csv"
RAW_UNLABELED = DATA_DIR / "unlabeled.csv"
SAMPLE_SUB = DATA_DIR / "sample_submission.csv"

TRAIN_DEV = DATA_DIR / "train_dev.parquet"
HOLDOUT = DATA_DIR / "holdout.parquet"
TEST = DATA_DIR / "test.parquet"
UNLABELED = DATA_DIR / "unlabeled.parquet"

EXPERIMENTS_DIR = ROOT / "experiments"
RESULTS_CSV = EXPERIMENTS_DIR / "results.csv"
OOF_DIR = EXPERIMENTS_DIR / "oof"
SUBMISSIONS_DIR = ROOT / "submissions"
FIGURES_DIR = ROOT / "figures"

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

FEATURES_V0 = NUM_COLS + CAT_COLS

assert ID_COL not in FEATURES_V0
assert TARGET not in FEATURES_V0

PHYSICAL_BOUNDS = {
    "learner_age": (10.0, 100.0),
    "attendance_rate": (0.0, 100.0),
    "engagement_index": (0.0, 100.0),
    "study_time": (0.0, 9.0),
    "study_minutes": (0.0, 540.0),
    "sleep_duration": (0.0, 24.0),
    "intake_marker": (0.0, 1.0),
}

TARGET_BOUNDS = (19.599, 100.0)
