import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import config as C
import cv as CV
import data as D
from features import make_preprocessor

train_dev = D.load_train_dev()
test = D.load_test()
X, y = D.xy(train_dev)

factory = lambda: Pipeline([("prep", make_preprocessor(scale=True)),
                            ("model", LinearRegression())])

print("CV (Protocol A):")
oof, _ = CV.run_cv(factory, X, y, name="step0_linreg_sanity", protocol="A")

model = factory()
model.fit(X, y)
pred = model.predict(test[C.FEATURES_V0])
CV.make_submission(pred, test[C.ID_COL], name="step0_linreg", clip=C.TARGET_BOUNDS)
