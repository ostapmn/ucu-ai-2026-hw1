# Homework 1: Predicting Assessment Scores

Build a regression pipeline to predict **`assessment_score`**. Compare approaches using **validation RMSE**, then submit predictions for the competition test set.

| File | Rows | Use |
|---|---:|---|
| `train.csv` | 441,000 | Training data with scores; create your validation folds here. |
| `test.csv` | 189,000 | Competition predictions; scores are withheld. |
| `unlabeled.csv` | 270,000 | Optional additional data without labels; not part of the submission. |
| `sample_submission.csv` | 189,000 | Required `record_id,assessment_score` format and ID order. |
| `data_dictionary.csv` | — | Feature meanings and types. |

## Required work

1. Explore the data (EDA) and explain the observations that inform your modeling choices.
2. Compare a training-mean baseline, linear regression, an initial/tuned decision tree, an initial/tuned random forest, and an initial/tuned XGBoost model on the same rows and folds.
3. Develop **Your Pipeline** to improve your leaderboard score: test at least two feature changes with the model settings fixed, and evaluate one selected change on another model as well.
4. As the additional modeling part of **Your Pipeline**, complete at least one further experiment: another model, AutoML, a pretrained model, an ensemble, or additional-data experiments.
5. Select your final approach, refit it, and predict all test rows. Preserve the sample-submission ID order and exclude `record_id` from model features.

Use proper cross-validation and out-of-fold evaluation. Fit learned preprocessing inside each training fold. Start the required model comparisons with the same basic features and appropriate preprocessing. Keep benchmarks consistent, show incremental changes, and explain unsuccessful attempts as well as improvements. If using a smaller sample for an advanced model, compare it with a reference trained on the same rows.

## Submission and grading

Submit predictions on Kaggle, selecting up to two final submissions. Submit **both a report and reproducible code through the LMS**. Choose your own report structure, making it easy to connect the reported experiments to your code. There will also be a one-on-one discussion of your approach, report, and code.

The homework is worth **10 points**: private leaderboard 2, implementation 5, report/discussion 3. The code checkpoints are EDA (0.50), Baseline (0.25), Decision Tree (1.00), Random Forest (1.00), XGBoost (1.00), and Your Pipeline (1.25). Your Pipeline contains feature engineering (0.50) and additional modeling experiments (0.75), graded separately. See Overview for the full requirements.

The competition opens **September 23, 2026, at 08:00 Kyiv time**. Final predictions, code, and the report are due **October 5, 2026, at 23:59 Kyiv time (UTC+3)**.

Searching for and effectively using the original public source data is not prohibited. Complete the required baseline comparisons using the provided train data, and describe additional-data experiments separately. Cite external resources and disclose source matches or overlap with evaluated rows. Do not describe target lookup as clean model validation.

Sharing solution code between participants and plagiarism are prohibited. The full task and grading are in **Overview**, the file descriptions are in **Data**, and the format is in **Submission File**. Check the competition **Timeline** for any announced changes.
