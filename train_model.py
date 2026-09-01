import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

DATA_FOLDER = Path("data")
MODEL_FOLDER = Path("models")

INPUT_FILE = DATA_FOLDER / "premier_league_features.csv"

MODEL_FILE = MODEL_FOLDER / "match_result_model.joblib"
FEATURES_FILE = MODEL_FOLDER / "model_features.json"
METRICS_FILE = MODEL_FOLDER / "model_metrics.json"
PREDICTIONS_FILE = DATA_FOLDER / "test_predictions.csv"


# ---------------------------------------------------------
# DATASET SPLIT
# ---------------------------------------------------------

# Earlier seasons are used to train the model.
TRAIN_SEASONS = [
    "2021-22",
    "2022-23",
    "2023-24",
]

# This season is used to evaluate model choices.
VALIDATION_SEASONS = [
    "2024-25",
]

# This season remains untouched until final testing.
TEST_SEASONS = [
    "2025-26",
]


# ---------------------------------------------------------
# MODEL FEATURES
# ---------------------------------------------------------

# These are all pre-match features.
#
# Do not include:
# Full_Time_Home_Goals
# Full_Time_Away_Goals
# Home_Shots
# Away_Shots
# Current-match cards, corners, fouls or bookmaker outcomes
#
# Those describe what happened during or after the match.

FEATURE_COLUMNS = [
    # Elo
    "Home_Elo",
    "Away_Elo",
    "Elo_Difference",
    "Home_Elo_With_Advantage",
    "Home_Expected_Score",
    "Away_Expected_Score",

    # Home rolling form
    "Home_Form_Points",
    "Home_Form_Goals_For",
    "Home_Form_Goals_Against",
    "Home_Form_Shots_For",
    "Home_Form_Shots_Against",
    "Home_Form_Shots_On_Target_For",
    "Home_Form_Shots_On_Target_Against",
    "Home_Previous_Matches_Available",

    # Away rolling form
    "Away_Form_Points",
    "Away_Form_Goals_For",
    "Away_Form_Goals_Against",
    "Away_Form_Shots_For",
    "Away_Form_Shots_Against",
    "Away_Form_Shots_On_Target_For",
    "Away_Form_Shots_On_Target_Against",
    "Away_Previous_Matches_Available",

    # Difference features
    "Form_Points_Difference",
    "Form_Goals_For_Difference",
    "Form_Goals_Against_Difference",
    "Form_Shots_Difference",
    "Form_Shots_On_Target_Difference",
]

TARGET_COLUMN = "Full_Time_Result"

CLASS_ORDER = ["H", "D", "A"]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found: {INPUT_FILE.resolve()}\n"
        "Run build_features.py first."
    )

df = pd.read_csv(INPUT_FILE)

required_columns = set(
    FEATURE_COLUMNS
    + [
        TARGET_COLUMN,
        "Season",
        "Date",
        "Home_Team",
        "Away_Team",
    ]
)

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "The feature dataset is missing these columns:\n"
        f"{sorted(missing_columns)}"
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "Date",
        "Season",
        TARGET_COLUMN,
    ]
).copy()

df = df[
    df[TARGET_COLUMN].isin(CLASS_ORDER)
].copy()

df = df.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------
# CREATE CHRONOLOGICAL SPLITS
# ---------------------------------------------------------

train_df = df[
    df["Season"].isin(TRAIN_SEASONS)
].copy()

validation_df = df[
    df["Season"].isin(VALIDATION_SEASONS)
].copy()

test_df = df[
    df["Season"].isin(TEST_SEASONS)
].copy()

if train_df.empty:
    raise ValueError(
        "Training dataset is empty. Check your Season values."
    )

if validation_df.empty:
    raise ValueError(
        "Validation dataset is empty. Check your Season values."
    )

if test_df.empty:
    raise ValueError(
        "Test dataset is empty. Check your Season values."
    )


X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COLUMN]

X_validation = validation_df[FEATURE_COLUMNS]
y_validation = validation_df[TARGET_COLUMN]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COLUMN]


# ---------------------------------------------------------
# BUILD MODEL PIPELINE
# ---------------------------------------------------------

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            FEATURE_COLUMNS,
        )
    ],
    remainder="drop",
)

model = LogisticRegression(
    max_iter=3000,
    multi_class="multinomial",
    random_state=42,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------

pipeline.fit(
    X_train,
    y_train,
)


# ---------------------------------------------------------
# EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)

    model_classes = list(
        pipeline.named_steps["model"].classes_
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions,
    )

    loss = log_loss(
        y,
        probabilities,
        labels=model_classes,
    )

    report = classification_report(
        y,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=CLASS_ORDER,
    ).tolist()

    print()
    print(f"----- {name} -----")
    print(f"Matches: {len(y)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )
    print(f"Log loss: {loss:.4f}")
    print()

    print(
        classification_report(
            y,
            predictions,
            labels=CLASS_ORDER,
            target_names=[
                "Home win",
                "Draw",
                "Away win",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "[rows=true, columns=predicted]"
    )
    print(
        pd.DataFrame(
            matrix,
            index=[
                "True_H",
                "True_D",
                "True_A",
            ],
            columns=[
                "Pred_H",
                "Pred_D",
                "Pred_A",
            ],
        )
    )

    return {
        "matches": int(len(y)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "log_loss": float(loss),
        "classification_report": report,
        "confusion_matrix": matrix,
    }


# ---------------------------------------------------------
# VALIDATION RESULTS
# ---------------------------------------------------------

validation_metrics = evaluate_model(
    name="Validation: 2024-25",
    X=X_validation,
    y=y_validation,
)


# ---------------------------------------------------------
# FINAL TEST RESULTS
# ---------------------------------------------------------

test_metrics = evaluate_model(
    name="Test: 2025-26",
    X=X_test,
    y=y_test,
)


# ---------------------------------------------------------
# SAVE TEST PREDICTIONS
# ---------------------------------------------------------

test_predictions = pipeline.predict(X_test)
test_probabilities = pipeline.predict_proba(X_test)

probability_columns = {
    class_name: test_probabilities[:, index]
    for index, class_name in enumerate(
        pipeline.named_steps["model"].classes_
    )
}

test_output = test_df[
    [
        "Date",
        "Season",
        "Home_Team",
        "Away_Team",
        TARGET_COLUMN,
    ]
].copy()

test_output["Predicted_Result"] = (
    test_predictions
)

test_output["Home_Win_Probability"] = (
    probability_columns["H"]
)

test_output["Draw_Probability"] = (
    probability_columns["D"]
)

test_output["Away_Win_Probability"] = (
    probability_columns["A"]
)

test_output["Prediction_Correct"] = (
    test_output["Predicted_Result"]
    == test_output[TARGET_COLUMN]
)

test_output.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ---------------------------------------------------------
# SAVE MODEL AND METADATA
# ---------------------------------------------------------

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

joblib.dump(
    pipeline,
    MODEL_FILE,
)

with open(
    FEATURES_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "class_order": CLASS_ORDER,
            "train_seasons": TRAIN_SEASONS,
            "validation_seasons": VALIDATION_SEASONS,
            "test_seasons": TEST_SEASONS,
        },
        file,
        indent=4,
    )

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        file,
        indent=4,
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("Model training completed successfully.")
print(f"Model saved to: {MODEL_FILE.resolve()}")
print(
    f"Feature list saved to: "
    f"{FEATURES_FILE.resolve()}"
)
print(
    f"Metrics saved to: "
    f"{METRICS_FILE.resolve()}"
)
print(
    f"Test predictions saved to: "
    f"{PREDICTIONS_FILE.resolve()}"
)