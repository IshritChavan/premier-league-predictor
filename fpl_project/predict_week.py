import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd


# ---------------------------------------------------------
# FILE LOCATIONS
# ---------------------------------------------------------

DATA_FOLDER = Path("data")
MODEL_FOLDER = Path("models")

HISTORICAL_FILE = (
    DATA_FOLDER / "premier_league_all.csv"
)

FIXTURES_FILE = (
    DATA_FOLDER / "upcoming_fixtures.csv"
)

OUTPUT_FILE = (
    DATA_FOLDER / "weekly_predictions.csv"
)

MODEL_FILE = (
    MODEL_FOLDER / "match_result_model.joblib"
)

FEATURES_FILE = (
    MODEL_FOLDER / "model_features.json"
)


# ---------------------------------------------------------
# ELO AND FORM SETTINGS
# ---------------------------------------------------------

INITIAL_ELO = 1500
PROMOTED_TEAM_ELO = 1425

K_FACTOR = 20
HOME_ADVANTAGE = 100
ROLLING_WINDOW = 5
NEW_SEASON_REGRESSION = 0.25


# Promoted teams are inferred automatically from the current season instead
# of being maintained by hand. The set is populated after historical data loads.
PROMOTED_TEAMS = set()


DEFAULT_FORM = {
    "points": 1.3,
    "goals_for": 1.4,
    "goals_against": 1.4,
    "shots_for": 12.0,
    "shots_against": 12.0,
    "shots_on_target_for": 4.0,
    "shots_on_target_against": 4.0,
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def expected_score(
    rating_a: float,
    rating_b: float,
) -> float:
    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def result_to_score(
    result: str,
) -> tuple[float, float]:
    if result == "H":
        return 1.0, 0.0

    if result == "A":
        return 0.0, 1.0

    return 0.5, 0.5


def result_to_points(
    result: str,
) -> tuple[int, int]:
    if result == "H":
        return 3, 0

    if result == "A":
        return 0, 3

    return 1, 1


def regress_elo_toward_mean(
    rating: float,
) -> float:
    return (
        INITIAL_ELO
        + (rating - INITIAL_ELO)
        * (1 - NEW_SEASON_REGRESSION)
    )


def safe_numeric_value(
    value,
    default: float = 0.0,
) -> float:
    if pd.isna(value):
        return default

    return float(value)


def average_history(
    history: deque,
    stat_name: str,
) -> float:
    if len(history) == 0:
        return DEFAULT_FORM[stat_name]

    values = [
        game[stat_name]
        for game in history
    ]

    return sum(values) / len(values)


def get_team_form(
    history: deque,
) -> dict:
    return {
        "Form_Points": average_history(
            history,
            "points",
        ),
        "Form_Goals_For": average_history(
            history,
            "goals_for",
        ),
        "Form_Goals_Against": average_history(
            history,
            "goals_against",
        ),
        "Form_Shots_For": average_history(
            history,
            "shots_for",
        ),
        "Form_Shots_Against": average_history(
            history,
            "shots_against",
        ),
        "Form_Shots_On_Target_For": average_history(
            history,
            "shots_on_target_for",
        ),
        "Form_Shots_On_Target_Against": average_history(
            history,
            "shots_on_target_against",
        ),
        "Previous_Matches_Available": len(history),
    }


def confidence_label(
    probability: float,
) -> str:
    if probability >= 0.80:
        return "High"

    if probability >= 0.65:
        return "Medium"

    return "Low"


# ---------------------------------------------------------
# CHECK FILES
# ---------------------------------------------------------

required_files = [
    HISTORICAL_FILE,
    FIXTURES_FILE,
    MODEL_FILE,
    FEATURES_FILE,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: "
            f"{file_path.resolve()}"
        )


# ---------------------------------------------------------
# LOAD MODEL AND FEATURE LIST
# ---------------------------------------------------------

model = joblib.load(MODEL_FILE)

with open(
    FEATURES_FILE,
    "r",
    encoding="utf-8",
) as file:
    model_metadata = json.load(file)

FEATURE_COLUMNS = model_metadata["feature_columns"]


# ---------------------------------------------------------
# LOAD HISTORICAL MATCHES
# ---------------------------------------------------------

matches = pd.read_csv(HISTORICAL_FILE)

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce",
)

matches = matches.dropna(
    subset=[
        "Date",
        "Season",
        "Home_Team",
        "Away_Team",
    ]
).copy()

sort_columns = ["Date"]

if "Time" in matches.columns:
    sort_columns.append("Time")

matches = (
    matches
    .sort_values(sort_columns)
    .reset_index(drop=True)
)

# Infer newly promoted teams for the latest season in the master dataset.
# A promoted side is one present in the latest season but absent from the
# immediately preceding season.
season_order = (
    matches[["Season", "Date"]]
    .drop_duplicates()
    .groupby("Season", as_index=False)["Date"]
    .min()
    .sort_values("Date")["Season"]
    .tolist()
)

if len(season_order) >= 2:
    latest_season = season_order[-1]
    previous_season = season_order[-2]

    latest_rows = matches[matches["Season"] == latest_season]
    previous_rows = matches[matches["Season"] == previous_season]

    latest_teams = set(latest_rows["Home_Team"]) | set(latest_rows["Away_Team"])
    previous_teams = set(previous_rows["Home_Team"]) | set(previous_rows["Away_Team"])
    PROMOTED_TEAMS.update(latest_teams - previous_teams)


# ---------------------------------------------------------
# RECONSTRUCT CURRENT ELO AND FORM
# ---------------------------------------------------------

elo_ratings = defaultdict(
    lambda: INITIAL_ELO
)

team_histories = defaultdict(
    lambda: deque(maxlen=ROLLING_WINDOW)
)

current_season = None


for _, match in matches.iterrows():

    result = match["Full_Time_Result"]

    if result not in {"H", "D", "A"}:
        continue

    season = match["Season"]

    # New season handling
    if season != current_season:

        if current_season is not None:

            for team in list(elo_ratings.keys()):
                elo_ratings[team] = (
                    regress_elo_toward_mean(
                        elo_ratings[team]
                    )
                )

            team_histories.clear()

        current_season = season

    home_team = match["Home_Team"]
    away_team = match["Away_Team"]

    if home_team not in elo_ratings and home_team in PROMOTED_TEAMS:
        elo_ratings[home_team] = PROMOTED_TEAM_ELO
    if away_team not in elo_ratings and away_team in PROMOTED_TEAMS:
        elo_ratings[away_team] = PROMOTED_TEAM_ELO

    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    home_expected = expected_score(
        home_elo + HOME_ADVANTAGE,
        away_elo,
    )

    away_expected = 1 - home_expected

    home_actual, away_actual = result_to_score(
        result
    )

    home_points, away_points = result_to_points(
        result
    )

    # Update Elo
    elo_ratings[home_team] = (
        home_elo
        + K_FACTOR
        * (home_actual - home_expected)
    )

    elo_ratings[away_team] = (
        away_elo
        + K_FACTOR
        * (away_actual - away_expected)
    )

    # Current-match statistics
    home_goals = safe_numeric_value(
        match["Full_Time_Home_Goals"]
    )

    away_goals = safe_numeric_value(
        match["Full_Time_Away_Goals"]
    )

    home_shots = safe_numeric_value(
        match.get("Home_Shots", 0)
    )

    away_shots = safe_numeric_value(
        match.get("Away_Shots", 0)
    )

    home_shots_on_target = safe_numeric_value(
        match.get(
            "Home_Shots_On_Target",
            0,
        )
    )

    away_shots_on_target = safe_numeric_value(
        match.get(
            "Away_Shots_On_Target",
            0,
        )
    )

    # Add completed match to rolling history
    team_histories[home_team].append(
        {
            "points": home_points,
            "goals_for": home_goals,
            "goals_against": away_goals,
            "shots_for": home_shots,
            "shots_against": away_shots,
            "shots_on_target_for": (
                home_shots_on_target
            ),
            "shots_on_target_against": (
                away_shots_on_target
            ),
        }
    )

    team_histories[away_team].append(
        {
            "points": away_points,
            "goals_for": away_goals,
            "goals_against": home_goals,
            "shots_for": away_shots,
            "shots_against": home_shots,
            "shots_on_target_for": (
                away_shots_on_target
            ),
            "shots_on_target_against": (
                home_shots_on_target
            ),
        }
    )


# ---------------------------------------------------------
# LOAD UPCOMING FIXTURES
# ---------------------------------------------------------

fixtures = pd.read_csv(FIXTURES_FILE)

required_fixture_columns = {
    "Date",
    "Home_Team",
    "Away_Team",
    "Season",
}

missing_fixture_columns = (
    required_fixture_columns - set(fixtures.columns)
)

if missing_fixture_columns:
    raise ValueError(
        "upcoming_fixtures.csv is missing: "
        f"{sorted(missing_fixture_columns)}"
    )

fixtures["Date"] = pd.to_datetime(
    fixtures["Date"],
    errors="coerce",
)

fixtures = fixtures.dropna(
    subset=[
        "Date",
        "Home_Team",
        "Away_Team",
        "Season",
    ]
).copy()

fixtures = fixtures.sort_values(
    ["Date", "Time"]
    if "Time" in fixtures.columns
    else ["Date"]
).reset_index(drop=True)

# If the current season has not yet produced a completed result, promoted
# teams can still be identified from the upcoming fixture list.
if season_order:
    previous_rows = matches[matches["Season"] == season_order[-1]]
    previous_teams = set(previous_rows["Home_Team"]) | set(previous_rows["Away_Team"])
    fixture_teams = set(fixtures["Home_Team"]) | set(fixtures["Away_Team"])
    if fixtures["Season"].iloc[0] != season_order[-1]:
        PROMOTED_TEAMS.update(fixture_teams - previous_teams)


# ---------------------------------------------------------
# MOVE INTO THE FIXTURE SEASON
# ---------------------------------------------------------

fixture_seasons = fixtures["Season"].unique()

if len(fixture_seasons) != 1:
    raise ValueError(
        "For now, upcoming_fixtures.csv should contain "
        "fixtures from one season only."
    )

fixture_season = fixture_seasons[0]

if fixture_season != current_season:

    for team in list(elo_ratings.keys()):
        elo_ratings[team] = (
            regress_elo_toward_mean(
                elo_ratings[team]
            )
        )

    team_histories.clear()

    current_season = fixture_season


# ---------------------------------------------------------
# TEAM ELO LOOKUP
# ---------------------------------------------------------

def get_fixture_team_elo(
    team: str,
) -> float:

    if team in elo_ratings:
        return elo_ratings[team]

    if team in PROMOTED_TEAMS:
        elo_ratings[team] = PROMOTED_TEAM_ELO
        return PROMOTED_TEAM_ELO

    raise ValueError(
        f"Unknown team: {team}\n"
        "Check that its name matches the historical data. "
        "If it is newly promoted, add it to PROMOTED_TEAMS."
    )


# ---------------------------------------------------------
# BUILD FIXTURE FEATURES
# ---------------------------------------------------------

fixture_feature_rows = []


for _, fixture in fixtures.iterrows():

    home_team = fixture["Home_Team"]
    away_team = fixture["Away_Team"]

    home_elo = get_fixture_team_elo(home_team)
    away_elo = get_fixture_team_elo(away_team)

    home_elo_with_advantage = (
        home_elo + HOME_ADVANTAGE
    )

    home_expected = expected_score(
        home_elo_with_advantage,
        away_elo,
    )

    away_expected = 1 - home_expected

    home_form = get_team_form(
        team_histories[home_team]
    )

    away_form = get_team_form(
        team_histories[away_team]
    )

    feature_row = {
        "Home_Elo": home_elo,
        "Away_Elo": away_elo,
        "Elo_Difference": (
            home_elo - away_elo
        ),
        "Home_Elo_With_Advantage": (
            home_elo_with_advantage
        ),
        "Home_Expected_Score": home_expected,
        "Away_Expected_Score": away_expected,
    }

    for column, value in home_form.items():
        feature_row[f"Home_{column}"] = value

    for column, value in away_form.items():
        feature_row[f"Away_{column}"] = value

    feature_row["Form_Points_Difference"] = (
        home_form["Form_Points"]
        - away_form["Form_Points"]
    )

    feature_row["Form_Goals_For_Difference"] = (
        home_form["Form_Goals_For"]
        - away_form["Form_Goals_For"]
    )

    feature_row[
        "Form_Goals_Against_Difference"
    ] = (
        home_form["Form_Goals_Against"]
        - away_form["Form_Goals_Against"]
    )

    feature_row["Form_Shots_Difference"] = (
        home_form["Form_Shots_For"]
        - away_form["Form_Shots_For"]
    )

    feature_row[
        "Form_Shots_On_Target_Difference"
    ] = (
        home_form["Form_Shots_On_Target_For"]
        - away_form["Form_Shots_On_Target_For"]
    )

    fixture_feature_rows.append(feature_row)


fixture_features = pd.DataFrame(
    fixture_feature_rows
)

missing_model_features = (
    set(FEATURE_COLUMNS)
    - set(fixture_features.columns)
)

if missing_model_features:
    raise ValueError(
        "Could not build these model features: "
        f"{sorted(missing_model_features)}"
    )

fixture_features = fixture_features[
    FEATURE_COLUMNS
]


# ---------------------------------------------------------
# MAKE PREDICTIONS
# ---------------------------------------------------------

predicted_classes = model.predict(
    fixture_features
)

predicted_probabilities = model.predict_proba(
    fixture_features
)

model_classes = list(
    model.named_steps["model"].classes_
)

class_indexes = {
    class_name: index
    for index, class_name in enumerate(model_classes)
}


# ---------------------------------------------------------
# CREATE WEEKLY OUTPUT
# ---------------------------------------------------------

prediction_rows = []


for index, fixture in fixtures.iterrows():

    home_probability = float(
        predicted_probabilities[
            index,
            class_indexes["H"],
        ]
    )

    draw_probability = float(
        predicted_probabilities[
            index,
            class_indexes["D"],
        ]
    )

    away_probability = float(
        predicted_probabilities[
            index,
            class_indexes["A"],
        ]
    )

    home_or_draw_probability = (
        home_probability + draw_probability
    )

    draw_or_away_probability = (
        draw_probability + away_probability
    )

    market_probabilities = {
        "Home Win": home_probability,
        "Draw": draw_probability,
        "Away Win": away_probability,
        "Home or Draw": (
            home_or_draw_probability
        ),
        "Draw or Away": (
            draw_or_away_probability
        ),
    }

    highest_probability_market = max(
        market_probabilities,
        key=market_probabilities.get,
    )

    highest_probability = (
        market_probabilities[
            highest_probability_market
        ]
    )

    single_result_probabilities = {
        "Home Win": home_probability,
        "Draw": draw_probability,
        "Away Win": away_probability,
    }

    best_single_result = max(
        single_result_probabilities,
        key=single_result_probabilities.get,
    )

    best_single_probability = (
        single_result_probabilities[
            best_single_result
        ]
    )

    prediction_row = {
        "Date": fixture["Date"].date(),
        "Time": fixture.get("Time", ""),
        "Season": fixture["Season"],
        "Matchweek": fixture.get(
            "Matchweek",
            "",
        ),
        "Home_Team": fixture["Home_Team"],
        "Away_Team": fixture["Away_Team"],

        "Predicted_Result": (
            predicted_classes[index]
        ),

        "Home_Win_Probability": round(
            home_probability,
            4,
        ),

        "Draw_Probability": round(
            draw_probability,
            4,
        ),

        "Away_Win_Probability": round(
            away_probability,
            4,
        ),

        "Home_Or_Draw_Probability": round(
            home_or_draw_probability,
            4,
        ),

        "Draw_Or_Away_Probability": round(
            draw_or_away_probability,
            4,
        ),

        "Best_Single_Result": (
            best_single_result
        ),

        "Best_Single_Probability": round(
            best_single_probability,
            4,
        ),

        "Highest_Probability_Market": (
            highest_probability_market
        ),

        "Highest_Probability": round(
            highest_probability,
            4,
        ),

        "Confidence": confidence_label(
            highest_probability
        ),
    }

    prediction_rows.append(prediction_row)


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_df = predictions_df.sort_values(
    "Highest_Probability",
    ascending=False,
).reset_index(drop=True)

# ---------------------------------------------------------
# SAVE CURRENT ELO RANKINGS
# ---------------------------------------------------------

ELO_FILE = DATA_FOLDER / "current_elo_ratings.csv"

elo_rows = [
    {"Team": team, "Elo": round(float(rating), 2)}
    for team, rating in elo_ratings.items()
]

elo_df = (
    pd.DataFrame(elo_rows)
    .sort_values("Elo", ascending=False)
    .reset_index(drop=True)
)
elo_df.insert(0, "Rank", range(1, len(elo_df) + 1))
elo_df.to_csv(ELO_FILE, index=False)


# ---------------------------------------------------------
# BUILD FIVE PARLAY TIERS
# ---------------------------------------------------------

PARLAYS_FILE = DATA_FOLDER / "weekly_parlays.csv"


def get_best_double_chance(row: pd.Series) -> tuple[str, float]:
    """
    Return the higher-probability double-chance market.
    """
    home_or_draw = float(
        row["Home_Or_Draw_Probability"]
    )

    draw_or_away = float(
        row["Draw_Or_Away_Probability"]
    )

    if home_or_draw >= draw_or_away:
        return "Home or Draw", home_or_draw

    return "Draw or Away", draw_or_away


def get_best_single_result(row: pd.Series) -> tuple[str, float]:
    """
    Return the highest-probability single-result market.
    """
    markets = {
        "Home Win": float(
            row["Home_Win_Probability"]
        ),
        "Draw": float(
            row["Draw_Probability"]
        ),
        "Away Win": float(
            row["Away_Win_Probability"]
        ),
    }

    best_market = max(
        markets,
        key=markets.get,
    )

    return best_market, markets[best_market]


def build_candidate_table(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    candidates = []

    for _, row in predictions.iterrows():

        fixture_name = (
            f"{row['Home_Team']} vs "
            f"{row['Away_Team']}"
        )

        double_market, double_probability = (
            get_best_double_chance(row)
        )

        single_market, single_probability = (
            get_best_single_result(row)
        )

        candidates.append(
            {
                "Date": row["Date"],
                "Home_Team": row["Home_Team"],
                "Away_Team": row["Away_Team"],
                "Fixture": fixture_name,
                "Double_Chance_Market": double_market,
                "Double_Chance_Probability": (
                    double_probability
                ),
                "Single_Result_Market": single_market,
                "Single_Result_Probability": (
                    single_probability
                ),
            }
        )

    return pd.DataFrame(candidates)


candidate_df = build_candidate_table(
    predictions_df
)


def create_parlay(
    parlay_name: str,
    risk_level: str,
    legs: list[dict],
) -> dict:
    """
    Create one parlay summary.

    Joint probability is a rough estimate calculated by
    multiplying the model probabilities of each leg.
    This assumes independence, which is not always true.
    """
    estimated_probability = 1.0

    leg_descriptions = []

    for leg in legs:
        probability = float(
            leg["Probability"]
        )

        estimated_probability *= probability

        leg_descriptions.append(
            f"{leg['Fixture']} — "
            f"{leg['Market']} "
            f"({probability:.1%})"
        )

    return {
        "Parlay": parlay_name,
        "Risk_Level": risk_level,
        "Number_Of_Legs": len(legs),
        "Estimated_Hit_Probability": round(
            estimated_probability,
            4,
        ),
        "Estimated_Hit_Percentage": round(
            estimated_probability * 100,
            2,
        ),
        "Legs": " | ".join(
            leg_descriptions
        ),
    }


# Sort candidates for double-chance parlays
double_candidates = candidate_df.sort_values(
    "Double_Chance_Probability",
    ascending=False,
).reset_index(drop=True)

# Sort candidates for single-result parlays
single_candidates = candidate_df.sort_values(
    "Single_Result_Probability",
    ascending=False,
).reset_index(drop=True)


def select_double_chance_legs(
    number_of_legs: int,
) -> list[dict]:
    selected = []

    for _, row in double_candidates.head(
        number_of_legs
    ).iterrows():

        selected.append(
            {
                "Fixture": row["Fixture"],
                "Market": row[
                    "Double_Chance_Market"
                ],
                "Probability": row[
                    "Double_Chance_Probability"
                ],
            }
        )

    return selected


def select_single_result_legs(
    number_of_legs: int,
) -> list[dict]:
    selected = []

    for _, row in single_candidates.head(
        number_of_legs
    ).iterrows():

        selected.append(
            {
                "Fixture": row["Fixture"],
                "Market": row[
                    "Single_Result_Market"
                ],
                "Probability": row[
                    "Single_Result_Probability"
                ],
            }
        )

    return selected


def select_mixed_legs() -> list[dict]:
    """
    Medium-risk parlay:
    two strong double-chance picks and
    two stronger single-result picks.
    """
    selected = []
    used_fixtures = set()

    for _, row in double_candidates.iterrows():

        fixture = row["Fixture"]

        if fixture in used_fixtures:
            continue

        selected.append(
            {
                "Fixture": fixture,
                "Market": row[
                    "Double_Chance_Market"
                ],
                "Probability": row[
                    "Double_Chance_Probability"
                ],
            }
        )

        used_fixtures.add(fixture)

        if len(selected) == 2:
            break

    for _, row in single_candidates.iterrows():

        fixture = row["Fixture"]

        if fixture in used_fixtures:
            continue

        selected.append(
            {
                "Fixture": fixture,
                "Market": row[
                    "Single_Result_Market"
                ],
                "Probability": row[
                    "Single_Result_Probability"
                ],
            }
        )

        used_fixtures.add(fixture)

        if len(selected) == 4:
            break

    return selected


parlays = [
    create_parlay(
        parlay_name="Parlay 1",
        risk_level="Highest Confidence",
        legs=select_double_chance_legs(2),
    ),

    create_parlay(
        parlay_name="Parlay 2",
        risk_level="Low Risk",
        legs=select_double_chance_legs(3),
    ),

    create_parlay(
        parlay_name="Parlay 3",
        risk_level="Medium Risk",
        legs=select_mixed_legs(),
    ),

    create_parlay(
        parlay_name="Parlay 4",
        risk_level="High Risk",
        legs=select_single_result_legs(5),
    ),

    create_parlay(
        parlay_name="Parlay 5",
        risk_level="Extreme Risk",
        legs=select_single_result_legs(7),
    ),
]

parlays_df = pd.DataFrame(parlays)

parlays_df.to_csv(
    PARLAYS_FILE,
    index=False,
)

predictions_df.to_csv(
    OUTPUT_FILE,
    index=False,
)

# ---------------------------------------------------------
# RANK ALL INDIVIDUAL BETTING MARKETS
# ---------------------------------------------------------

ALL_MARKETS_FILE = DATA_FOLDER / "ranked_all_markets.csv"
OUTRIGHT_MARKETS_FILE = DATA_FOLDER / "ranked_outright_predictions.csv"

all_market_rows = []
outright_rows = []

for _, row in predictions_df.iterrows():

    fixture_name = (
        f"{row['Home_Team']} vs "
        f"{row['Away_Team']}"
    )

    all_markets = {
        "Home Win": float(
            row["Home_Win_Probability"]
        ),
        "Draw": float(
            row["Draw_Probability"]
        ),
        "Away Win": float(
            row["Away_Win_Probability"]
        ),
        "Home or Draw": float(
            row["Home_Or_Draw_Probability"]
        ),
        "Draw or Away": float(
            row["Draw_Or_Away_Probability"]
        ),
    }

    outright_markets = {
        "Home Win": float(
            row["Home_Win_Probability"]
        ),
        "Draw": float(
            row["Draw_Probability"]
        ),
        "Away Win": float(
            row["Away_Win_Probability"]
        ),
    }

    for market, probability in all_markets.items():
        all_market_rows.append(
            {
                "Date": row["Date"],
                "Home_Team": row["Home_Team"],
                "Away_Team": row["Away_Team"],
                "Fixture": fixture_name,
                "Market": market,
                "Probability": probability,
                "Probability_Percentage": round(
                    probability * 100,
                    2,
                ),
            }
        )

    for market, probability in outright_markets.items():
        outright_rows.append(
            {
                "Date": row["Date"],
                "Home_Team": row["Home_Team"],
                "Away_Team": row["Away_Team"],
                "Fixture": fixture_name,
                "Prediction": market,
                "Probability": probability,
                "Probability_Percentage": round(
                    probability * 100,
                    2,
                ),
            }
        )


ranked_all_markets_df = pd.DataFrame(
    all_market_rows
).sort_values(
    "Probability",
    ascending=False,
).reset_index(drop=True)

ranked_outright_df = pd.DataFrame(
    outright_rows
).sort_values(
    "Probability",
    ascending=False,
).reset_index(drop=True)


# Add ranking numbers
ranked_all_markets_df.insert(
    0,
    "Rank",
    range(
        1,
        len(ranked_all_markets_df) + 1,
    ),
)

ranked_outright_df.insert(
    0,
    "Rank",
    range(
        1,
        len(ranked_outright_df) + 1,
    ),
)


# Save ranked outputs
ranked_all_markets_df.to_csv(
    ALL_MARKETS_FILE,
    index=False,
)

ranked_outright_df.to_csv(
    OUTRIGHT_MARKETS_FILE,
    index=False,
)


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

print()
print("Weekly predictions created successfully.")
print(f"Saved to: {OUTPUT_FILE.resolve()}")
print()

display_columns = [
    "Date",
    "Home_Team",
    "Away_Team",
    "Home_Win_Probability",
    "Draw_Probability",
    "Away_Win_Probability",
    "Home_Or_Draw_Probability",
    "Draw_Or_Away_Probability",
    "Highest_Probability_Market",
    "Highest_Probability",
]

print(
    predictions_df[
        display_columns
    ].to_string(index=False)
)

print()
print("Five parlay tiers created.")
print(f"Saved to: {PARLAYS_FILE.resolve()}")
print()

print(
    parlays_df[
        [
            "Parlay",
            "Risk_Level",
            "Number_Of_Legs",
            "Estimated_Hit_Percentage",
            "Legs",
        ]
    ].to_string(index=False)
)

print()
print("----- TOP INDIVIDUAL MARKETS -----")
print()

print(
    ranked_all_markets_df[
        [
            "Rank",
            "Fixture",
            "Market",
            "Probability_Percentage",
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print()
print("----- TOP OUTRIGHT PREDICTIONS -----")
print()

print(
    ranked_outright_df[
        [
            "Rank",
            "Fixture",
            "Prediction",
            "Probability_Percentage",
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print()
print(
    f"All ranked markets saved to: "
    f"{ALL_MARKETS_FILE.resolve()}"
)

print(
    f"Outright predictions saved to: "
    f"{OUTRIGHT_MARKETS_FILE.resolve()}"
)