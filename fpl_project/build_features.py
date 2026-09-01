import pandas as pd
from collections import defaultdict, deque
from pathlib import Path


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

DATA_FOLDER = Path("data")

INPUT_FILE = DATA_FOLDER / "premier_league_all.csv"
OUTPUT_FILE = DATA_FOLDER / "premier_league_features.csv"

INITIAL_ELO = 1500
PROMOTED_TEAM_ELO = 1425
K_FACTOR = 20
HOME_ADVANTAGE = 100
ROLLING_WINDOW = 5

# At the start of each new season, move each team's Elo
# 25% of the way back toward the average rating of 1500.
NEW_SEASON_REGRESSION = 0.25


# These values are used when a team has no previous matches
# available in the current season.
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
# LOAD DATA
# ---------------------------------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE.resolve()}"
    )

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "Date",
    "Season",
    "Home_Team",
    "Away_Team",
    "Full_Time_Home_Goals",
    "Full_Time_Away_Goals",
    "Full_Time_Result",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        "The input file is missing these required columns: "
        f"{sorted(missing_columns)}"
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

# Remove rows where the date could not be parsed.
df = df.dropna(subset=["Date"]).copy()

sort_columns = ["Date"]

if "Time" in df.columns:
    sort_columns.append("Time")

df = df.sort_values(sort_columns).reset_index(drop=True)


# ---------------------------------------------------------
# ELO FUNCTIONS
# ---------------------------------------------------------

def expected_score(
    rating_a: float,
    rating_b: float
) -> float:
    """
    Calculate Team A's expected Elo result against Team B.

    The result is between 0 and 1:
    1 means Team A is strongly expected to win.
    0.5 means both teams are evenly matched.
    0 means Team A is strongly expected to lose.
    """
    return 1 / (
        1 + 10 ** ((rating_b - rating_a) / 400)
    )


def result_to_score(
    result: str
) -> tuple[float, float]:
    """
    Convert the match result into Elo result values.

    Win  = 1.0
    Draw = 0.5
    Loss = 0.0
    """
    if result == "H":
        return 1.0, 0.0

    if result == "A":
        return 0.0, 1.0

    return 0.5, 0.5


def result_to_points(
    result: str
) -> tuple[int, int]:
    """
    Convert the match result into football league points.

    Win  = 3 points
    Draw = 1 point
    Loss = 0 points
    """
    if result == "H":
        return 3, 0

    if result == "A":
        return 0, 3

    return 1, 1


def regress_elo_toward_mean(
    rating: float
) -> float:
    """
    Pull a team's Elo rating partly toward the league average
    at the start of a new season.
    """
    return (
        INITIAL_ELO
        + (rating - INITIAL_ELO)
        * (1 - NEW_SEASON_REGRESSION)
    )


# ---------------------------------------------------------
# ROLLING-FORM FUNCTIONS
# ---------------------------------------------------------

def average_history(
    history: deque,
    stat_name: str
) -> float:
    """
    Calculate the average of a statistic from the matches
    currently stored in a team's rolling history.
    """
    if len(history) == 0:
        return DEFAULT_FORM[stat_name]

    values = [
        match[stat_name]
        for match in history
    ]

    return sum(values) / len(values)


def get_team_form(
    history: deque
) -> dict:
    """
    Calculate a team's rolling averages using only matches
    played before the current match.
    """
    return {
        "Form_Points": average_history(
            history,
            "points"
        ),
        "Form_Goals_For": average_history(
            history,
            "goals_for"
        ),
        "Form_Goals_Against": average_history(
            history,
            "goals_against"
        ),
        "Form_Shots_For": average_history(
            history,
            "shots_for"
        ),
        "Form_Shots_Against": average_history(
            history,
            "shots_against"
        ),
        "Form_Shots_On_Target_For": average_history(
            history,
            "shots_on_target_for"
        ),
        "Form_Shots_On_Target_Against": average_history(
            history,
            "shots_on_target_against"
        ),
        "Previous_Matches_Available": len(history),
    }


def safe_numeric_value(
    value,
    default: float = 0.0
) -> float:
    """
    Return a numeric value, replacing missing values with
    a safe default.
    """
    if pd.isna(value):
        return default

    return float(value)


# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

# Every unseen team automatically begins with 1500 Elo.
elo_ratings = defaultdict(
    lambda: INITIAL_ELO
)

# Each team stores only its previous five league matches.
team_histories = defaultdict(
    lambda: deque(maxlen=ROLLING_WINDOW)
)

feature_rows = []

# Used to detect when the dataset moves into a new season.
current_season = None


# ---------------------------------------------------------
# PROCESS MATCHES CHRONOLOGICALLY
# ---------------------------------------------------------

for _, match in df.iterrows():

    season = match["Season"]

    # -----------------------------------------------------
    # NEW-SEASON HANDLING
    # -----------------------------------------------------

    if season != current_season:

        # Do not regress Elo before the first season.
        if current_season is not None:

            # Keep previous-season strength, but reduce its
            # influence by pulling Elo toward 1500.
            for team in list(elo_ratings.keys()):
                elo_ratings[team] = (
                    regress_elo_toward_mean(
                        elo_ratings[team]
                    )
                )

            # Reset short-term league form.
            # Friendly matches are not included.
            team_histories.clear()

        current_season = season

    home_team = match["Home_Team"]
    away_team = match["Away_Team"]
    result = match["Full_Time_Result"]

    # Skip incomplete or invalid matches.
    if result not in {"H", "D", "A"}:
        continue

    # -----------------------------------------------------
    # 1. GET PRE-MATCH ELO RATINGS
    # -----------------------------------------------------

    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]

    home_elo_with_advantage = (
        home_elo + HOME_ADVANTAGE
    )

    home_expected = expected_score(
        home_elo_with_advantage,
        away_elo
    )

    away_expected = 1 - home_expected

    # -----------------------------------------------------
    # 2. GET PRE-MATCH ROLLING FORM
    # -----------------------------------------------------

    home_form = get_team_form(
        team_histories[home_team]
    )

    away_form = get_team_form(
        team_histories[away_team]
    )

    # -----------------------------------------------------
    # 3. SAVE FEATURES BEFORE USING CURRENT MATCH DATA
    # -----------------------------------------------------

    feature_row = match.to_dict()

    feature_row["Home_Elo"] = home_elo
    feature_row["Away_Elo"] = away_elo

    feature_row["Elo_Difference"] = (
        home_elo - away_elo
    )

    feature_row["Home_Elo_With_Advantage"] = (
        home_elo_with_advantage
    )

    feature_row["Home_Expected_Score"] = (
        home_expected
    )

    feature_row["Away_Expected_Score"] = (
        away_expected
    )

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

    feature_rows.append(feature_row)

    # -----------------------------------------------------
    # 4. UPDATE ELO AFTER SAVING PRE-MATCH FEATURES
    # -----------------------------------------------------

    home_actual, away_actual = result_to_score(
        result
    )

    home_points, away_points = result_to_points(
        result
    )

    new_home_elo = home_elo + K_FACTOR * (
        home_actual - home_expected
    )

    new_away_elo = away_elo + K_FACTOR * (
        away_actual - away_expected
    )

    elo_ratings[home_team] = new_home_elo
    elo_ratings[away_team] = new_away_elo

    # -----------------------------------------------------
    # 5. PREPARE CURRENT MATCH FOR FUTURE FORM FEATURES
    # -----------------------------------------------------

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
            0
        )
    )

    away_shots_on_target = safe_numeric_value(
        match.get(
            "Away_Shots_On_Target",
            0
        )
    )

    home_match_history = {
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

    away_match_history = {
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

    # This happens only after the current match's feature
    # row has already been saved.
    team_histories[home_team].append(
        home_match_history
    )

    team_histories[away_team].append(
        away_match_history
    )


# ---------------------------------------------------------
# SAVE FEATURE DATASET
# ---------------------------------------------------------

features_df = pd.DataFrame(feature_rows)

if features_df.empty:
    raise ValueError(
        "No feature rows were created. Check the input data."
    )

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Feature dataset created successfully.")
print(f"Saved to: {OUTPUT_FILE.resolve()}")
print(f"Rows: {features_df.shape[0]}")
print(f"Columns: {features_df.shape[1]}")
print()


# ---------------------------------------------------------
# DISPLAY A PREVIEW
# ---------------------------------------------------------

display_columns = [
    "Date",
    "Season",
    "Home_Team",
    "Away_Team",
    "Home_Elo",
    "Away_Elo",
    "Elo_Difference",
    "Home_Form_Points",
    "Away_Form_Points",
    "Home_Previous_Matches_Available",
    "Away_Previous_Matches_Available",
    "Form_Points_Difference",
    "Full_Time_Result",
]

print(
    features_df[
        display_columns
    ].head(20).to_string(index=False)
)