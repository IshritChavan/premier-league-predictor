"""Download completed 2026-27 EPL results and rebuild the master match history."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from project_config import (
    BASE_HISTORICAL_FILE,
    CURRENT_SEASON,
    CURRENT_SEASON_RAW_FILE,
    FOOTBALL_DATA_CO_UK_URL,
    MASTER_MATCH_FILE,
    internal_team_name,
)


RAW_TO_PROJECT_COLUMNS = {
    "Div": "Division",
    "Date": "Date",
    "Time": "Time",
    "HomeTeam": "Home_Team",
    "AwayTeam": "Away_Team",
    "FTHG": "Full_Time_Home_Goals",
    "FTAG": "Full_Time_Away_Goals",
    "FTR": "Full_Time_Result",
    "HTHG": "Half_Time_Home_Goals",
    "HTAG": "Half_Time_Away_Goals",
    "HTR": "Half_Time_Result",
    "Referee": "Referee",
    "HS": "Home_Shots",
    "AS": "Away_Shots",
    "HST": "Home_Shots_On_Target",
    "AST": "Away_Shots_On_Target",
    "HF": "Home_Fouls",
    "AF": "Away_Fouls",
    "HC": "Home_Corners",
    "AC": "Away_Corners",
    "HY": "Home_Yellow_Cards",
    "AY": "Away_Yellow_Cards",
    "HR": "Home_Red_Cards",
    "AR": "Away_Red_Cards",
}


def download_current_season() -> pd.DataFrame:
    """
    Download the current Premier League season CSV
    from Football-Data.co.uk.
    """

    response = requests.get(
        FOOTBALL_DATA_CO_UK_URL,
        timeout=30,
    )

    response.raise_for_status()

    raw = pd.read_csv(
        StringIO(response.text)
    )

    CURRENT_SEASON_RAW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the untouched source data for reference.
    raw.to_csv(
        CURRENT_SEASON_RAW_FILE,
        index=False,
    )

    return raw


def normalize_current_season(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert Football-Data.co.uk data into the
    format used internally by this project.
    """

    # -----------------------------------------------------
    # 1. KEEP COMPLETED MATCHES ONLY
    # -----------------------------------------------------

    if "FTR" not in raw.columns:
        raise ValueError(
            "Downloaded current-season data has no "
            "FTR result column."
        )

    completed = raw[
        raw["FTR"].isin(["H", "D", "A"])
    ].copy()

    # -----------------------------------------------------
    # 2. RENAME MATCH / STAT COLUMNS
    # -----------------------------------------------------

    rename_map = {
        source: target
        for source, target
        in RAW_TO_PROJECT_COLUMNS.items()
        if source in completed.columns
    }

    completed = completed.rename(
        columns=rename_map
    )

    # -----------------------------------------------------
    # 3. NORMALIZE TEAM NAMES
    # -----------------------------------------------------
    #
    # This prevents things like:
    #
    # Hull       vs Hull City
    # Coventry   vs Coventry City
    # Ipswich    vs Ipswich Town
    #
    # from being treated as different clubs.

    completed["Home_Team"] = (
        completed["Home_Team"]
        .apply(internal_team_name)
    )

    completed["Away_Team"] = (
        completed["Away_Team"]
        .apply(internal_team_name)
    )

    # -----------------------------------------------------
    # 4. NORMALIZE DATE
    # -----------------------------------------------------

    completed["Date"] = pd.to_datetime(
        completed["Date"],
        dayfirst=True,
        errors="coerce",
    )

    completed = completed.dropna(
        subset=["Date"]
    ).copy()

    completed["Date"] = (
        completed["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    # -----------------------------------------------------
    # 5. ADD SEASON
    # -----------------------------------------------------

    completed["Season"] = CURRENT_SEASON

    return completed


def rebuild_master(
    current: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine historical EPL data with the current
    season's completed matches.
    """

    base = pd.read_csv(
        BASE_HISTORICAL_FILE
    )

    # -----------------------------------------------------
    # NORMALIZE HISTORICAL TEAM NAMES TOO
    # -----------------------------------------------------
    #
    # This is important because the older master data may
    # contain team names such as "Ipswich" while the current
    # project uses "Ipswich Town".
    #
    # Normalizing both sources gives every club one identity.

    if "Home_Team" in base.columns:
        base["Home_Team"] = (
            base["Home_Team"]
            .apply(internal_team_name)
        )

    if "Away_Team" in base.columns:
        base["Away_Team"] = (
            base["Away_Team"]
            .apply(internal_team_name)
        )

    # -----------------------------------------------------
    # COMBINE HISTORY
    # -----------------------------------------------------

    master = pd.concat(
        [base, current],
        ignore_index=True,
        sort=False,
    )

    # -----------------------------------------------------
    # NORMALIZE DATE
    # -----------------------------------------------------

    master["Date"] = pd.to_datetime(
        master["Date"],
        errors="coerce",
    )

    master = master.dropna(
        subset=["Date"]
    ).copy()

    sort_columns = ["Date"]

    if "Time" in master.columns:
        sort_columns.append("Time")

    master = (
        master
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    master["Date"] = (
        master["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    # -----------------------------------------------------
    # REMOVE DUPLICATE FIXTURES
    # -----------------------------------------------------

    master = (
        master
        .drop_duplicates(
            subset=[
                "Season",
                "Date",
                "Home_Team",
                "Away_Team",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # SAVE MASTER HISTORY
    # -----------------------------------------------------

    MASTER_MATCH_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    master.to_csv(
        MASTER_MATCH_FILE,
        index=False,
    )

    return master


def main() -> None:
    print(
        "Downloading latest Premier League "
        "results/statistics..."
    )

    raw = download_current_season()

    current = normalize_current_season(
        raw
    )

    master = rebuild_master(
        current
    )

    print()
    print(
        f"Current-season completed matches: "
        f"{len(current)}"
    )

    print(
        f"Master completed-match rows: "
        f"{len(master)}"
    )

    print(
        f"Saved current season to: "
        f"{CURRENT_SEASON_RAW_FILE}"
    )

    print(
        f"Saved master history to: "
        f"{MASTER_MATCH_FILE}"
    )

    print()


if __name__ == "__main__":
    main()