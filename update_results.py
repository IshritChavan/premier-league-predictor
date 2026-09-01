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
    response = requests.get(FOOTBALL_DATA_CO_UK_URL, timeout=30)
    response.raise_for_status()

    raw = pd.read_csv(StringIO(response.text))
    CURRENT_SEASON_RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CURRENT_SEASON_RAW_FILE, index=False)
    return raw


def normalize_current_season(raw: pd.DataFrame) -> pd.DataFrame:
    # Keep completed matches only. Football-Data.co.uk may publish fixture/odds rows
    # before a score exists, and those must not affect Elo or form.
    if "FTR" not in raw.columns:
        raise ValueError("Downloaded current-season data has no FTR result column.")

    completed = raw[raw["FTR"].isin(["H", "D", "A"])].copy()

    # Rename the core match/stat fields. Betting columns keep their source names.
    rename_map = {
        source: target
        for source, target in RAW_TO_PROJECT_COLUMNS.items()
        if source in completed.columns
    }
    completed = completed.rename(columns=rename_map)

    # The master dataset uses ISO-style dates.
    completed["Date"] = pd.to_datetime(
        completed["Date"], dayfirst=True, errors="coerce"
    )
    completed = completed.dropna(subset=["Date"]).copy()
    completed["Date"] = completed["Date"].dt.strftime("%Y-%m-%d")
    completed["Season"] = CURRENT_SEASON

    return completed


def rebuild_master(current: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_csv(BASE_HISTORICAL_FILE)

    # If this script is re-run, the base file still contains only through 2025-26,
    # so concatenating the latest current-season snapshot cannot create duplicates.
    master = pd.concat([base, current], ignore_index=True, sort=False)

    master["Date"] = pd.to_datetime(master["Date"], errors="coerce")
    master = master.dropna(subset=["Date"]).copy()
    master = master.sort_values(["Date", "Time"] if "Time" in master.columns else ["Date"])
    master["Date"] = master["Date"].dt.strftime("%Y-%m-%d")

    # Defensive dedupe on the identity of a league fixture.
    master = master.drop_duplicates(
        subset=["Season", "Date", "Home_Team", "Away_Team"],
        keep="last",
    ).reset_index(drop=True)

    MASTER_MATCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(MASTER_MATCH_FILE, index=False)
    return master


def main() -> None:
    print("Downloading latest Premier League results/statistics...")
    raw = download_current_season()
    current = normalize_current_season(raw)
    master = rebuild_master(current)

    print(f"Current-season completed matches: {len(current)}")
    print(f"Master completed-match rows: {len(master)}")
    print(f"Saved current season to: {CURRENT_SEASON_RAW_FILE}")
    print(f"Saved master history to: {MASTER_MATCH_FILE}")


if __name__ == "__main__":
    main()
