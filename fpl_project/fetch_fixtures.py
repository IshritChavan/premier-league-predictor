"""Fetch the next unplayed Premier League matchweek from football-data.org."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from project_config import (
    CURRENT_SEASON,
    CURRENT_SEASON_START_YEAR,
    FOOTBALL_DATA_API_BASE,
    FOOTBALL_DATA_COMPETITION,
    UPCOMING_FIXTURES_FILE,
    internal_team_name,
)


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FOOTBALL_DATA_API_KEY is missing. Create a .env file from "
            ".env.example and paste your football-data.org key there."
        )
    return key


def fetch_matches() -> list[dict]:
    url = (
        f"{FOOTBALL_DATA_API_BASE}/competitions/"
        f"{FOOTBALL_DATA_COMPETITION}/matches"
    )
    response = requests.get(
        url,
        headers={"X-Auth-Token": get_api_key()},
        params={"season": CURRENT_SEASON_START_YEAR},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("matches", [])


def choose_next_matchweek(matches: list[dict]) -> list[dict]:
    now = datetime.now(timezone.utc)
    candidates: list[tuple[dict, datetime]] = []

    for match in matches:
        status = str(match.get("status", "")).upper()
        if status in {"FINISHED", "CANCELLED"}:
            continue

        utc_raw = match.get("utcDate")
        if not utc_raw:
            continue

        kickoff = datetime.fromisoformat(utc_raw.replace("Z", "+00:00"))

        # Keep genuine future fixtures. A tiny tolerance is unnecessary here because
        # this script is for prediction before kickoff, not live match tracking.
        if kickoff <= now:
            continue

        candidates.append((match, kickoff))

    if not candidates:
        raise RuntimeError("No future Premier League fixtures were returned by the API.")

    # Prefer matchday because that gives the whole next gameweek instead of one day.
    matchdays = [
        int(match["matchday"])
        for match, _ in candidates
        if match.get("matchday") is not None
    ]

    if matchdays:
        next_matchday = min(matchdays)
        selected = [
            match
            for match, _ in candidates
            if match.get("matchday") is not None
            and int(match["matchday"]) == next_matchday
        ]
    else:
        # Fallback: if the provider ever omits matchday, take the earliest 7-day block.
        earliest = min(kickoff for _, kickoff in candidates)
        cutoff = earliest.timestamp() + 7 * 24 * 60 * 60
        selected = [
            match
            for match, kickoff in candidates
            if kickoff.timestamp() <= cutoff
        ]

    return selected


def to_dataframe(matches: list[dict]) -> pd.DataFrame:
    rows = []
    for match in matches:
        kickoff_utc = pd.to_datetime(match["utcDate"], utc=True)
        # Your manually-entered times were Los Angeles time, so preserve that UX.
        kickoff_la = kickoff_utc.tz_convert("America/Los_Angeles")

        rows.append(
            {
                "Date": kickoff_la.strftime("%Y-%m-%d"),
                "Time": kickoff_la.strftime("%H:%M"),
                "Home_Team": internal_team_name(match["homeTeam"]["name"]),
                "Away_Team": internal_team_name(match["awayTeam"]["name"]),
                "Season": CURRENT_SEASON,
                "Matchweek": match.get("matchday", ""),
                "API_Match_ID": match.get("id", ""),
                "Status": match.get("status", ""),
            }
        )

    fixtures = pd.DataFrame(rows)
    fixtures = fixtures.sort_values(["Date", "Time"]).reset_index(drop=True)
    return fixtures


def main() -> None:
    print("Fetching upcoming Premier League fixtures...")
    all_matches = fetch_matches()
    next_week = choose_next_matchweek(all_matches)
    fixtures = to_dataframe(next_week)

    UPCOMING_FIXTURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(UPCOMING_FIXTURES_FILE, index=False)

    matchweek = fixtures["Matchweek"].iloc[0] if not fixtures.empty else "?"
    print(f"Fetched {len(fixtures)} fixtures for matchweek {matchweek}.")
    print(f"Saved to: {UPCOMING_FIXTURES_FILE}")
    if not fixtures.empty:
        print(fixtures[["Date", "Time", "Home_Team", "Away_Team"]].to_string(index=False))


if __name__ == "__main__":
    main()
