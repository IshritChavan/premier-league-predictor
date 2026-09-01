from pathlib import Path

DATA_FOLDER = Path("data")
MODEL_FOLDER = Path("models")

# Historical data used by the model before the current season.
BASE_HISTORICAL_FILE = DATA_FOLDER / "premier_league_2021_2026.csv"

# Downloaded current-season Football-Data.co.uk file.
CURRENT_SEASON_RAW_FILE = DATA_FOLDER / "s26_27.csv"

# Combined completed-match history used by the predictor.
MASTER_MATCH_FILE = DATA_FOLDER / "premier_league_all.csv"

# Generated automatically from football-data.org.
UPCOMING_FIXTURES_FILE = DATA_FOLDER / "upcoming_fixtures.csv"

CURRENT_SEASON = "2026-27"
CURRENT_SEASON_START_YEAR = 2026
FOOTBALL_DATA_COMPETITION = "PL"

# Rich match stats source (goals, shots, shots on target, etc.)
FOOTBALL_DATA_CO_UK_URL = (
    "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
)

# football-data.org API
FOOTBALL_DATA_API_BASE = "https://api.football-data.org/v4"

# Project-internal team names. The API normally uses long official names.
TEAM_NAME_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry City",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull City",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Norwich City FC": "Norwich",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "Watford FC": "Watford",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def internal_team_name(api_name: str) -> str:
    """Convert football-data.org names to names used by this project."""
    if api_name in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[api_name]

    # Safe fallback for straightforward names such as "Chelsea FC".
    cleaned = api_name.strip()
    if cleaned.endswith(" FC"):
        cleaned = cleaned[:-3]
    return cleaned

# ---------------------------------------------------------
# CURRENT PREMIER LEAGUE TEAMS
# ---------------------------------------------------------

CURRENT_PL_TEAMS = {
    "Arsenal",
    "Aston Villa",
    "Bournemouth",
    "Brentford",
    "Brighton",
    "Chelsea",
    "Coventry City",
    "Crystal Palace",
    "Everton",
    "Fulham",
    "Hull City",
    "Ipswich Town",
    "Leeds",
    "Liverpool",
    "Man City",
    "Man United",
    "Newcastle",
    "Nott'm Forest",
    "Sunderland",
    "Tottenham",
}