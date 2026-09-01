from pathlib import Path


# ---------------------------------------------------------
# FOLDERS
# ---------------------------------------------------------

DATA_FOLDER = Path("data")
MODEL_FOLDER = Path("models")


# ---------------------------------------------------------
# DATA FILES
# ---------------------------------------------------------

# Historical data used by the model before the current season.
BASE_HISTORICAL_FILE = (
    DATA_FOLDER / "premier_league_2021_2026.csv"
)

# Downloaded current-season Football-Data.co.uk file.
CURRENT_SEASON_RAW_FILE = (
    DATA_FOLDER / "s26_27.csv"
)

# Combined completed-match history used by the predictor.
MASTER_MATCH_FILE = (
    DATA_FOLDER / "premier_league_all.csv"
)

# Generated automatically from football-data.org.
UPCOMING_FIXTURES_FILE = (
    DATA_FOLDER / "upcoming_fixtures.csv"
)


# ---------------------------------------------------------
# CURRENT SEASON
# ---------------------------------------------------------

CURRENT_SEASON = "2026-27"
CURRENT_SEASON_START_YEAR = 2026


# ---------------------------------------------------------
# FOOTBALL DATA SOURCES
# ---------------------------------------------------------

FOOTBALL_DATA_COMPETITION = "PL"

# Rich match stats source:
# goals, shots, shots on target, etc.
FOOTBALL_DATA_CO_UK_URL = (
    "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
)

# football-data.org API
FOOTBALL_DATA_API_BASE = (
    "https://api.football-data.org/v4"
)


# ---------------------------------------------------------
# TEAM NAME NORMALIZATION
# ---------------------------------------------------------

# Different data sources use different names for the same club.
# Everything gets converted into one project-internal name.

TEAM_NAME_ALIASES = {

    # Arsenal
    "Arsenal FC": "Arsenal",
    "Arsenal": "Arsenal",

    # Aston Villa
    "Aston Villa FC": "Aston Villa",
    "Aston Villa": "Aston Villa",

    # Bournemouth
    "AFC Bournemouth": "Bournemouth",
    "Bournemouth": "Bournemouth",

    # Brentford
    "Brentford FC": "Brentford",
    "Brentford": "Brentford",

    # Brighton
    "Brighton & Hove Albion FC": "Brighton",
    "Brighton": "Brighton",

    # Burnley
    "Burnley FC": "Burnley",
    "Burnley": "Burnley",

    # Chelsea
    "Chelsea FC": "Chelsea",
    "Chelsea": "Chelsea",

    # Coventry
    "Coventry City FC": "Coventry City",
    "Coventry City": "Coventry City",
    "Coventry": "Coventry City",

    # Crystal Palace
    "Crystal Palace FC": "Crystal Palace",
    "Crystal Palace": "Crystal Palace",

    # Everton
    "Everton FC": "Everton",
    "Everton": "Everton",

    # Fulham
    "Fulham FC": "Fulham",
    "Fulham": "Fulham",

    # Hull
    "Hull City AFC": "Hull City",
    "Hull City": "Hull City",
    "Hull": "Hull City",

    # Ipswich
    "Ipswich Town FC": "Ipswich Town",
    "Ipswich Town": "Ipswich Town",
    "Ipswich": "Ipswich Town",

    # Leeds
    "Leeds United FC": "Leeds",
    "Leeds": "Leeds",

    # Leicester
    "Leicester City FC": "Leicester",
    "Leicester": "Leicester",

    # Liverpool
    "Liverpool FC": "Liverpool",
    "Liverpool": "Liverpool",

    # Luton
    "Luton Town FC": "Luton",
    "Luton": "Luton",

    # Manchester City
    "Manchester City FC": "Man City",
    "Manchester City": "Man City",
    "Man City": "Man City",

    # Manchester United
    "Manchester United FC": "Man United",
    "Manchester United": "Man United",
    "Man United": "Man United",

    # Newcastle
    "Newcastle United FC": "Newcastle",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",

    # Norwich
    "Norwich City FC": "Norwich",
    "Norwich": "Norwich",

    # Nottingham Forest
    "Nottingham Forest FC": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",

    # Sheffield United
    "Sheffield United FC": "Sheffield United",
    "Sheffield United": "Sheffield United",

    # Southampton
    "Southampton FC": "Southampton",
    "Southampton": "Southampton",

    # Sunderland
    "Sunderland AFC": "Sunderland",
    "Sunderland": "Sunderland",

    # Tottenham
    "Tottenham Hotspur FC": "Tottenham",
    "Tottenham Hotspur": "Tottenham",
    "Tottenham": "Tottenham",

    # Watford
    "Watford FC": "Watford",
    "Watford": "Watford",

    # West Ham
    "West Ham United FC": "West Ham",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",

    # Wolves
    "Wolverhampton Wanderers FC": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "Wolves": "Wolves",
}


def internal_team_name(external_name: str) -> str:
    """
    Convert team names from external data sources
    into the names used internally by this project.
    """

    if external_name is None:
        return external_name

    cleaned = str(external_name).strip()

    if cleaned in TEAM_NAME_ALIASES:
        return TEAM_NAME_ALIASES[cleaned]

    # Safe fallback for straightforward names ending in " FC".
    if cleaned.endswith(" FC"):
        cleaned = cleaned[:-3]

    return cleaned


# ---------------------------------------------------------
# CURRENT PREMIER LEAGUE TEAMS
# ---------------------------------------------------------

# Used for the Streamlit Elo table so historical/relegated
# teams do not appear in the current-season rankings.

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