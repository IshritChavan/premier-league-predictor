import pandas as pd

matches = pd.read_csv("data/premier_league_2021_2026.csv")

teams = sorted(
    set(matches["Home_Team"].dropna())
    | set(matches["Away_Team"].dropna())
)

for team in teams:
    print(team)