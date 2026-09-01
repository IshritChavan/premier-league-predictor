# Premier League Predictor — automation setup

## First local setup

```bash
cd /path/to/your/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and replace `PASTE_YOUR_KEY_HERE` with your football-data.org API key.
Never commit `.env`.

## Refresh everything

```bash
python run_pipeline.py
```

This downloads the latest completed 2026-27 Premier League results/statistics from Football-Data.co.uk, rebuilds `data/premier_league_all.csv`, downloads the next unplayed Premier League matchweek from football-data.org, then runs the existing prediction logic.

## Run the web app locally

```bash
streamlit run app.py
```

## GitHub automation

After pushing the project to GitHub, create a repository secret named:

`FOOTBALL_DATA_API_KEY`

The included workflow `.github/workflows/refresh.yml` refreshes the project every six hours and commits changed data/output CSVs.

## Streamlit Community Cloud

Deploy `app.py` from the GitHub repository. The Streamlit app only reads the CSV outputs committed by the GitHub Action, so the API key does not need to be placed in the web app.
