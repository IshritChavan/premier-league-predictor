from pathlib import Path

import pandas as pd
import streamlit as st


DATA = Path("data")

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="wide")
st.title("⚽ Premier League Predictor")
st.caption("Automated fixtures, Elo/form updates, predictions and ranked markets")


def load_csv(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


predictions = load_csv("weekly_predictions.csv")
markets = load_csv("ranked_all_markets.csv")
outrights = load_csv("ranked_outright_predictions.csv")
parlays = load_csv("weekly_parlays.csv")
elos = load_csv("current_elo_ratings.csv")
current_results = load_csv("s26_27.csv")

if predictions.empty:
    st.warning("No predictions found yet. Run `python run_pipeline.py` first.")
else:
    matchweek = predictions.get("Matchweek", pd.Series(dtype=str))
    if not matchweek.empty:
        st.subheader(f"Next Matchweek: {matchweek.iloc[0]}")
    else:
        st.subheader("Upcoming Predictions")

    cards = st.columns(3)
    for i, (_, row) in enumerate(predictions.sort_values(["Date", "Time"]).iterrows()):
        with cards[i % 3]:
            st.markdown(f"**{row['Home_Team']} vs {row['Away_Team']}**")
            st.caption(f"{row['Date']}  {row.get('Time', '')}")
            st.write(
                f"Home **{row['Home_Win_Probability']:.1%}** · "
                f"Draw **{row['Draw_Probability']:.1%}** · "
                f"Away **{row['Away_Win_Probability']:.1%}**"
            )
            st.write(
                f"Top market: **{row['Highest_Probability_Market']}** "
                f"({row['Highest_Probability']:.1%})"
            )
            st.divider()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Predictions", "Best Markets", "Parlays", "Elo Rankings", "Model"]
)

with tab1:
    if predictions.empty:
        st.info("Predictions will appear after the pipeline runs.")
    else:
        columns = [
            "Date", "Time", "Home_Team", "Away_Team",
            "Home_Win_Probability", "Draw_Probability", "Away_Win_Probability",
            "Best_Single_Result", "Best_Single_Probability",
            "Highest_Probability_Market", "Highest_Probability", "Confidence",
        ]
        st.dataframe(predictions[[c for c in columns if c in predictions.columns]], use_container_width=True)

with tab2:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top all markets")
        if not markets.empty:
            st.dataframe(markets.head(20), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Top outright predictions")
        if not outrights.empty:
            st.dataframe(outrights.head(20), use_container_width=True, hide_index=True)

with tab3:
    if parlays.empty:
        st.info("No parlay output found.")
    else:
        st.dataframe(parlays, use_container_width=True, hide_index=True)

with tab4:
    if elos.empty:
        st.info("Elo rankings will be generated on the next prediction run.")
    else:
        st.dataframe(elos, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("#### Current model")
    st.write("Multinomial logistic regression using pre-match Elo and rolling form features.")
    metrics_path = Path("models/model_metrics.json")
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        test = metrics.get("test", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", f"{test.get('accuracy', 0):.1%}")
        c2.metric("Balanced accuracy", f"{test.get('balanced_accuracy', 0):.1%}")
        c3.metric("Log loss", f"{test.get('log_loss', 0):.3f}")
