from pathlib import Path
import json

import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DATA = Path("data")
MODELS = Path("models")

CURRENT_SEASON = "2026-27"


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Premier League Predictor")

st.caption(
    "Automated fixtures, Elo/form updates, predictions and ranked markets"
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_csv(name: str) -> pd.DataFrame:
    path = DATA / name

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def result_label(result: str) -> str:
    mapping = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win",
    }

    return mapping.get(result, result)


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "Date" in df.columns:
        df = df.copy()

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce",
        ).dt.date

    return df


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

predictions = load_csv("weekly_predictions.csv")

markets = load_csv("ranked_all_markets.csv")

outrights = load_csv(
    "ranked_outright_predictions.csv"
)

parlays = load_csv("weekly_parlays.csv")

elos = load_csv("current_elo_ratings.csv")

current_results = load_csv("premier_league_all.csv")

prediction_history = load_csv(
    "prediction_history.csv"
)


# ---------------------------------------------------------
# UPCOMING MATCH CARDS
# ---------------------------------------------------------

if predictions.empty:

    st.warning(
        "No predictions found yet. "
        "Run `python run_pipeline.py` first."
    )

else:

    matchweek = predictions.get(
        "Matchweek",
        pd.Series(dtype=str),
    )

    if not matchweek.empty:

        st.subheader(
            f"Next Matchweek: {matchweek.iloc[0]}"
        )

    else:

        st.subheader("Upcoming Predictions")

    cards = st.columns(3)

    sorted_predictions = predictions.sort_values(
        ["Date", "Time"]
    )

    for i, (_, row) in enumerate(
        sorted_predictions.iterrows()
    ):

        with cards[i % 3]:

            st.markdown(
                f"**{row['Home_Team']} vs "
                f"{row['Away_Team']}**"
            )

            st.caption(
                f"{row['Date']}  "
                f"{row.get('Time', '')}"
            )

            st.write(
                f"Home "
                f"**{row['Home_Win_Probability']:.1%}** · "
                f"Draw "
                f"**{row['Draw_Probability']:.1%}** · "
                f"Away "
                f"**{row['Away_Win_Probability']:.1%}**"
            )

            st.write(
                f"Top market: "
                f"**{row['Highest_Probability_Market']}** "
                f"({row['Highest_Probability']:.1%})"
            )

            st.divider()


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Predictions",
        "Best Markets",
        "Parlays",
        "Elo Rankings",
        "Model",
    ]
)


# ---------------------------------------------------------
# TAB 1 — PREDICTIONS
# ---------------------------------------------------------

with tab1:

    if predictions.empty:

        st.info(
            "Predictions will appear after "
            "the pipeline runs."
        )

    else:

        columns = [
            "Date",
            "Time",
            "Home_Team",
            "Away_Team",
            "Home_Win_Probability",
            "Draw_Probability",
            "Away_Win_Probability",
            "Best_Single_Result",
            "Best_Single_Probability",
            "Highest_Probability_Market",
            "Highest_Probability",
            "Confidence",
        ]

        visible_columns = [
            column
            for column in columns
            if column in predictions.columns
        ]

        st.dataframe(
            predictions[visible_columns],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# TAB 2 — MARKETS
# ---------------------------------------------------------

with tab2:

    left, right = st.columns(2)

    with left:

        st.markdown("#### Top all markets")

        if markets.empty:

            st.info(
                "No ranked markets available yet."
            )

        else:

            st.dataframe(
                markets.head(20),
                use_container_width=True,
                hide_index=True,
            )

    with right:

        st.markdown(
            "#### Top outright predictions"
        )

        if outrights.empty:

            st.info(
                "No outright predictions "
                "available yet."
            )

        else:

            st.dataframe(
                outrights.head(20),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------
# TAB 3 — PARLAYS
# ---------------------------------------------------------

with tab3:

    if parlays.empty:

        st.info(
            "No parlay output found."
        )

    else:

        st.dataframe(
            parlays,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# TAB 4 — ELO
# ---------------------------------------------------------

with tab4:

    if elos.empty:

        st.info(
            "Elo rankings will be generated "
            "on the next prediction run."
        )

    else:

        st.dataframe(
            elos,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# TAB 5 — MODEL
# ---------------------------------------------------------

with tab5:

    # -----------------------------------------------------
    # MODEL DESCRIPTION
    # -----------------------------------------------------

    st.markdown("### Model")

    st.write(
        "Multinomial logistic regression using "
        "pre-match Elo and rolling form features."
    )

    st.divider()


    # -----------------------------------------------------
    # HISTORICAL BACKTEST
    # -----------------------------------------------------

    st.markdown(
        "### Historical Backtest — 2025–26"
    )

    st.caption(
        "Fixed evaluation on the historical "
        "2025–26 test season."
    )

    metrics_path = (
        MODELS / "model_metrics.json"
    )

    if metrics_path.exists():

        metrics = json.loads(
            metrics_path.read_text()
        )

        test = metrics.get(
            "test",
            {},
        )

        matches_tested = test.get(
            "matches",
            0,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Test accuracy",
            f"{test.get('accuracy', 0):.1%}",
        )

        c2.metric(
            "Balanced accuracy",
            f"{test.get('balanced_accuracy', 0):.1%}",
        )

        c3.metric(
            "Log loss",
            f"{test.get('log_loss', 0):.3f}",
        )

        c4.metric(
            "Matches tested",
            f"{matches_tested}",
        )

    else:

        st.info(
            "Historical model metrics not found."
        )


    st.divider()


    # -----------------------------------------------------
    # CURRENT SEASON
    # -----------------------------------------------------

    st.markdown(
        f"### Current Season — {CURRENT_SEASON}"
    )

    st.caption(
        "Live results and prediction performance. "
        "This section updates as completed matches "
        "are added by the pipeline."
    )


    # -----------------------------------------------------
    # PREPARE CURRENT RESULTS
    # -----------------------------------------------------

    current_results = normalize_date_column(
        current_results
    )

    prediction_history = normalize_date_column(
        prediction_history
    )

    completed_results = current_results.copy()

    if not completed_results.empty:

        if "Full_Time_Result" in completed_results.columns:

            completed_results = completed_results[
                completed_results[
                    "Full_Time_Result"
                ].isin(
                    ["H", "D", "A"]
                )
            ].copy()

        if "Season" in completed_results.columns:

            completed_results = completed_results[
                completed_results["Season"]
                == CURRENT_SEASON
            ].copy()


    # -----------------------------------------------------
    # CURRENT SEASON BASIC RESULT COUNT
    # -----------------------------------------------------

    completed_count = len(
        completed_results
    )

    if prediction_history.empty:

        c1, c2 = st.columns(2)

        c1.metric(
            "Completed matches",
            completed_count,
        )

        c2.metric(
            "Predictions evaluated",
            0,
        )

        st.info(
            "Current results are available, but "
            "`prediction_history.csv` does not exist yet. "
            "Live accuracy will begin once predictions "
            "are archived before matches are played."
        )

    else:

        current_predictions = (
            prediction_history.copy()
        )

        if "Season" in current_predictions.columns:

            current_predictions = (
                current_predictions[
                    current_predictions["Season"]
                    == CURRENT_SEASON
                ].copy()
            )


        # -------------------------------------------------
        # MATCH PREDICTIONS TO ACTUAL RESULTS
        # -------------------------------------------------

        merge_columns = [
            "Home_Team",
            "Away_Team",
        ]

        if (
            "Date" in current_predictions.columns
            and
            "Date" in completed_results.columns
        ):

            merge_columns.insert(
                0,
                "Date",
            )


        if (
            not current_predictions.empty
            and
            not completed_results.empty
            and
            "Predicted_Result"
            in current_predictions.columns
            and
            "Full_Time_Result"
            in completed_results.columns
        ):

            actual_columns = (
                merge_columns
                + [
                    "Full_Time_Result",
                ]
            )

            # Add score columns when available

            for score_column in [
                "Full_Time_Home_Goals",
                "Full_Time_Away_Goals",
            ]:

                if (
                    score_column
                    in completed_results.columns
                ):

                    actual_columns.append(
                        score_column
                    )


            evaluated = current_predictions.merge(
                completed_results[
                    actual_columns
                ],
                on=merge_columns,
                how="inner",
            )


            # ---------------------------------------------
            # CALCULATE LIVE PERFORMANCE
            # ---------------------------------------------

            if not evaluated.empty:

                evaluated[
                    "Prediction_Correct"
                ] = (
                    evaluated[
                        "Predicted_Result"
                    ]
                    ==
                    evaluated[
                        "Full_Time_Result"
                    ]
                )

                evaluated_count = len(
                    evaluated
                )

                correct_count = int(
                    evaluated[
                        "Prediction_Correct"
                    ].sum()
                )

                live_accuracy = (
                    correct_count
                    / evaluated_count
                )


                # -----------------------------------------
                # MAIN LIVE METRICS
                # -----------------------------------------

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Completed matches",
                    completed_count,
                )

                c2.metric(
                    "Predictions evaluated",
                    evaluated_count,
                )

                c3.metric(
                    "Correct predictions",
                    correct_count,
                )

                c4.metric(
                    "Live accuracy",
                    f"{live_accuracy:.1%}",
                )


                # -----------------------------------------
                # RESULT TYPE PERFORMANCE
                # -----------------------------------------

                st.markdown(
                    "#### Performance by result"
                )

                result_columns = st.columns(3)

                result_types = [
                    ("H", "Home wins"),
                    ("D", "Draws"),
                    ("A", "Away wins"),
                ]

                for column, (
                    result_code,
                    label,
                ) in zip(
                    result_columns,
                    result_types,
                ):

                    actual_type = evaluated[
                        evaluated[
                            "Full_Time_Result"
                        ]
                        == result_code
                    ]

                    total_type = len(
                        actual_type
                    )

                    if total_type > 0:

                        correct_type = int(
                            actual_type[
                                "Prediction_Correct"
                            ].sum()
                        )

                        accuracy_type = (
                            correct_type
                            / total_type
                        )

                        column.metric(
                            label,
                            f"{accuracy_type:.1%}",
                            help=(
                                f"{correct_type} correct "
                                f"out of {total_type}"
                            ),
                        )

                    else:

                        column.metric(
                            label,
                            "—",
                        )


                # -----------------------------------------
                # MATCH-BY-MATCH TABLE
                # -----------------------------------------

                st.markdown(
                    "#### Current-season prediction results"
                )

                evaluated[
                    "Prediction"
                ] = evaluated[
                    "Predicted_Result"
                ].apply(
                    result_label
                )

                evaluated[
                    "Actual"
                ] = evaluated[
                    "Full_Time_Result"
                ].apply(
                    result_label
                )

                evaluated[
                    "Correct"
                ] = evaluated[
                    "Prediction_Correct"
                ].map(
                    {
                        True: "✅",
                        False: "❌",
                    }
                )

                evaluated[
                    "Fixture"
                ] = (
                    evaluated[
                        "Home_Team"
                    ]
                    + " vs "
                    + evaluated[
                        "Away_Team"
                    ]
                )


                # Build score if score data exists

                if (
                    "Full_Time_Home_Goals"
                    in evaluated.columns
                    and
                    "Full_Time_Away_Goals"
                    in evaluated.columns
                ):

                    evaluated[
                        "Score"
                    ] = (
                        evaluated[
                            "Full_Time_Home_Goals"
                        ]
                        .astype(int)
                        .astype(str)
                        + "–"
                        + evaluated[
                            "Full_Time_Away_Goals"
                        ]
                        .astype(int)
                        .astype(str)
                    )


                table_columns = [
                    "Date",
                    "Fixture",
                ]

                if "Score" in evaluated.columns:

                    table_columns.append(
                        "Score"
                    )

                table_columns += [
                    "Prediction",
                    "Actual",
                    "Correct",
                ]


                if (
                    "Best_Single_Probability"
                    in evaluated.columns
                ):

                    table_columns.insert(
                        -2,
                        "Best_Single_Probability",
                    )


                display_results = (
                    evaluated[
                        table_columns
                    ]
                    .sort_values(
                        "Date",
                        ascending=False,
                    )
                    .reset_index(
                        drop=True
                    )
                )


                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Best_Single_Probability":
                            st.column_config
                            .PercentageColumn(
                                "Model confidence",
                                format="%.1f%%",
                            )
                    },
                )

            else:

                c1, c2 = st.columns(2)

                c1.metric(
                    "Completed matches",
                    completed_count,
                )

                c2.metric(
                    "Predictions evaluated",
                    0,
                )

                st.info(
                    "No archived predictions currently "
                    "match completed fixtures."
                )

        else:

            st.info(
                "Prediction history does not yet contain "
                "enough information to calculate "
                "live accuracy."
            )


    # -----------------------------------------------------
    # RECENT ACTUAL RESULTS
    # -----------------------------------------------------

    st.markdown("#### Recent Premier League results")

    if completed_results.empty:

        st.info(
            "No completed 2026–27 matches found."
        )

    else:

        recent_results = (
            completed_results
            .sort_values(
                "Date",
                ascending=False,
            )
            .copy()
        )

        recent_results[
            "Fixture"
        ] = (
            recent_results[
                "Home_Team"
            ]
            + " vs "
            + recent_results[
                "Away_Team"
            ]
        )

        recent_results[
            "Result"
        ] = recent_results[
            "Full_Time_Result"
        ].apply(
            result_label
        )


        if (
            "Full_Time_Home_Goals"
            in recent_results.columns
            and
            "Full_Time_Away_Goals"
            in recent_results.columns
        ):

            recent_results[
                "Score"
            ] = (
                recent_results[
                    "Full_Time_Home_Goals"
                ]
                .astype(int)
                .astype(str)
                + "–"
                + recent_results[
                    "Full_Time_Away_Goals"
                ]
                .astype(int)
                .astype(str)
            )


        recent_columns = [
            "Date",
            "Fixture",
        ]

        if "Score" in recent_results.columns:

            recent_columns.append(
                "Score"
            )

        recent_columns.append(
            "Result"
        )


        st.dataframe(
            recent_results[
                recent_columns
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )