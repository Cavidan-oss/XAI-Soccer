import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import streamlit as st
from datetime import datetime
from src.function import load_simulated_matches


st.set_page_config(
    page_title="XAI-Soccer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state = 'collapsed'
)


simulated_matches = load_simulated_matches(
    csv_path="match_summary.csv",
)


live_matches = [

]


def render_match_row(match: dict, kind: str, key_prefix: str):
    """
    kind = 'sim' or 'live'
    key_prefix is only used to keep unique Streamlit keys.
    Navigates to pages/Match_View.py and passes match_id in query params.
    """

    match_id = match["match_id"]

    with st.container():
        st.markdown("---")

        # Row 1: competition label
        c1, c2 = st.columns([1, 6])
        with c1:
            st.markdown(f"**{match['competition_short']}**")
        with c2:
            st.caption(match["competition"])

        # Row 2: team names + score
        c1, c2, c3 = st.columns([4, 2, 4])
        score_str = f"{match['home_score']} - {match['away_score']}"

        with c1:
            st.markdown(
                f"<div style='font-size:1.1rem;font-weight:700;text-align:left;'>{match['home_team']}</div>",
                unsafe_allow_html=True,
            )

        with c2:
            if kind == "live":
                st.markdown(
                    f"""
                    <div style='text-align:center;'>
                        <div style='font-size:1.6rem;font-weight:800;'>{score_str}</div>
                        <div style='font-size:0.8rem;color:#ef4444;font-weight:600;'>{match['minute']}' Live</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='font-size:1.6rem;font-weight:800;text-align:center;'>{score_str}</div>",
                    unsafe_allow_html=True,
                )

        with c3:
            st.markdown(
                f"<div style='font-size:1.1rem;font-weight:700;text-align:right;'>{match['away_team']}</div>",
                unsafe_allow_html=True,
            )

        # Row 3: navigation button to Match_View page
        btn_key = f"{key_prefix}_{match_id}"

        if st.button("Open match view", key=btn_key, use_container_width=True):
            st.session_state["match_id"] = match_id

            # set URL param using ONLY st.query_params
            st.query_params["match_id"] = str(match_id)

            # go to match page
            st.switch_page("pages/Match_View.py")

        st.write("")  # small gap between games


# ------------- Header -------------
left, right = st.columns([3, 1])

with left:
    st.markdown(
        """
        <h1 style="margin-bottom:0;">XAI-Soccer</h1>
        <p style="color:#6B7280;margin-top:0.2rem;">
            Understanding the Game Behind the Odds
        </p>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        f"""
        <div style="text-align:right;color:#6B7280;font-size:0.8rem;margin-top:0.6rem;">
            {datetime.now().strftime('%d %b %Y • %H:%M')}
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# ------------- Simulated games (default open) -------------
with st.expander("Simulated games", expanded=True):
    if not simulated_matches:
        st.info("No simulated games available.")
    else:
        for m in simulated_matches:
            render_match_row(m, kind="sim", key_prefix="sim")


# ------------- Live games (default closed) -------------
with st.expander("Live games", expanded=False):
    if not live_matches:
        st.info("No live games at the moment.")
    else:
        for m in live_matches:
            render_match_row(m, kind="live", key_prefix="live")