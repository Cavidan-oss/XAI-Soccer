import os
import sys
import time

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Match View",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.function import (
    plot_shap_for_holdout_state,
    plot_lime_for_holdout_state,  
    get_match_probabilities,
    p as processed_df,
    model as trained_model,
    features as feature_cols,
)


st.markdown(
    """
    <style>
      .header-card, .prob-card {
          background: var(--secondary-background-color);
          color: var(--text-color);
          border: 1px solid rgba(128,128,128,0.35);
          border-radius: 12px;
      }
      .header-card {
          padding: 16px;
          text-align: center;
      }
      .prob-card {
          padding: 12px;
          text-align: center;
      }

      /* Ensure all Streamlit buttons are readable in any theme */
      .stButton > button {
          background: var(--primary-color) !important;
          color: white !important;
          border: none !important;
          border-radius: 8px !important;
          padding: 0.5rem 0.9rem !important;
          font-weight: 700 !important;
      }
      .stButton > button:hover {
          filter: brightness(0.95);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

def get_match_id(default=None):
    qp = st.query_params
    if "match_id" in qp:
        try:
            return int(qp["match_id"])
        except Exception:
            return default
    if "match_id" in st.session_state:
        return int(st.session_state["match_id"])
    return default

def autorefresh(interval_sec=5):
    if st.session_state.get("running", False):
        time.sleep(interval_sec)
        st.rerun()

# ------------------------------
# Main
# ------------------------------
match_id = get_match_id(default=None)
HOME_PAGE_PATH = "Home.py"
REFRESH_SEC = 1  # advance interval every 5s

# Back button + match id
top_cols = st.columns([1, 9])
with top_cols[0]:
    if st.button("← Back", use_container_width=True):
        st.switch_page(HOME_PAGE_PATH)


if match_id is None:
    st.error("No match_id provided. Open a match from Home page.")
    st.stop()

match_df = processed_df[processed_df["matchId"] == match_id].copy()
if match_df.empty:
    st.error(f"No processed data found for match_id={match_id}")
    st.stop()

# Available intervals for this match (sorted ascending)
intervals = sorted(match_df["time_interval"].unique().tolist())
if not intervals:
    st.error(f"No time intervals found for match_id={match_id}")
    st.stop()

# Identify home/away team ids once
home_team_id = match_df.query("is_home_team == 1").teamId.unique()[0]
away_team_id = match_df.query("is_home_team == 0").teamId.unique()[0]

home_intervals = set(
    match_df.loc[match_df["is_home_team"] == 1, "time_interval"].unique()
)
away_intervals = set(
    match_df.loc[match_df["is_home_team"] == 0, "time_interval"].unique()
)

valid_intervals = sorted(home_intervals & away_intervals)

# If for some reason intersection is empty, fall back to all intervals
if not valid_intervals:
    valid_intervals = intervals

MAX_MINUTE = valid_intervals[-1]   # cap at last valid minute

def latest_valid_minute(target_minute: int) -> int:
    """Return latest valid minute <= target_minute."""
    idx = np.searchsorted(valid_intervals, target_minute, side="right") - 1
    if idx < 0:
        return valid_intervals[0]
    return valid_intervals[idx]

# Initialize session minute
if "current_minute" not in st.session_state:
    st.session_state["current_minute"] = valid_intervals[0]

# If user just pressed Start, don't auto-advance on the same run
just_started = st.session_state.pop("just_started", False)

# Auto-advance minute when running (skip gaps)
if st.session_state.get("running", False) and not just_started:
    cur = st.session_state["current_minute"]

    cur = latest_valid_minute(cur)

    if cur >= MAX_MINUTE:
        st.session_state["running"] = False
    else:
        i = valid_intervals.index(cur)
        st.session_state["current_minute"] = valid_intervals[i + 1]

current_minute = latest_valid_minute(st.session_state["current_minute"])
st.session_state["current_minute"] = current_minute

home_row = match_df[
    (match_df["is_home_team"] == 1) &
    (match_df["time_interval"] == current_minute)
]
away_row = match_df[
    (match_df["is_home_team"] == 0) &
    (match_df["time_interval"] == current_minute)
]

# If still empty for some weird reason, step backward to previous valid minute
if home_row.empty or away_row.empty:
    prev_minute = latest_valid_minute(current_minute - 1)
    current_minute = prev_minute
    st.session_state["current_minute"] = current_minute

    home_row = match_df[
        (match_df["is_home_team"] == 1) &
        (match_df["time_interval"] == current_minute)
    ]
    away_row = match_df[
        (match_df["is_home_team"] == 0) &
        (match_df["time_interval"] == current_minute)
    ]

# If even previous minute doesn't exist, stop with a clear error
if home_row.empty or away_row.empty:
    st.error(
        f"No valid state found for match_id={match_id}. "
        f"Checked minute={current_minute} and previous minutes."
    )
    st.stop()

home_row = home_row.iloc[0]
away_row = away_row.iloc[0]

# probabilities
probs_dict = get_match_probabilities(
    match_id=match_id,
    minute=current_minute,
    p=processed_df,
    model=trained_model,
    features=feature_cols,
    n_sims=5000,
)

home_team = probs_dict["home_team_name"]
away_team = probs_dict["away_team_name"]
home_team_id = probs_dict["home_team_id"]

home_score = int(home_row.get("goals_scored", 0))
away_score = int(away_row.get("goals_scored", 0))

probs = {
    "home_win": probs_dict["home_win_prob"],
    "draw": probs_dict["draw_prob"],
    "away_win": probs_dict["away_win_prob"],
}

# SHAP for home team at current minute
shap_info = plot_shap_for_holdout_state(
    match_id=match_id,
    team_id=home_team_id,
    time_interval=current_minute,
    p=processed_df,
    model=trained_model,
    features=feature_cols,
    top_n=10,
)
plt.close("all")

shap_top_feats = shap_info["features"]
shap_top_vals = shap_info["shap_values"]

# ----- LIME (same state) -----
lime_info = plot_lime_for_holdout_state(
    match_id=match_id,
    team_id=home_team_id,
    time_interval=current_minute,
    p=processed_df,
    model=trained_model,
    features=feature_cols,
    top_n=10,
)
plt.close("all")

lime_top_feats = lime_info["features"]
lime_top_vals  = lime_info["lime_weights"]

# -------- UI --------
st.markdown(
    f"""
    <div class="header-card">
        <div style="font-size:1.6rem;font-weight:700;">
            {home_team} vs {away_team}
        </div>
        <div style="font-size:2.3rem;font-weight:900;margin-top:6px;">
            {home_score} - {away_score}
        </div>
        <div style="font-size:0.95rem;opacity:0.8;margin-top:4px;">
            Minute: {current_minute}'
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

btn_cols = st.columns([1, 1, 1, 7])

with btn_cols[0]:
    if st.button("Start", use_container_width=True):
        st.session_state["running"] = True
        st.session_state["just_started"] = True

with btn_cols[1]:
    if st.button("Stop", use_container_width=True):
        st.session_state["running"] = False

with btn_cols[2]:
    if st.button("Restart", use_container_width=True):
        # reset to earliest valid interval and pause
        st.session_state["running"] = False
        st.session_state["current_minute"] = valid_intervals[0]
        # optional: clear any startup flag
        st.session_state.pop("just_started", None)
        st.rerun()
st.write("")

st.markdown("## Win Probabilities")
pcols = st.columns(3)
labels = ["Home Win", "Draw", "Away Win"]
values = [probs["home_win"], probs["draw"], probs["away_win"]]

for c, lab, val in zip(pcols, labels, values):
    with c:
        st.markdown(
            f"""
            <div class="prob-card">
                <div style="font-size:0.95rem;font-weight:700;opacity:0.8;">{lab}</div>
                <div style="font-size:1.6rem;font-weight:900;margin-top:2px;">
                    {val:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

left_col, right_col = st.columns([4, 1])

# 4) Bottom area: SHAP (left) + LIME (right)
# 4) Bottom area: SHAP (left) + LIME (right)
plot_cols = st.columns(2)

def short_feat_name(s: str) -> str:
    """Make feature names compact and readable."""
    mapping = {
        "goals_scored": "goals",
        "player_differential": "player_diff",
        "own_yellow_cards": "own_yc",
        "opposition_yellow_cards": "opp_yc",
        "is_home_team": "home",
        "avg_team_xt": "avg_xt",
        "avg_opp_xt": "avg_opp_xt",
        "running_xt_differential": "run_xt_diff",
        "score_differential": "score_diff",
    }
    return mapping.get(s, s.replace("_", " "))

# ---------- prep SHAP ----------
shap_feats = [short_feat_name(f) for f in shap_top_feats]
shap_vals  = np.array(shap_top_vals, dtype=float)
shap_order = np.argsort(np.abs(shap_vals))[::-1]
shap_feats = [shap_feats[i] for i in shap_order]
shap_vals  = shap_vals[shap_order]

# ---------- prep LIME ----------
# LIME features come with bins like "avg_team_xt <= 2.24"
# We shorten only the base name before first space or <=, >=, etc.
def short_lime_name(s: str) -> str:
    base = s.split(" ")[0]  # take feature token
    return short_feat_name(base)

lime_feats = [short_lime_name(f) for f in lime_top_feats]
lime_vals  = np.array(lime_top_vals, dtype=float)
lime_order = np.argsort(np.abs(lime_vals))[::-1]
lime_feats = [lime_feats[i] for i in lime_order]
lime_vals  = lime_vals[lime_order]

# ---------- FIXED, ALIGNED HEIGHT ----------
# Use a shared symmetric y-limit so both charts match height every time.
shared_max_abs = float(
    max(
        np.max(np.abs(shap_vals)) if len(shap_vals) else 0,
        np.max(np.abs(lime_vals)) if len(lime_vals) else 0,
        1e-6
    )
)
shared_ylim = 1.15 * shared_max_abs  # stable padding

# ---------- SHAP plot ----------
with plot_cols[0]:
    st.markdown("## Key Feature Drivers (SHAP)")
    fig, ax = plt.subplots(figsize=(5, 2.6))  # same size always
    colors = ["tab:red" if v < 0 else "tab:blue" for v in shap_vals]

    ax.bar(range(len(shap_feats)), shap_vals, color=colors)
    ax.set_xticks(range(len(shap_feats)))
    ax.set_xticklabels(shap_feats, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylim(-shared_ylim, shared_ylim)
    ax.set_ylabel("SHAP", fontsize=9)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ---------- LIME plot ----------
with plot_cols[1]:
    st.markdown("## Key Feature Drivers (LIME)")
    fig, ax = plt.subplots(figsize=(5, 2.6))  # SAME figsize => same height
    colors = ["tab:red" if v < 0 else "tab:blue" for v in lime_vals]

    ax.bar(range(len(lime_feats)), lime_vals, color=colors)
    ax.set_xticks(range(len(lime_feats)))
    ax.set_xticklabels(lime_feats, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, linewidth=0.8)
    ax.set_ylim(-shared_ylim, shared_ylim)
    ax.set_ylabel("LIME", fontsize=9)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.write("")

# 5) Open Simulation button BELOW both plots
btn_row = st.columns([3, 2, 3])
with btn_row[1]:
    if st.button("Open Simulation", use_container_width=True):
        # pass context via session_state
        st.session_state["match_id"] = match_id
        st.session_state["time_interval"] = current_minute
        # navigate to Simulation page (must live in dashboard/pages/Simulation.py)
        st.switch_page("pages/Simulation.py")

autorefresh(interval_sec=REFRESH_SEC)