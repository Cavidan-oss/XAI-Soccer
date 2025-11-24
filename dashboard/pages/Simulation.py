

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Simulation",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.function import (
    p as processed_df,
    model as trained_model,
    features as feature_cols,
    team_names_dict,
)

HOME_PAGE_PATH = "Home.py"
MATCH_VIEW_PATH = "pages/Match_View.py"  


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


def get_query_int(name: str, default=None):
    qp = st.query_params
    if name in qp:
        try:
            return int(qp[name])
        except Exception:
            return default
    if name in st.session_state:
        try:
            return int(st.session_state[name])
        except Exception:
            return default
    return default


def get_initial_feature_row(match_id=None, time_interval=None):
    """If match_id/time provided, prefill from processed_df; else None."""
    if match_id is None or time_interval is None:
        return None

    df = processed_df[processed_df["matchId"] == match_id]
    if df.empty:
        return None

    # default to home team row if exists
    row = df[(df["is_home_team"] == 1) & (df["time_interval"] == time_interval)]
    if row.empty:
        # fallback to latest <= time_interval
        home_rows = df[df["is_home_team"] == 1]
        if home_rows.empty:
            return None
        valid_times = sorted(home_rows["time_interval"].unique())
        idx = np.searchsorted(valid_times, time_interval, side="right") - 1
        idx = max(0, idx)
        row = home_rows[home_rows["time_interval"] == valid_times[idx]]

    if row.empty:
        return None
    return row.iloc[0]


# ------------------------------
# Pull optional context from URL/session
# ------------------------------
match_id = get_query_int("match_id", default=None)
time_interval = get_query_int("time_interval", default=0)

prefill_row = get_initial_feature_row(match_id, time_interval)

# Infer team names if match context exists
home_team_name = away_team_name = "Simulation"
home_score = away_score = 0
minute = time_interval

if match_id is not None:
    md = processed_df[processed_df["matchId"] == match_id]
    if not md.empty:
        home_team_id = md.query("is_home_team == 1").teamId.unique()[0]
        away_team_id = md.query("is_home_team == 0").teamId.unique()[0]
        home_team_name = team_names_dict.get(home_team_id, str(home_team_id))
        away_team_name = team_names_dict.get(away_team_id, str(away_team_id))

        if prefill_row is not None:
            # approximate score using goals_scored feature when available
            home_score = int(prefill_row.get("goals_scored", 0))
            away_row = md[(md["is_home_team"] == 0) & (md["time_interval"] == prefill_row["time_interval"])]
            if not away_row.empty:
                away_score = int(away_row.iloc[0].get("goals_scored", 0))

# ------------------------------
# Header area (same layout as MatchView)
# ------------------------------
# Top back buttons row
nav_cols = st.columns([1, 1, 8])
with nav_cols[0]:
    if st.button("← Home", use_container_width=True):
        st.switch_page(HOME_PAGE_PATH)
with nav_cols[1]:
    if match_id is not None and st.button("← Match", use_container_width=True):
        # optional: go back to match view
        st.switch_page(MATCH_VIEW_PATH)

st.markdown(
    f"""
    <div class="header-card">
        <div style="font-size:1.6rem;font-weight:700;">
            {home_team_name} vs {away_team_name}
        </div>
        <div style="font-size:2.3rem;font-weight:900;margin-top:6px;">
            {home_score} - {away_score}
        </div>
        <div style="font-size:0.95rem;opacity:0.8;margin-top:4px;">
            Simulation at minute: {minute}'
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ------------------------------
# Feature controls (replace SHAP/LIME area)
# We'll use dataset min/max as slider ranges when possible.
# ------------------------------
feat_stats = processed_df[feature_cols].agg(["min", "max", "median"]).to_dict()


def stat_or_default(feat, which, default):
    val = feat_stats.get(feat, {}).get(which, default)
    if pd.isna(val):
        return default
    return float(val)


def make_slider(feat, label, default_val):
    """Create sliders with fixed, football-sensible ranges."""
    fixed_ranges = {
        "goals_scored": (0, 15, 1),
        "player_differential": (-5, 5, 1),
        "own_yellow_cards": (0, 11, 1),
        "opposition_yellow_cards": (0, 11, 1),
        "score_differential": (-11, 11, 1),
        "avg_team_xt": (0.0, 3.5, 0.05),
        "avg_opp_xt": (0.0, 3.5, 0.05),
        "running_xt_differential": (0.0, 0.5, 0.01),
    }

    if feat in fixed_ranges:
        lo, hi, step = fixed_ranges[feat]
        # int sliders
        if step == 1 and isinstance(lo, int) and isinstance(hi, int):
            return st.slider(label, int(lo), int(hi), int(round(default_val)))
        return st.slider(label, float(lo), float(hi), float(default_val), step=float(step))

    fmin = stat_or_default(feat, "min", default_val - 1)
    fmax = stat_or_default(feat, "max", default_val + 1)
    pad = 0.1 * (fmax - fmin) if fmax > fmin else 1.0
    lo = fmin - pad
    hi = fmax + pad
    return st.slider(label, float(lo), float(hi), float(default_val), step=(hi - lo) / 200 if hi > lo else 0.01)


# Defaults from prefill row or sensible zeros
defaults = {f: 0 for f in feature_cols}
if prefill_row is not None:
    for f in feature_cols:
        defaults[f] = float(prefill_row.get(f, 0))

# Two-column layout for controls
control_cols = st.columns(2)

with control_cols[0]:
    st.markdown("## Simulation Inputs")
    goals_scored = make_slider("goals_scored", "Goals scored (this team)", defaults["goals_scored"])
    player_diff  = make_slider("player_differential", "Player differential", defaults["player_differential"])
    own_yc       = make_slider("own_yellow_cards", "Own yellow cards", defaults["own_yellow_cards"])
    opp_yc       = make_slider("opposition_yellow_cards", "Opposition yellow cards", defaults["opposition_yellow_cards"])

with control_cols[1]:
    st.markdown("## Possession / xT Inputs")
    avg_xt       = make_slider("avg_team_xt", "Avg team xT", defaults["avg_team_xt"])
    avg_opp_xt   = make_slider("avg_opp_xt", "Avg opposition xT", defaults["avg_opp_xt"])
    run_xt_diff  = make_slider("running_xt_differential", "Running xT differential", defaults["running_xt_differential"])
    score_diff   = make_slider("score_differential", "Score differential", defaults["score_differential"])

# Hidden is_home_team for prediction (model may still require it)
if "is_home_team" in feature_cols:
    is_home_team = int(round(defaults.get("is_home_team", 1)))
else:
    is_home_team = None

# Build single-row input for prediction
base_dict = {
    "goals_scored": goals_scored,
    "player_differential": player_diff,
    "own_yellow_cards": own_yc,
    "opposition_yellow_cards": opp_yc,
    "avg_team_xt": avg_xt,
    "avg_opp_xt": avg_opp_xt,
    "running_xt_differential": run_xt_diff,
    "score_differential": score_diff,
}
if is_home_team is not None:
    base_dict["is_home_team"] = is_home_team

input_row = pd.DataFrame([base_dict])

# Predict and clamp to [0,1]
pred = float(trained_model.predict(input_row[feature_cols])[0])
pred = float(np.clip(pred, 0, 1))

st.write("")


# ------------------------------
# Win probabilities (Home / Draw / Away)
# ------------------------------
st.markdown("## Win Probabilities")

# Treat sliders as HOME team state; keep away team state from prefill if available.
h_prob = float(np.clip(pred, 0, 1))

if prefill_row is not None and match_id is not None:
    md = processed_df[processed_df["matchId"] == match_id]
    away_prefill = md[(md["is_home_team"] == 0) & (md["time_interval"] == prefill_row["time_interval"])]
    if away_prefill.empty:
        away_prefill = md[md["is_home_team"] == 0].sort_values("time_interval").tail(1)
    if away_prefill.empty:
        a_prob = h_prob
    else:
        a_dict = away_prefill.iloc[0][feature_cols].to_dict()
        a_prob = float(np.clip(trained_model.predict(pd.DataFrame([a_dict])[feature_cols])[0], 0, 1))
else:
    a_prob = h_prob

# Use simulated score differential as home lead; away is negative of that.
hsd = int(score_diff)
asd = -hsd

n_sims = 5000
remaining = max(0, 100 - int(minute))

home_goals_sim = np.random.binomial(remaining, h_prob, n_sims)
away_goals_sim = np.random.binomial(remaining, a_prob, n_sims)
home_sd_sim = hsd + (home_goals_sim - away_goals_sim)

home_win = np.mean(home_sd_sim > 0) * 100
draw_win = np.mean(home_sd_sim == 0) * 100
away_win = np.mean(home_sd_sim < 0) * 100

pcols = st.columns(3)
labels = ["Home Win", "Draw", "Away Win"]
values = [home_win, draw_win, away_win]

for c, lab, val in zip(pcols, labels, values):
    with c:
        st.markdown(
            f"""
            <div class="prob-card">
                <div style="font-size:0.95rem;font-weight:700;opacity:0.8;">{lab}</div>
                <div style="font-size:1.6rem;font-weight:900;margin-top:2px;">{val:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")

with st.expander("Show input row"):
    st.dataframe(input_row, use_container_width=True)
