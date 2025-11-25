import json
import os
import pickle
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")


features = [
    'goals_scored',
    'player_differential',
    'own_yellow_cards',
    'opposition_yellow_cards',
    'is_home_team',
    'avg_team_xt',
    'avg_opp_xt',
    'running_xt_differential',
    'score_differential'
]

target = 'scored_goal_after'

WYSCOUT_DATA_FOLDER = "wyscout_figshare_data"  

# processed holdout / match-state data
p = pd.read_csv("sample_games.csv")

with open(os.path.join('teams.json')) as f:
    teams_meta = json.load(f)

team_names_dict = {t['wyId']: t['name'] for t in teams_meta}

with open("models/lightgbm/football.pkl", "rb") as f:
    model = pickle.load(f)


def plot_shap_for_holdout_state(
    match_id,
    team_id,
    time_interval,
    p,
    model,
    features,
    top_n=10
):
    """
    Plot a vertical SHAP barplot for a single row identified by
    (matchId, teamId, time_interval) in the holdout set.
    """

    mask = (
        (p["matchId"] == match_id) &
        (p["teamId"] == team_id) &
        (p["time_interval"] == time_interval)
    )

    if not mask.any():
        raise ValueError(
            f"No row found for matchId={match_id}, teamId={team_id}, "
            f"time_interval={time_interval}"
        )

    row = p.loc[mask].iloc[0]
    x_row = row[features].to_frame().T

    explainer = shap.TreeExplainer(model)
    shap_values_row = explainer.shap_values(x_row)[0]

    abs_shap = np.abs(shap_values_row)
    order = np.argsort(abs_shap)[::-1][:top_n]

    shap_top = shap_values_row[order]
    feats_top = [features[i] for i in order]
    vals_top = x_row.values[0][order]

    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(feats_top))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in shap_top]

    plt.bar(x_pos, shap_top, color=colors)
    plt.xticks(x_pos, feats_top, rotation=90)
    plt.axhline(0, color="black", linewidth=0.8)

    max_abs = np.max(np.abs(shap_top))
    ylim = 1.1 * max_abs
    plt.ylim(-ylim, ylim)

    plt.title(
        f"SHAP values – Match {match_id}, Team {team_id}, Minute {time_interval}"
    )
    plt.ylabel("SHAP value (impact on prediction)")
    plt.tight_layout()
    plt.show()

    return {
        "features": feats_top,
        "shap_values": shap_top,
        "feature_values": vals_top,
        "row_index": row.name,
    }


def get_match_probabilities(
    match_id: int,
    minute: int,
    p,
    model,
    features,
    n_sims: int = 50000,
):
    """
    Compute home/draw/away probabilities at a given (match_id, minute).
    """

    df = p[p["matchId"] == match_id].copy()
    if df.empty:
        raise ValueError(f"No data found for match_id={match_id}")

    home_team_id = df.query("is_home_team == 1").teamId.unique()[0]
    away_team_id = df.query("is_home_team == 0").teamId.unique()[0]

    home_team_name = team_names_dict.get(home_team_id, str(home_team_id))
    away_team_name = team_names_dict.get(away_team_id, str(away_team_id))

    home_row = df[(df["is_home_team"] == 1) & (df["time_interval"] == minute)]
    away_row = df[(df["is_home_team"] == 0) & (df["time_interval"] == minute)]

    if home_row.empty or away_row.empty:
        raise ValueError(
            f"No data for match_id={match_id} at minute={minute}"
        )

    home_row = home_row.iloc[0]
    away_row = away_row.iloc[0]

    hsd = home_row["score_differential"]

    h_prob = model.predict(home_row[features].values.reshape(1, -1))[0]
    a_prob = model.predict(away_row[features].values.reshape(1, -1))[0]

    # guard for invalid probs
    h_prob = float(np.clip(h_prob, 0, 1))
    a_prob = float(np.clip(a_prob, 0, 1))

    remaining = 100 - minute

    home_goals_sim = np.random.binomial(remaining, h_prob, n_sims)
    away_goals_sim = np.random.binomial(remaining, a_prob, n_sims)

    final_sd = hsd + (home_goals_sim - away_goals_sim)

    home_wins = np.mean(final_sd > 0)
    away_wins = np.mean(final_sd < 0)
    draws = np.mean(final_sd == 0)

    return {
        "match_id": match_id,
        "minute": minute,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "home_win_prob": home_wins * 100,
        "draw_prob": draws * 100,
        "away_win_prob": away_wins * 100,
    }


def load_simulated_matches(
    csv_path: str = "match_summary.csv",
    team_id_to_name: dict = team_names_dict,
    competition: str = "Premier League",
    competition_short: str = "PL",
    minute_col_candidates: tuple = ("minute", "match_minute", "time_interval"),
):
    """
    Read match_table CSV and return list of dicts for Home page.
    """

    df = pd.read_csv(csv_path)

    if "match_id" not in df.columns and "matchId" in df.columns:
        df = df.rename(columns={"matchId": "match_id"})

    required = {"match_id", "home_score", "away_score"}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(
            f"{csv_path} missing required columns: {missing_req}. "
            f"Found columns: {list(df.columns)}"
        )

    if "home_team" not in df.columns or "away_team" not in df.columns:
        if team_id_to_name is None:
            if "home_team_id" in df.columns:
                df["home_team"] = df["home_team_id"].astype(str)
            if "away_team_id" in df.columns:
                df["away_team"] = df["away_team_id"].astype(str)
        else:
            if "home_team_id" in df.columns:
                df["home_team"] = df["home_team_id"].map(team_id_to_name)
            if "away_team_id" in df.columns:
                df["away_team"] = df["away_team_id"].map(team_id_to_name)

    if "home_team" in df.columns and df["home_team"].isna().any() and "home_team_id" in df.columns:
        df.loc[df["home_team"].isna(), "home_team"] = df.loc[df["home_team"].isna(), "home_team_id"].astype(str)
    if "away_team" in df.columns and df["away_team"].isna().any() and "away_team_id" in df.columns:
        df.loc[df["away_team"].isna(), "away_team"] = df.loc[df["away_team"].isna(), "away_team_id"].astype(str)

    minute_col = next((c for c in minute_col_candidates if c in df.columns), None)

    matches = []
    for _, row in df.iterrows():
        item = {
            "match_id": int(row["match_id"]),
            "competition": row.get("competition", competition),
            "competition_short": row.get("competition_short", competition_short),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_score": int(row["home_score"]),
            "away_score": int(row["away_score"]),
        }
        if minute_col is not None and pd.notna(row[minute_col]):
            item["minute"] = int(row[minute_col])
        matches.append(item)

    return matches

def plot_lime_for_holdout_state(
    match_id,
    team_id,
    time_interval,
    p,
    model,
    features,
    top_n=10,
    training_sample_size=5000,
    random_state=42
):
    """
    Plot a vertical LIME barplot for a single row identified by
    (matchId, teamId, time_interval) in the holdout set.

    Parameters
    ----------
    match_id : int
    team_id : int
    time_interval : int
    p : pd.DataFrame
        Full processed dataframe containing matchId, teamId, time_interval, features, etc.
    model : fitted model (LightGBM regressor in your case)
    features : list of str
        Feature column names used to train the model.
    top_n : int
        Number of top features (by |LIME weight|) to display.
    training_sample_size : int
        LIME uses a background distribution; we sample rows from p for speed.
    random_state : int
        Reproducibility for sampling and LIME’s perturbations.

    Returns
    -------
    dict with:
        features, lime_weights, feature_values, row_index
    """

    # 1) Find the row for this state
    mask = (
        (p["matchId"] == match_id) &
        (p["teamId"] == team_id) &
        (p["time_interval"] == time_interval)
    )

    if not mask.any():
        raise ValueError(
            f"No row found for matchId={match_id}, teamId={team_id}, "
            f"time_interval={time_interval}"
        )

    row = p.loc[mask].iloc[0]
    x_row = row[features].values.astype(float)  # 1D array for LIME

    # 2) Build background data for LIME (sample for speed)
    bg = p[features].dropna()
    if len(bg) > training_sample_size:
        bg = bg.sample(training_sample_size, random_state=random_state)

    explainer = LimeTabularExplainer(
        training_data=bg.values,
        feature_names=features,
        mode="regression",
        discretize_continuous=True,
        random_state=random_state
    )

    # 3) Explain this single instance
    exp = explainer.explain_instance(
        data_row=x_row,
        predict_fn=lambda X: model.predict(X),
        num_features=top_n
    )

    # LIME returns (feature_name_with_bin, weight)
    lime_list = exp.as_list()

    feats_top = [f for f, w in lime_list]
    lime_weights = np.array([w for f, w in lime_list], dtype=float)

    # 4) Plot vertical bar chart
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(feats_top))
    colors = ["tab:red" if v < 0 else "tab:blue" for v in lime_weights]

    plt.bar(x_pos, lime_weights, color=colors)
    plt.xticks(x_pos, feats_top, rotation=90)
    plt.axhline(0, color="black", linewidth=0.8)

    max_abs = float(np.max(np.abs(lime_weights))) if len(lime_weights) else 1.0
    ylim = 1.1 * max_abs
    plt.ylim(-ylim, ylim)

    plt.title(
        f"LIME weights – Match {match_id}, Team {team_id}, Minute {time_interval}"
    )
    plt.ylabel("LIME weight (local impact)")
    plt.tight_layout()
    plt.show()

    return {
        "features": feats_top,
        "lime_weights": lime_weights,
        "feature_values": row[features].values,
        "row_index": row.name,
        "lime_raw_explanation": lime_list
    }