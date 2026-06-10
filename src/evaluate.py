"""
src/evaluate.py
---------------
TODO: Shea 

Functions to implement:
    - elo_expected(elo_w, elo_b)
    - evaluate(df, players, skill_mean)
    - learning_curve(df, n_players, fractions)
    - prior_sensitivity(df, n_players, sigmas)
    - fig_posterior_skills(trace, players, out)
    - fig_skill_vs_elo(df, players, skill_mean, out)
    - fig_model_comparison(results, lc_df, out)
    - fig_trace_diagnostics(trace, out)
    - fig_prior_sensitivity(ps_df, out)
    - fig_calibration(results, out)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
from pathlib import Path
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR   = Path(__file__).parent.parent / "results"


def elo_expected(elo_w, elo_b):
    pass

def evaluate(df, players, skill_mean):
    pass

def learning_curve(df, n_players, fractions=None):
    pass

def prior_sensitivity(df, n_players, sigmas=None):
    pass

def fig_posterior_skills(trace, players, out):
    pass

def fig_skill_vs_elo(df, players, skill_mean, out):
    pass

def fig_model_comparison(results, lc_df, out):
    pass

def fig_trace_diagnostics(trace, out):
    pass

def fig_prior_sensitivity(ps_df, out):
    pass

def fig_calibration(results, out):
    pass


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df      = pd.read_csv(PROCESSED_DIR / "games_processed.csv")
    players = pd.read_csv(PROCESSED_DIR / "players.csv")["player"].tolist()
    skills  = pd.read_csv(RESULTS_DIR / "skill_estimates.csv")

    player_to_id = {p: i for i, p in enumerate(players)}
    skill_mean   = np.zeros(len(players))
    for _, row in skills.iterrows():
        if row["player"] in player_to_id:
            skill_mean[player_to_id[row["player"]]] = row["skill_mean"]

    trace_path = str(RESULTS_DIR / "trace" / "skill_trace.nc")
    trace      = az.from_netcdf(trace_path)

    results = evaluate(df, players, skill_mean)

    lc_df = learning_curve(df, len(players))
    lc_df.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)

    ps_df = prior_sensitivity(df, len(players))
    ps_df.to_csv(RESULTS_DIR / "prior_sensitivity.csv", index=False)

    fig_posterior_skills(trace, players,          RESULTS_DIR / "fig_posterior_skills.png")
    fig_skill_vs_elo    (df, players, skill_mean, RESULTS_DIR / "fig_skill_vs_elo.png")
    fig_model_comparison(results, lc_df,          RESULTS_DIR / "fig_model_comparison.png")
    fig_trace_diagnostics(trace,                  RESULTS_DIR / "fig_trace_diagnostics.png")
    fig_prior_sensitivity(ps_df,                  RESULTS_DIR / "fig_prior_sensitivity.png")
    fig_calibration     (results,                 RESULTS_DIR / "fig_calibration.png")

    metrics = {k: v for k, v in results.items() if k not in ("p_bayes", "y_true")}
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    main()