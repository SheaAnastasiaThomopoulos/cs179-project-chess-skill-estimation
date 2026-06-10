"""
src/model.py
------------
Bradley-Terry latent skill model implemented in PyMC.

Reads:  data/processed/games_processed.csv
        data/processed/players.csv
Writes: results/trace/skill_trace.nc
        results/skill_estimates.csv

Author: Pragya Jhunjhunwala
"""

import warnings
import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR   = Path(__file__).parent.parent / "results"


def build_and_sample(df_processed: pd.DataFrame,
                     n_players:    int,
                     draws:        int = 1000,
                     tune:         int = 1000,
                     chains:       int = 2) -> az.InferenceData:
    """
    Bradley-Terry latent variable model.

        skill_i ~ Normal(0, 1)          for each player i
        P(white wins | game k)
            = sigmoid(skill_white_k - skill_black_k)

    Draws are excluded from the Bernoulli likelihood (standard BT practice).
    """
    df_decisive = df_processed[df_processed["outcome"] != 0.5].reset_index(drop=True)
    print(f"  Decisive games (used in likelihood): {len(df_decisive)}")
    print(f"  Draws excluded:                      {(df_processed['outcome'] == 0.5).sum()}")

    white_idx = df_decisive["white_id"].values
    black_idx = df_decisive["black_id"].values
    observed  = (df_decisive["outcome"].values == 1).astype(int)

    with pm.Model() as model:
        skill = pm.Normal("skill", mu=0.0, sigma=1.0, shape=n_players)
        diff  = skill[white_idx] - skill[black_idx]
        pm.Bernoulli("obs", p=pm.math.invlogit(diff), observed=observed)

        print(f"\n  Sampling: {draws} draws, {tune} tune steps, {chains} chain(s) ...")
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            progressbar=True,
            random_seed=42,
            target_accept=0.9,
        )

    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws",  type=int, default=1000)
    parser.add_argument("--tune",   type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trace_dir = RESULTS_DIR / "trace"
    trace_dir.mkdir(exist_ok=True)

    processed_path = PROCESSED_DIR / "games_processed.csv"
    if not processed_path.exists():
        raise FileNotFoundError(f"{processed_path} not found. Run src/preprocess.py first.")
    df = pd.read_csv(processed_path)

    players_path = PROCESSED_DIR / "players.csv"
    if not players_path.exists():
        raise FileNotFoundError(f"{players_path} not found. Run src/preprocess.py first.")
    players   = pd.read_csv(players_path)["player"].tolist()
    n_players = len(players)

    print(f"  Players: {n_players}  |  Total games: {len(df)}")

    trace = build_and_sample(df, n_players,
                              draws=args.draws,
                              tune=args.tune,
                              chains=args.chains)

    trace_path = str(trace_dir / "skill_trace.nc")
    trace.to_netcdf(trace_path)
    print(f"\n  Trace saved to {trace_path}")

    skill_samples = trace.posterior["skill"].values.reshape(-1, n_players)
    means  = skill_samples.mean(axis=0)
    lo89   = np.percentile(skill_samples,  5.5, axis=0)
    hi89   = np.percentile(skill_samples, 94.5, axis=0)

    skill_df = pd.DataFrame({
        "player":     players,
        "skill_mean": means,
        "skill_lo89": lo89,
        "skill_hi89": hi89,
    }).sort_values("skill_mean", ascending=False).reset_index(drop=True)

    skill_path = RESULTS_DIR / "skill_estimates.csv"
    skill_df.to_csv(skill_path, index=False)
    print(f"  Skill estimates saved to {skill_path}")

    summary   = az.summary(trace, var_names=["skill"])
    rhat_vals = pd.to_numeric(summary["r_hat"],    errors="coerce")
    ess_vals  = pd.to_numeric(summary["ess_bulk"], errors="coerce")
    print(f"\n  Median R-hat:       {rhat_vals.median():.4f}")
    print(f"  Median ESS (bulk):  {ess_vals.median():.0f}")
    print(f"  Max R-hat:          {rhat_vals.max():.4f}")


if __name__ == "__main__":
    main()