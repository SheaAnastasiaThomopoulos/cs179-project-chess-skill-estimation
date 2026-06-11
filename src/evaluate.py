"""
src/evaluate.py
---------------
Author: Shea Thomopoulos
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
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_w) / 400))


def evaluate(df, players, skill_mean):
    n       = len(df)
    test_df = df.iloc[int(0.8 * n):].reset_index(drop=True)
    test_df = test_df[test_df["outcome"] != 0.5].reset_index(drop=True)

    if len(test_df) == 0:
        test_df = df[df["outcome"] != 0.5].reset_index(drop=True)

    p_bayes = expit(skill_mean[test_df["white_id"].values]
                    - skill_mean[test_df["black_id"].values])
    y_true  = (test_df["outcome"].values == 1).astype(int)

    results = {
        "bayes_accuracy": accuracy_score(y_true, (p_bayes >= 0.5).astype(int)),
        "bayes_logloss":  log_loss(y_true, p_bayes),
    }

    has_elo = test_df["white_elo"].notna() & test_df["black_elo"].notna()
    if has_elo.sum() > 10:
        df_elo = test_df[has_elo].reset_index(drop=True)
        p_elo  = np.array([elo_expected(r["white_elo"], r["black_elo"])
                            for _, r in df_elo.iterrows()])
        y_elo  = (df_elo["outcome"].values == 1).astype(int)
        results["elo_accuracy"] = accuracy_score(y_elo, (p_elo >= 0.5).astype(int))
        results["elo_logloss"]  = log_loss(y_elo, p_elo)
    else:
        results["elo_accuracy"] = 0.5
        results["elo_logloss"]  = log_loss(y_true, np.full_like(p_bayes, 0.5))

    results["p_bayes"] = p_bayes
    results["y_true"]  = y_true
    return results


def learning_curve(df, n_players, fractions=None):
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8]

    df_d    = df[df["outcome"] != 0.5].reset_index(drop=True)
    records = []

    for frac in fractions:
        n_train  = int(frac * len(df_d))
        df_train = df_d.iloc[:n_train]
        df_val   = df_d.iloc[n_train:].reset_index(drop=True)
        if len(df_train) < 20 or len(df_val) < 10:
            continue

        with pm.Model():
            skill = pm.Normal("skill", mu=0, sigma=1, shape=n_players)
            diff  = skill[df_train["white_id"].values] - skill[df_train["black_id"].values]
            pm.Bernoulli("obs", p=pm.math.invlogit(diff),
                         observed=(df_train["outcome"].values == 1).astype(int))
            tr = pm.sample(500, tune=500, chains=1, cores=1,
                           progressbar=False, random_seed=42)

        sm = tr.posterior["skill"].values.reshape(-1, n_players).mean(0)
        pv = expit(sm[df_val["white_id"].values] - sm[df_val["black_id"].values])
        yv = (df_val["outcome"].values == 1).astype(int)
        records.append({
            "n_games":  n_train,
            "fraction": frac,
            "accuracy": accuracy_score(yv, (pv >= 0.5).astype(int)),
            "log_loss": log_loss(yv, pv),
        })
        print(f"  frac={frac:.1f}  n={n_train}  acc={records[-1]['accuracy']:.3f}  ll={records[-1]['log_loss']:.3f}")

    return pd.DataFrame(records)


def prior_sensitivity(df, n_players, sigmas=None):
    if sigmas is None:
        sigmas = [0.25, 0.5, 1.0, 2.0, 4.0]

    df_d     = df[df["outcome"] != 0.5].reset_index(drop=True)
    n        = len(df_d)
    df_train = df_d.iloc[:int(0.8 * n)]
    df_val   = df_d.iloc[int(0.8 * n):].reset_index(drop=True)

    if len(df_val) < 10:
        df_train = df_d.iloc[:int(0.9 * n)]
        df_val   = df_d.iloc[int(0.9 * n):].reset_index(drop=True)

    records = []
    for sigma in sigmas:
        with pm.Model():
            skill = pm.Normal("skill", mu=0, sigma=sigma, shape=n_players)
            diff  = skill[df_train["white_id"].values] - skill[df_train["black_id"].values]
            pm.Bernoulli("obs", p=pm.math.invlogit(diff),
                         observed=(df_train["outcome"].values == 1).astype(int))
            tr = pm.sample(500, tune=500, chains=1, cores=1,
                           progressbar=False, random_seed=42)

        sm       = tr.posterior["skill"].values.reshape(-1, n_players).mean(0)
        pv       = expit(sm[df_val["white_id"].values] - sm[df_val["black_id"].values])
        yv       = (df_val["outcome"].values == 1).astype(int)
        post_std = tr.posterior["skill"].values.reshape(-1, n_players).std(axis=0).mean()

        records.append({
            "sigma":         sigma,
            "accuracy":      accuracy_score(yv, (pv >= 0.5).astype(int)),
            "log_loss":      log_loss(yv, pv),
            "mean_post_std": float(post_std),
        })
        print(f"  sigma={sigma:.2f}  acc={records[-1]['accuracy']:.3f}  "
              f"ll={records[-1]['log_loss']:.3f}  post_std={post_std:.3f}")

    return pd.DataFrame(records)


def fig_posterior_skills(trace, players, out):
    # Figure 1 in report: horizontal bar chart sorted by posterior mean
    n       = len(players)
    samples = trace.posterior["skill"].values.reshape(-1, n)
    means   = samples.mean(0)
    lo      = np.percentile(samples,  5.5, 0)
    hi      = np.percentile(samples, 94.5, 0)
    order   = np.argsort(means)
    labels  = [players[i][:14] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.35)))
    y = np.arange(n)
    ax.barh(y, means[order],
            xerr=[means[order]-lo[order], hi[order]-means[order]],
            color="steelblue", alpha=0.7, ecolor="black", capsize=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Posterior Skill (latent units)")
    ax.set_title("Posterior Skill Estimates with 89% Credible Intervals")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_trace_diagnostics(trace, out):
    # Figure 2 in report: R-hat and ESS histograms
    summary   = az.summary(trace, var_names=["skill"])
    rhat_vals = pd.to_numeric(summary["r_hat"],    errors="coerce")
    ess_vals  = pd.to_numeric(summary["ess_bulk"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(rhat_vals.dropna(), bins=15, color="steelblue", alpha=0.8, edgecolor="white")
    axes[0].axvline(1.01, color="red", linestyle="--", label="R-hat = 1.01")
    axes[0].set_xlabel("R-hat")
    axes[0].set_title("R-hat Convergence")
    axes[0].legend()

    axes[1].hist(ess_vals.dropna(), bins=15, color="coral", alpha=0.8, edgecolor="white")
    axes[1].axvline(400, color="red", linestyle="--", label="ESS = 400")
    axes[1].set_xlabel("ESS (bulk)")
    axes[1].set_title("Effective Sample Size")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_model_comparison(results, lc_df, out):
    # Figure 3 in report: bar chart comparison + learning curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: Bayesian vs Elo bar chart
    ax = axes[0]
    metrics   = ["Accuracy", "Log-Loss"]
    bayes_val = [results["bayes_accuracy"], results["bayes_logloss"]]
    elo_val   = [results["elo_accuracy"],   results["elo_logloss"]]
    x = np.arange(2)
    w = 0.3
    ax.bar(x-w/2, bayes_val, w, label="Bayesian", color="steelblue", alpha=0.8)
    ax.bar(x+w/2, elo_val,   w, label="Elo",      color="coral",     alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Heldout Model Comparison")
    ax.legend()
    for bar in ax.patches:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.3f}", ha="center", fontsize=8)

    # Right: learning curve
    ax2 = axes[1]
    if lc_df is not None and len(lc_df) > 1:
        ax2.plot(lc_df["n_games"], lc_df["accuracy"], "o-", color="steelblue",
                 linewidth=2, markersize=6, label="Accuracy")
        ax2r = ax2.twinx()
        ax2r.plot(lc_df["n_games"], lc_df["log_loss"], "s--", color="coral",
                  linewidth=2, markersize=6, label="Log-Loss")
        ax2.set_xlabel("Training Games")
        ax2.set_ylabel("Accuracy", color="steelblue")
        ax2r.set_ylabel("Log-Loss", color="coral")
        ax2.set_title("Learning Curve")
        l1, lb1 = ax2.get_legend_handles_labels()
        l2, lb2 = ax2r.get_legend_handles_labels()
        ax2.legend(l1+l2, lb1+lb2, fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_calibration(results, out):
    # Figure 4 in report: calibration plot
    p_pred = results["p_bayes"]
    y_true = results["y_true"]
    bins   = np.linspace(0, 1, 11)
    bm, br = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() >= 3:
            bm.append(p_pred[mask].mean())
            br.append(y_true[mask].mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Perfect calibration")
    ax.scatter(bm, br, s=80, color="steelblue", zorder=5, label="Model")
    ax.plot(bm, br, color="steelblue", alpha=0.5)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.set_title("Calibration Plot for Bayesian Predictions")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_skill_vs_elo(df, players, skill_mean, out):
    # Figure 5 in report: posterior mean skill vs average Elo scatter
    elo_map = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("white_elo")):
            elo_map.setdefault(row["white"], []).append(row["white_elo"])
        if pd.notna(row.get("black_elo")):
            elo_map.setdefault(row["black"], []).append(row["black_elo"])

    xs, ys, names = [], [], []
    for i, p in enumerate(players):
        if p in elo_map and len(elo_map[p]) >= 3:
            xs.append(skill_mean[i])
            ys.append(np.mean(elo_map[p]))
            names.append(p[:12])

    if len(xs) < 5:
        print("  Skipping skill-vs-Elo plot (not enough Elo data).")
        return

    r, _ = pearsonr(xs, ys)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, alpha=0.7, s=60, color="steelblue")
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), fontsize=7, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")
    m, b = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 100)
    ax.plot(xr, m*xr+b, "r--", linewidth=1.5, label=f"r = {r:.2f}")
    ax.set_xlabel("Posterior Mean Skill")
    ax.set_ylabel("Average Lichess Elo")
    ax.set_title("Posterior Mean Skill vs. Average Lichess Elo")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def fig_prior_sensitivity(ps_df, out):
    # Figure 6 in report: prior sensitivity two-panel
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(ps_df["sigma"], ps_df["accuracy"], "o-", color="steelblue",
            linewidth=2, markersize=7, label="Accuracy")
    ax2 = ax.twinx()
    ax2.plot(ps_df["sigma"], ps_df["log_loss"], "s--", color="coral",
             linewidth=2, markersize=7, label="Log-Loss")
    ax.set_xlabel("Prior Std (σ)")
    ax.set_ylabel("Accuracy", color="steelblue")
    ax2.set_ylabel("Log-Loss", color="coral")
    ax.set_title("Prior Sensitivity: Predictive Performance")
    ax.set_xscale("log")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, lb1+lb2, fontsize=9)
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    ax3 = axes[1]
    ax3.plot(ps_df["sigma"], ps_df["mean_post_std"], "D-", color="seagreen",
             linewidth=2, markersize=7)
    ax3.set_xlabel("Prior Std (σ)")
    ax3.set_ylabel("Mean Posterior Std of Skill")
    ax3.set_title("Prior Width vs. Posterior Uncertainty")
    ax3.set_xscale("log")
    ax3.axvline(1.0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


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

    print("Running evaluation ...")
    results = evaluate(df, players, skill_mean)

    print("Running learning curve ...")
    lc_df = learning_curve(df, len(players))
    lc_df.to_csv(RESULTS_DIR / "learning_curve.csv", index=False)

    print("Running prior sensitivity ...")
    ps_df = prior_sensitivity(df, len(players))
    ps_df.to_csv(RESULTS_DIR / "prior_sensitivity.csv", index=False)

    print("Generating figures ...")
    fig_posterior_skills(trace, players,          RESULTS_DIR / "fig_posterior_skills.png")
    fig_trace_diagnostics(trace,                  RESULTS_DIR / "fig_trace_diagnostics.png")
    fig_model_comparison(results, lc_df,          RESULTS_DIR / "fig_model_comparison.png")
    fig_calibration     (results,                 RESULTS_DIR / "fig_calibration.png")
    fig_skill_vs_elo    (df, players, skill_mean, RESULTS_DIR / "fig_skill_vs_elo.png")
    fig_prior_sensitivity(ps_df,                  RESULTS_DIR / "fig_prior_sensitivity.png")

    metrics = {k: v for k, v in results.items() if k not in ("p_bayes", "y_true")}
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\n── Final Metrics ──────────────────────────────")
    print(f"  Bayesian accuracy:  {results['bayes_accuracy']:.3f}")
    print(f"  Bayesian log-loss:  {results['bayes_logloss']:.3f}")
    print(f"  Elo accuracy:       {results['elo_accuracy']:.3f}")
    print(f"  Elo log-loss:       {results['elo_logloss']:.3f}")


if __name__ == "__main__":
    main()