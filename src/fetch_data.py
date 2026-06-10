"""
src/fetch_data.py
-----------------
Fetches rated blitz games from the Lichess public API and saves them
as a CSV to data/raw/games.csv.

Usage:
    python src/fetch_data.py

If the Lichess API is unreachable, falls back to a synthetic dataset
so the rest of the pipeline still runs.

Author: Pragya Jhunjhunwala
"""

import json
import time
import warnings
import argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy.special import expit

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

LICHESS_USERS = [
    "DrNykterstein",    # Magnus Carlsen
    "nihalsarin2004",
    "Hikaru",
    "penguingim1",
    "alireza2003",
    "LyonBeast",
    "rpragchess",
    "GrandmaJoanne",
    "IMRosen",
    "duhless",
    "Zhigalko_Sergei",
    "anishgiri",
    "ghandeevam2003",
    "Firouzja2003",
    "vincentkeymer2002",
    "Baskaran2610",
    "RaunakSadhwani2005",
    "BambooTiger",
    "Grischuk",
    "Oleksandr_Bortnyk",
]

GAMES_PER_USER = 100
PERF_TYPE      = "blitz"
MAX_RETRIES    = 3
MIN_GAMES_PER_PLAYER = 5


def fetch_games_for_user(username: str, max_games: int = GAMES_PER_USER) -> list[dict]:
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max":      max_games,
        "rated":    "true",
        "perfType": PERF_TYPE,
        "opening":  "false",
        "clocks":   "false",
        "evals":    "false",
        "moves":    "false",
    }
    headers = {"Accept": "application/x-ndjson"}

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                stream=True, timeout=20)
            if resp.status_code == 429:
                print(f"  Rate-limited; sleeping 60s ...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                print(f"  Warning: {username} returned HTTP {resp.status_code}")
                return []
            games = []
            for line in resp.iter_lines():
                if line:
                    try:
                        games.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return games
        except requests.RequestException as e:
            print(f"  Request error ({attempt+1}/{MAX_RETRIES}): {e}")
            time.sleep(5)
    return []


def games_to_dataframe(all_games: list[dict]) -> pd.DataFrame:
    rows = []
    for g in all_games:
        try:
            players    = g.get("players", {})
            white      = players.get("white", {})
            black      = players.get("black", {})
            white_name = white.get("user", {}).get("name", None)
            black_name = black.get("user", {}).get("name", None)
            if white_name is None or black_name is None:
                continue
            white_elo = white.get("rating", None)
            black_elo = black.get("rating", None)
            status    = g.get("status", "unknown")
            winner    = g.get("winner", None)

            if winner == "white":
                outcome = 1.0
            elif winner == "black":
                outcome = 0.0
            elif status in ("draw", "stalemate"):
                outcome = 0.5
            else:
                continue

            rows.append({
                "white":     white_name.lower(),
                "black":     black_name.lower(),
                "outcome":   outcome,
                "white_elo": white_elo,
                "black_elo": black_elo,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def generate_synthetic_data(n_players: int = 20,
                             n_games:   int = 500,
                             seed:      int = 42) -> pd.DataFrame:
    rng          = np.random.default_rng(seed)
    true_skills  = rng.normal(0, 1, n_players)
    player_elos  = (true_skills * 200 + 1500).astype(int)
    player_names = [f"player_{i:02d}" for i in range(n_players)]

    rows = []
    for _ in range(n_games):
        w, b      = rng.choice(n_players, size=2, replace=False)
        diff      = true_skills[w] - true_skills[b]
        p_white   = expit(diff)
        draw_prob = 0.15 * np.exp(-0.5 * diff**2)

        r = rng.random()
        if r < p_white * (1 - draw_prob):
            outcome = 1.0
        elif r < p_white * (1 - draw_prob) + draw_prob:
            outcome = 0.5
        else:
            outcome = 0.0

        rows.append({
            "white":     player_names[w],
            "black":     player_names[b],
            "outcome":   outcome,
            "white_elo": player_elos[w],
            "black_elo": player_elos[b],
        })

    df = pd.DataFrame(rows)
    skill_map = dict(zip(player_names, true_skills))
    df["white_true_skill"] = df["white"].map(skill_map)
    df["black_true_skill"] = df["black"].map(skill_map)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--n_games", type=int, default=500)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "games.csv"

    if args.synthetic:
        print("Using synthetic data (--synthetic flag set).")
        df = generate_synthetic_data(n_games=args.n_games)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} synthetic games to {out_path}")
        return

    print("Fetching games from Lichess API ...")
    all_games = []
    for i, user in enumerate(LICHESS_USERS):
        print(f"  [{i+1}/{len(LICHESS_USERS)}] {user} ...", end=" ")
        games = fetch_games_for_user(user)
        print(f"{len(games)} games")
        all_games.extend(games)
        time.sleep(1.5)

    df = games_to_dataframe(all_games)

    if len(df) < 50:
        print(f"\nOnly {len(df)} games fetched — falling back to synthetic data.")
        df = generate_synthetic_data()
    else:
        counts = pd.concat([df["white"], df["black"]]).value_counts()
        active = counts[counts >= MIN_GAMES_PER_PLAYER].index
        df     = df[df["white"].isin(active) & df["black"].isin(active)].reset_index(drop=True)
        print(f"\nFinal: {len(df)} games, {len(active)} players")

    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()