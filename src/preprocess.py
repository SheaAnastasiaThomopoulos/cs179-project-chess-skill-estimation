"""
src/preprocess.py
-----------------
Cleans raw game data and encodes player IDs.

Reads:  data/raw/games.csv
Writes: data/processed/games_processed.csv
        data/processed/players.csv

Author: Pragya Jhunjhunwala
"""

import pandas as pd
from pathlib import Path

RAW_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MIN_GAMES     = 5


def preprocess(raw_path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(raw_path)
    print(f"  Raw games loaded: {len(df)}")

    df["white"] = df["white"].str.lower().str.strip()
    df["black"] = df["black"].str.lower().str.strip()

    df = df.dropna(subset=["outcome"]).reset_index(drop=True)

    counts = pd.concat([df["white"], df["black"]]).value_counts()
    active = counts[counts >= MIN_GAMES].index
    df     = df[df["white"].isin(active) & df["black"].isin(active)].reset_index(drop=True)

    all_players  = sorted(set(df["white"]) | set(df["black"]))
    player_to_id = {p: i for i, p in enumerate(all_players)}
    df["white_id"] = df["white"].map(player_to_id)
    df["black_id"] = df["black"].map(player_to_id)

    print(f"  After filtering: {len(df)} games, {len(all_players)} players")
    print(f"  Outcome distribution: {df['outcome'].value_counts().to_dict()}")
    return df, all_players


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "games.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run src/fetch_data.py first."
        )

    df, all_players = preprocess(raw_path)

    games_out   = PROCESSED_DIR / "games_processed.csv"
    players_out = PROCESSED_DIR / "players.csv"

    df.to_csv(games_out, index=False)
    pd.DataFrame({"player": all_players}).to_csv(players_out, index=False)

    print(f"  Saved: {games_out}")
    print(f"  Saved: {players_out}")


if __name__ == "__main__":
    main()