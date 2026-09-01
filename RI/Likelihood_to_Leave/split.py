"""Time-based train/test split.

Train on the earliest snapshots, test on the most recent ones, so evaluation
mimics the real deployment scenario (predicting forward in time) rather than
letting future market conditions leak into training via a random split.
"""

import pandas as pd

TRAIN_FRACTION = 0.70


def time_based_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    dates = sorted(df["snapshot_date"].unique())
    n_train_dates = max(1, round(len(dates) * train_fraction))
    train_dates = set(dates[:n_train_dates])
    test_dates = set(dates[n_train_dates:])

    train = df[df["snapshot_date"].isin(train_dates)].copy()
    test = df[df["snapshot_date"].isin(test_dates)].copy()

    train_agents = set(train["mls_agent_id"])
    test_agents = set(test["mls_agent_id"])
    overlap = train_agents & test_agents

    print(
        f"[split] train: {sorted(d.date() for d in train_dates)[0]} .. "
        f"{sorted(d.date() for d in train_dates)[-1]} ({len(train)} rows, "
        f"{train['label_left_3m'].mean():.3%} positive)"
    )
    print(
        f"[split] test:  {sorted(d.date() for d in test_dates)[0]} .. "
        f"{sorted(d.date() for d in test_dates)[-1]} ({len(test)} rows, "
        f"{test['label_left_3m'].mean():.3%} positive)"
    )
    print(
        f"[split] agent overlap: {len(overlap)} of {len(test_agents)} test agents "
        f"also appear in train (expected/fine — this is forecasting the same "
        f"population forward, not a fresh-agent generalization test)."
    )

    return train, test


if __name__ == "__main__":
    from data import load_clean

    df = load_clean()
    time_based_split(df)
