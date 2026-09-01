"""
Quick baseline backtest: how well does a simple model do at 12m/24m/36m
sales-forecast horizons? Standalone script -- does not modify data.py.
Mirrors the project's existing "round 1" gut-check methodology (Ridge vs.
naive persistence vs. naive mean, single forward time split, same metrics)
but generalizes the target construction to an arbitrary horizon in months,
reusing the exact self-join logic already proven correct for 12m.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

import data


def build_targets_h(df: pd.DataFrame, months: int) -> pd.DataFrame:
    df = df.sort_values(["mls_agent_id", "snapshot_date"]).reset_index(drop=True)
    future = df[["mls_agent_id", "snapshot_date", "volume_12m", "units_12m"]].copy()
    future = future.rename(columns={
        "volume_12m": f"target_volume_next_{months}m",
        "units_12m": f"target_units_next_{months}m",
    })
    future["snapshot_date"] = future["snapshot_date"] - pd.DateOffset(months=months)
    merged = df.merge(future, on=["mls_agent_id", "snapshot_date"], how="left")
    target_cols = [f"target_volume_next_{months}m", f"target_units_next_{months}m"]
    resolved = merged.dropna(subset=target_cols).reset_index(drop=True)
    return resolved, target_cols


def evaluate(months: int, test_frac: float = 0.3):
    raw = data.load_clean()
    df, target_cols = build_targets_h(raw, months)
    df = pd.get_dummies(df, columns=data.CATEGORICAL_FEATURE_COLUMNS, prefix="brand")

    quarters = sorted(df["snapshot_date"].unique())
    n_test = max(1, round(len(quarters) * test_frac))
    train_quarters = quarters[:-n_test]
    test_quarters = quarters[-n_test:]

    train = df[df["snapshot_date"].isin(train_quarters)]
    test = df[df["snapshot_date"].isin(test_quarters)]

    exclude = set(data.ID_COLUMNS) | set(target_cols) | {"target_volume_next_12m", "target_units_next_12m"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for target_col, label in [(target_cols[0], "volume"), (target_cols[1], "units")]:
        X_train, y_train = train[feature_cols], train[target_col]
        X_test, y_test = test[feature_cols], test[target_col]

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)
        X_train_s = scaler.fit_transform(X_train_i)
        X_test_s = scaler.transform(X_test_i)

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_s, y_train)
        pred = np.clip(ridge.predict(X_test_s), 0, None)

        # naive persistence: predict target = current trailing-12m value
        source_col = "volume_12m" if label == "volume" else "units_12m"
        naive_pred = test[source_col].values

        # naive mean
        mean_pred = np.full(len(y_test), y_train.mean())

        def metrics(y_true, y_pred, name):
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            sp = spearmanr(y_true, y_pred).correlation if np.std(y_pred) > 0 else float("nan")
            return dict(horizon=months, target=label, model=name, n_train=len(y_train),
                        n_test=len(y_test), mae=mae, rmse=rmse, r2=r2, spearman=sp)

        results.append(metrics(y_test.values, pred, "Ridge"))
        results.append(metrics(y_test.values, naive_pred, "naive_persistence"))
        results.append(metrics(y_test.values, mean_pred, "naive_mean"))

    return results, train_quarters, test_quarters


if __name__ == "__main__":
    all_results = []
    for months in [12, 24, 36]:
        results, train_q, test_q = evaluate(months)
        print(f"\n=== horizon {months}m -- train quarters {train_q[0].date()}..{train_q[-1].date()} "
              f"({len(train_q)}q) / test quarters {test_q[0].date()}..{test_q[-1].date()} ({len(test_q)}q) ===")
        for r in results:
            print(f"  {r['target']:>6} | {r['model']:>18} | n_test={r['n_test']:>6} | "
                  f"MAE={r['mae']:>14,.1f} | RMSE={r['rmse']:>14,.1f} | R2={r['r2']:>7.4f} | Spearman={r['spearman']:>7.4f}")
        all_results.extend(results)

    out = pd.DataFrame(all_results)
    out.to_csv("backtest_long_horizons_results.csv", index=False)
    print("\nSaved backtest_long_horizons_results.csv")
