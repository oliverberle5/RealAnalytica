"""Logistic regression baseline — a gut-check, not a production model.

Purpose: confirm that the obviously-important signals (tenure, relative
sales performance, peer turnover) point the direction business intuition
expects, including the suspected inverted-U shape for tenure, before
trusting a more opaque model like a GBM.

Tenure handling: tenure_current_office_months and career_months are each
binned into buckets and one-hot encoded (dropping the first bucket as
reference), rather than entered as raw linear terms. A linear term could
only express "more tenure = uniformly more/less risk" — binning lets the
model freely discover a non-monotonic shape (e.g. risk peaking in the
"flight risk window" of the middle years) without assuming a parametric
form up front.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from data import RAW_FEATURE_COLUMNS, load_clean
from split import time_based_split

TENURE_BINS = [-0.01, 6, 12, 24, 36, 60, 120, np.inf]
TENURE_LABELS = ["0-6mo", "6-12mo", "1-2y", "2-3y", "3-5y", "5-10y", "10y+"]

CAREER_BINS = [-0.01, 12, 36, 60, 120, 240, np.inf]
CAREER_LABELS = ["0-1y", "1-3y", "3-5y", "5-10y", "10-20y", "20y+"]

BINNED_FEATURES = ["tenure_current_office_months", "career_months"]
CATEGORICAL_FEATURES = ["cohort_volume_quartile"]
BINARY_FEATURES = ["office_is_branch"]

# market_avg_dom_12m (state-level) is the one remaining market column with a
# real pre-2024-01 coverage gap -- kept IN the feature set (not excluded)
# since it replaces the broken per-agent avg_dom_12m as the days-on-market /
# market-heat signal, sanity-checked as legitimate (52-70 day range). Some
# training windows (e.g. entirely pre-2024) will have zero coverage of it --
# build_design_matrix's median-NaN fallback handles that case. All other
# market_* columns (market_avg_price_12m/_trend, market_total_volume_12m/
# _trend, market_total_sides_12m, market_closed_sides_trend_12m) and the
# macro columns (state_unemployment_rate/_change, cpi_yoy_pct) now have 100%
# coverage across all 20 quarters in the 2026-07 data update -- the old
# exclusion list is gone.
MARKET_FEATURES_EXCLUDED: list[str] = []

CONTINUOUS_FEATURES = [
    c
    for c in RAW_FEATURE_COLUMNS
    if c not in BINNED_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + MARKET_FEATURES_EXCLUDED
]


def _bin_column(series: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=bins, labels=labels)


def build_design_matrix(df: pd.DataFrame, medians: pd.Series | None = None):
    """Build the logistic-regression design matrix.

    If `medians` is None (fit mode), medians are computed from `df` and
    returned for reuse on the held-out set. Missingness gets an explicit
    indicator column rather than being silently imputed away.
    """
    out = pd.DataFrame(index=df.index)

    tenure_bin = _bin_column(df["tenure_current_office_months"], TENURE_BINS, TENURE_LABELS)
    career_bin = _bin_column(df["career_months"], CAREER_BINS, CAREER_LABELS)
    out = pd.concat(
        [
            out,
            pd.get_dummies(tenure_bin, prefix="tenure", drop_first=True),
            pd.get_dummies(career_bin, prefix="career", drop_first=True),
            pd.get_dummies(df["cohort_volume_quartile"], prefix="cohort_q", drop_first=True),
        ],
        axis=1,
    )

    out["office_is_branch"] = df["office_is_branch"].fillna(0)

    fit_mode = medians is None
    if fit_mode:
        medians = df[CONTINUOUS_FEATURES].median()
        # A column with ZERO coverage in this training window (e.g.
        # market_avg_dom_12m in a fold trained entirely before 2024-06-30)
        # has an undefined (NaN) median. Fall back to 0 -- the missing
        # indicator flag still marks every such row, so the model can learn
        # to discount the fabricated 0 rather than choke on a NaN.
        medians = medians.fillna(0.0)

    for col in CONTINUOUS_FEATURES:
        out[f"{col}_missing"] = df[col].isna().astype(int)
        out[col] = df[col].fillna(medians[col])

    out = out.astype(float)
    return out, medians


def main():
    df = load_clean()
    train, test = time_based_split(df)

    X_train, medians = build_design_matrix(train)
    X_test, _ = build_design_matrix(test, medians=medians)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    y_train = train["label_left_3m"].to_numpy()
    y_test = test["label_left_3m"].to_numpy()

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=2000)
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict_proba(X_train_scaled)[:, 1]
    test_pred = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n[baseline] train AUC: {roc_auc_score(y_train, train_pred):.4f}  "
          f"PR-AUC: {average_precision_score(y_train, train_pred):.4f}")
    print(f"[baseline] test  AUC: {roc_auc_score(y_test, test_pred):.4f}  "
          f"PR-AUC: {average_precision_score(y_test, test_pred):.4f}")

    coefs = pd.Series(model.coef_[0], index=X_train.columns).sort_values()
    print("\n[baseline] tenure_current_office bucket coefficients (vs 0-6mo reference):")
    print(coefs[[c for c in coefs.index if c.startswith("tenure_")]])
    print("\n[baseline] career bucket coefficients (vs 0-1y reference):")
    print(coefs[[c for c in coefs.index if c.startswith("career_")]])

    print("\n[baseline] top 10 positive (risk-increasing) coefficients:")
    print(coefs.tail(10))
    print("\n[baseline] top 10 negative (risk-decreasing) coefficients:")
    print(coefs.head(10))


if __name__ == "__main__":
    main()
