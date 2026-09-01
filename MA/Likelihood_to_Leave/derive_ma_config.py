"""Decide the MA model's feature list honestly, instead of assuming RI's transfers.

The RI production config prunes the bottom-45-importance features (keeping 49 of
94), where the ranking came from fitting a depth=3 SqrtBalanced CatBoost on
FOLD 1's TRAINING WINDOW ONLY. That ranking is a property of RIAR data. Two
things could make it wrong for MLSPIN: MA has a brand RI doesn't (`bhgre`), and
the brand mix differs sharply (berkshire_hathaway 3.40% of MA rows vs 0.01% of
RI; remax 4.32% vs 10.50%). A feature that ranked near-zero in RI purely because
the brand barely existed there could matter in MA.

So this script measures BOTH on the SAME folds and prints them side by side:

  A. TRANSFERRED  -- RI's hardcoded PRODUCTION_FEATURES, used as-is.
  B. RE-DERIVED   -- the identical pruning procedure re-run on MA's fold-1
                     training window, producing an MA-specific list.

Neither is assumed to win. Whichever is chosen gets hardcoded into crossval.py
as a fixed, reviewable constant -- never recomputed at import time -- exactly as
the RI project does it.

IMPORTANT (carried over from the RI project's stated honesty bar): a bare mean-AUC
difference between A and B is NOT by itself evidence that one is better. The RI
project documents a 95% CI of about +/-0.019 on 5-fold mean AUC, and that figure
understates it since folds share training windows and agents. Per-fold numbers
and the worst fold are printed so a difference can be judged against fold spread
rather than a single averaged number. MA's larger panel (~11.4k positives vs RI's
~2.2k) should tighten this relative to RI, but not to zero.

Runs on the current production population (exclude_team=True, non-member
placeholder excluded) -- i.e. what load_clean() returns with no arguments.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from crossval import PRODUCTION_CATBOOST_PARAMS, PRODUCTION_FEATURES, _forward_folds
from data import RAW_FEATURE_COLUMNS, load_clean

TREE_FEATURES = list(RAW_FEATURE_COLUMNS)

# Four constraints, matching crossval.py::MONOTONE. `volume_trend_vs_market_12m`
# was removed 2026-08-07 -- it never bound in the deployed model (not in
# PRODUCTION_FEATURES), see crossval.py for the full note.
MONOTONE = {
    "office_exit_rate_12m": 1,
    "months_since_last_closing": 1,
    "share_of_office_volume_12m": -1,
    "company_exit_rate_12m": 1,
}

N_DROP = 45  # same count the RI derivation used


def _xy(df, features):
    return df[features].astype(float), df["label_left_3m"].to_numpy()


def _pruned_features(train_first_fold, n_drop=N_DROP):
    """Identical procedure to the RI project's derive_winning_config.py."""
    from catboost import CatBoostClassifier

    X, y = _xy(train_first_fold, TREE_FEATURES)
    model = CatBoostClassifier(
        iterations=400, depth=3, learning_rate=0.03, l2_leaf_reg=3.0,
        monotone_constraints=[MONOTONE.get(c, 0) for c in TREE_FEATURES],
        auto_class_weights="SqrtBalanced",
        loss_function="Logloss", eval_metric="PRAUC", verbose=False,
        allow_writing_files=False,
    )
    model.fit(X, y)
    ranked = sorted(zip(TREE_FEATURES, model.get_feature_importance()), key=lambda t: t[1])
    dropped = {c for c, _ in ranked[:n_drop]}
    return [c for c in TREE_FEATURES if c not in dropped]


def _fit_eval(train, test, features):
    from catboost import CatBoostClassifier

    X_train, y_train = _xy(train, features)
    X_test, y_test = _xy(test, features)
    model = CatBoostClassifier(
        monotone_constraints=[MONOTONE.get(c, 0) for c in features],
        **PRODUCTION_CATBOOST_PARAMS,
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, p), average_precision_score(y_test, p)


def main():
    df = load_clean()
    folds = _forward_folds(df["snapshot_date"].unique())
    print(f"\n[cfg] {len(folds)} forward-chaining folds, {len(df)} rows, "
          f"{df['label_left_3m'].sum()} positives ({df['label_left_3m'].mean():.3%})\n")

    train_fold1 = df[df["snapshot_date"].isin(folds[0][0])]
    rederived = _pruned_features(train_fold1)

    transferred = [c for c in PRODUCTION_FEATURES if c in df.columns]
    missing = [c for c in PRODUCTION_FEATURES if c not in df.columns]
    if missing:
        print(f"[cfg] WARNING: {len(missing)} RI production features absent from MA data: {missing}\n")

    only_ri = [c for c in transferred if c not in rederived]
    only_ma = [c for c in rederived if c not in transferred]
    print(f"[cfg] TRANSFERRED (RI) list: {len(transferred)} features")
    print(f"[cfg] RE-DERIVED  (MA) list: {len(rederived)} features")
    print(f"[cfg] overlap: {len(set(transferred) & set(rederived))}")
    print(f"\n[cfg] kept by RI but pruned by MA ({len(only_ri)}):")
    for c in only_ri:
        print(f"    {c}")
    print(f"\n[cfg] kept by MA but pruned by RI ({len(only_ma)}):")
    for c in only_ma:
        print(f"    {c}")

    variants = {"transferred_RI": transferred, "rederived_MA": rederived}
    results = {k: {"auc": [], "prauc": []} for k in variants}

    print("\n" + "=" * 78)
    for fi, (train_dates, test_dates) in enumerate(folds, 1):
        train = df[df["snapshot_date"].isin(train_dates)]
        test = df[df["snapshot_date"].isin(test_dates)]
        line = f"[fold {fi}] test {min(test_dates).date()}..{max(test_dates).date()} " \
               f"({len(test)} rows, {test['label_left_3m'].mean():.2%} pos)"
        print(line)
        for name, feats in variants.items():
            auc, prauc = _fit_eval(train, test, feats)
            results[name]["auc"].append(auc)
            results[name]["prauc"].append(prauc)
            print(f"          {name:16s} AUC {auc:.4f}  PR-AUC {prauc:.4f}")
    print("=" * 78)

    print(f"\n{'variant':18s} {'mean AUC':>10s} {'worst':>10s} {'spread':>10s} {'mean PR':>10s}")
    for name in variants:
        a = np.array(results[name]["auc"])
        print(f"{name:18s} {a.mean():10.6f} {a.min():10.6f} {a.max()-a.min():10.6f} "
              f"{np.mean(results[name]['prauc']):10.6f}")

    d = np.array(results["rederived_MA"]["auc"]) - np.array(results["transferred_RI"]["auc"])
    print(f"\nper-fold delta (MA re-derived - RI transferred): "
          f"{' '.join(f'{x:+.4f}' for x in d)}")
    print(f"mean delta {d.mean():+.6f}, wins {int((d > 0).sum())}/{len(d)} folds")
    print("\nJudge this against fold spread above, not in isolation -- see module docstring.")

    print("\n\nMA RE-DERIVED FEATURE LIST (paste into crossval.py if adopted):")
    print("MA_PRODUCTION_FEATURES = [")
    for c in rederived:
        print(f"    {c!r},")
    print("]")


if __name__ == "__main__":
    main()
