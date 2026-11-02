"""Run the full replication of CS229 #34 end to end.

    python main.py --download     # first run: fetch Yahoo + FRED data into data/raw/
    python main.py                # re-run on the cached data
    python main.py --quick        # 100 trees instead of 1000 (fast demo, not for the report)
    python main.py --synthetic    # offline code check on FAKE data (never report these)

Outputs (results/, figures/) are listed in README.md.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

import config as C
from src import backtest, data_loader, evaluate, features, models, plots, preprocess, significance, synthetic

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def banner(s):
    print(f"\n{'=' * 8} {s} {'=' * (70 - len(s))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download any missing raw files")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    ap.add_argument("--quick", action="store_true", help=f"{C.QUICK_RF_TREES} RF trees (demo only)")
    ap.add_argument("--synthetic", action="store_true", help="fake data, for testing the code only")
    args = ap.parse_args()

    rf_params = dict(C.RF_PARAMS)
    if args.quick:
        rf_params["n_estimators"] = C.QUICK_RF_TREES

    if args.synthetic:
        C.RAW_DIR = C.ROOT / "data" / "synthetic"
        C.PROCESSED_DIR = C.ROOT / "data" / "processed_synthetic"
        C.RESULTS_DIR = C.ROOT / "results_synthetic"
        C.FIG_DIR = C.ROOT / "figures_synthetic"
        synthetic.write(C, C.RAW_DIR)
        print("!! SYNTHETIC DATA: these results are meaningless. Never report them. !!")
    elif args.download or args.force:
        data_loader.download_all(C, force=args.force)

    for d in (C.PROCESSED_DIR, C.RESULTS_DIR, C.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- data
    banner("DATA")
    fx_raw, idx_raw, fred = data_loader.load_all(C)
    fx, idx, report = preprocess.clean_all(fx_raw, idx_raw, C)
    (C.RESULTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2, default=str))
    stats = preprocess.summary_stats(fx)
    stats.to_csv(C.RESULTS_DIR / "summary_stats.csv")
    print(stats.round(4).to_string())
    for k, r in report.items():
        if isinstance(r, dict) and r.get("flagged_jumps"):
            print(f"[check] {k}: {len(r['flagged_jumps'])} flagged jumps -> see data_report.json")

    macro = features.macro_daily(idx, fred, C)
    tables, sets, splits, bounds = {}, None, {}, {}
    for k, df in fx.items():
        tables[k], sets = features.build(df, macro, C)
        tables[k].to_csv(C.PROCESSED_DIR / f"features_{k}.csv", index=False)
        tr, va, te = features.split(tables[k], C.SPLIT)
        splits[k] = (tr, va, te)
        bounds[k] = (va["date"].min(), te["date"].min())
        print(f"[split] {k}: train {len(tr)} ({tr['date'].min():%Y-%m-%d}..{tr['date'].max():%Y-%m-%d})  "
              f"val {len(va)} ({va['date'].min():%Y-%m-%d}..)  "
              f"test {len(te)} ({te['date'].min():%Y-%m-%d}..{te['date'].max():%Y-%m-%d})")

    plots.rates(fx, bounds, C.COLORS, C.FIG_DIR / "01_exchange_rates.png")
    plots.returns_hist(fx, C.COLORS, C.FIG_DIR / "02_returns_hist.png")

    # ------------------------------------------- Experiment 1: sign prediction
    banner("EXPERIMENT 1: SIGN PREDICTION (classification)")
    rows = []
    for k, (tr, va, te) in splits.items():
        for b in ["Majority", "SameAsToday"]:
            p = models.baseline_sign(b, pd.concat([tr, va]), te)
            rows.append({"pair": k, "feature_set": "baseline", "model": b,
                         **evaluate.sign_metrics(te["up"], p)})
        for fs in C.FEATURE_SETS_EXP1:
            for name, (est, grid) in models.classifiers(rf_params, C.RANDOM_STATE).items():
                m, params, vs = models.select_and_fit(est, grid, tr, va, sets[fs], "up",
                                                      evaluate.acc_score_fn, C.REFIT_ON_TRAIN_VAL)
                p = m.predict(te[sets[fs]])
                rows.append({"pair": k, "feature_set": fs, "model": name, "val_Acc": vs,
                             "params": json.dumps(params), **evaluate.sign_metrics(te["up"], p)})
        print(f"[exp1] {k} done ({time.time() - t0:.0f}s)")
    e1 = pd.DataFrame(rows)
    e1.to_csv(C.RESULTS_DIR / "exp1_sign_by_pair.csv", index=False)
    cols = ["Acc", "NMSE", "MAE", "DS", "CU", "CD"]
    e1_avg = e1.groupby(["feature_set", "model"], sort=False)[cols].mean().reset_index()
    e1_avg.to_csv(C.RESULTS_DIR / "exp1_sign_avg.csv", index=False)
    print("\nPaper Table 1 equivalent (test set, averaged over pairs):")
    print(e1_avg.round(3).to_string(index=False))
    plots.grouped_dots(e1_avg[e1_avg.feature_set != "baseline"], "Acc", "feature_set",
                       "Exp 1: test accuracy, averaged over pairs", C.FIG_DIR / "03_exp1_accuracy.png",
                       ref=0.5, ref_label="coin flip 0.5")

    # ------------------------------------------- Experiment 2: return regression
    banner("EXPERIMENT 2: LOG-RETURN REGRESSION")
    rows, bt_frames, rf_imp = [], {}, []
    for k, (tr, va, te) in splits.items():
        trva = pd.concat([tr, va])
        for b in ["RandomWalk", "HistMean", "Momentum"]:
            p = models.baseline_return(b, trva, te)
            rows.append({"pair": k, "feature_set": "baseline", "model": b,
                         **evaluate.return_metrics(te["next_ret"], p)})
            bt_frames.setdefault(b, {})[k] = te.assign(pred_ret=p)
        for fs in C.FEATURE_SETS_EXP2:
            for name, (est, grid) in models.regressors(rf_params, C.RANDOM_STATE).items():
                m, params, vs = models.select_and_fit(est, grid, tr, va, sets[fs], "next_ret",
                                                      evaluate.r2_score_fn, C.REFIT_ON_TRAIN_VAL)
                p = m.predict(te[sets[fs]])
                rows.append({"pair": k, "feature_set": fs, "model": name, "val_R2": vs,
                             "params": json.dumps(params), **evaluate.return_metrics(te["next_ret"], p)})
                if fs == C.BACKTEST_FEATURE_SET:
                    bt_frames.setdefault(name, {})[k] = te.assign(pred_ret=p)
                    if name == "RandomForest":
                        rf_imp.append(pd.Series(m.feature_importances_, index=sets[fs]))
        print(f"[exp2] {k} done ({time.time() - t0:.0f}s)")
    e2 = pd.DataFrame(rows)
    e2.to_csv(C.RESULTS_DIR / "exp2_return_by_pair.csv", index=False)
    cols = ["val_R2", "R2", "R2_vs_RW", "MAE_bp", "RMSE_bp", "DirAcc"]
    e2_avg = e2.groupby(["feature_set", "model"], sort=False)[cols].mean().reset_index()
    e2_avg.to_csv(C.RESULTS_DIR / "exp2_return_avg.csv", index=False)
    print("\nPaper Table 2 equivalent = val_R2 (used to pick the model). Test metrics averaged over pairs:")
    print(e2_avg.round(4).to_string(index=False))

    t4 = e2[(e2.model == "RandomForest") & (e2.feature_set == C.BACKTEST_FEATURE_SET)]
    t4 = t4.set_index("pair")[["R2", "R2_vs_RW", "MAE_bp", "DirAcc"]]
    t4.to_csv(C.RESULTS_DIR / "exp2_table4_rf_by_pair.csv")
    print("\nPaper Table 4 equivalent (Random Forest, test R^2 by pair):")
    print(t4.round(4).to_string())

    plots.grouped_dots(e2_avg[e2_avg.feature_set != "baseline"], "R2_vs_RW", "feature_set",
                       "Exp 2: test R² vs random walk, averaged over pairs (> 0 beats random walk)",
                       C.FIG_DIR / "04_exp2_r2_vs_rw.png", ref=0.0, ref_label="random walk")
    plots.pred_vs_actual(bt_frames["RandomForest"], "RandomForest", C.FIG_DIR / "05_exp2_rf_pred_vs_actual.png")
    if rf_imp:
        plots.importances(pd.concat(rf_imp, axis=1).mean(axis=1),
                          f"Random Forest feature importance ({C.BACKTEST_FEATURE_SET}, mean over pairs)",
                          C.FIG_DIR / "07_rf_importance.png")

    # ------------------------------------------- Profit simulation
    banner("PROFIT SIMULATION (test period)")
    ccys = [k for k in C.PAIRS if k in C.USD_BASED]
    pred = {m: {k: f for k, f in fr.items() if k in ccys} for m, fr in bt_frames.items() if m != "HistMean"}
    bt_sum, paths = backtest.run(pred, ccys, C.START_CASH, C.COSTS_PER_SIDE)
    bt_sum.to_csv(C.RESULTS_DIR / "backtest_summary.csv", index=False)

    bt_avg = e2_avg[e2_avg.feature_set == C.BACKTEST_FEATURE_SET]
    chosen = bt_avg.sort_values("val_R2", ascending=False)["model"].iloc[0]
    (C.RESULTS_DIR / "selected_model.txt").write_text(
        f"Model with best mean validation R^2 on {C.BACKTEST_FEATURE_SET}: {chosen}\n"
        f"(the paper selected RandomForest the same way)\n")
    print(f"Model selected by validation R^2: {chosen}  (paper: RandomForest)")
    for m in dict.fromkeys(["RandomForest", chosen, "Momentum"]):
        t3 = bt_sum[bt_sum.model == m].pivot(index="combo", columns="cost_per_side", values="final_cash")
        print(f"\nPaper Table 3 equivalent - final cash from $100, model = {m}:")
        print(t3.round(2).to_string())
    all3 = "+".join(ccys)
    combos = [all3] + ccys if len(ccys) > 1 else ccys
    for m in dict.fromkeys(["RandomForest", chosen]):
        plots.cash_paths(paths, m, combos, max(C.COSTS_PER_SIDE), C.FIG_DIR / f"06_cash_{m}.png")

    # ------------------------------------------- Significance
    banner("SIGNIFICANCE (is any accuracy above 0.5 more than luck?)")
    s1, s2 = significance.run(C.RESULTS_DIR, C.PROCESSED_DIR)
    cols = ["feature_set", "model", "test_acc", "p_value", "p_bonferroni", "chosen_by_validation"]
    print("Exp 1, top 5 by test accuracy:")
    print(s1[cols].head(5).round(4).to_string(index=False))
    print("Exp 2 (direction of predicted return), top 5:")
    print(s2[cols].head(5).round(4).to_string(index=False))

    banner(f"DONE in {time.time() - t0:.0f}s -> {C.RESULTS_DIR.name}/, {C.FIG_DIR.name}/")


if __name__ == "__main__":
    main()
