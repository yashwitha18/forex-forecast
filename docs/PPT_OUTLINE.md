# Slide outline (13 slides, ~10–12 min + 2 min demo)

All numbers come from the full run in `results/`. Figures are in `figures/`.

| # | Slide | On the slide | Visual | Speaker |
|---|---|---|---|---|
| 1 | **Title** | Foreign Currency Exchange Rate Forecast: Replicating CS229 #34 · Catherine D (CS120), Yashwitha (CS121) · ML Mini Project · Nov 2026 | — | Catherine |
| 2 | **Problem** | Can ML predict tomorrow's GBP, EUR, JPY vs USD? Meese & Rogoff (1983): a random walk beats economic models. CS229 #34 claims Random Forest R² ≈ 0.74 and +10.5% trading profit. | — | Catherine |
| 3 | **What we replicate** | 3 currency pairs · 15 FX features + 8 macro factors · 6 models · 80/10/10 split · Exp 1 sign, Exp 2 return + $100 simulation | small pipeline diagram | Catherine |
| 4 | **Data** | Yahoo FX + indexes, FRED macro · 1 Dec 2003 – 30 Nov 2018 · ~3,880 trading days per pair · paper vs ours (Bloomberg/CRSP → Yahoo/FRED; 2001 → 2003) | `01_exchange_rates.png` | Catherine |
| 5 | **Data quality** | Yahoo had 9 bad prices in 2008 (5 EUR, 4 JPY, all on the 8th of a month, 3–15% spikes that snap back). Removed by a spike-and-revert rule; Brexit kept. EUR kurtosis 89.5 → 2.3 | `00_exchange_rates_before_cleaning.png` vs `01_...` | Catherine |
| 6 | **Features & timing** | 15 FX features (returns, averages, log rate and powers) + 8 macro. Macro used only after release (CPI +45 days) → no look-ahead | feature table | Catherine |
| 7 | **Setup** | Train Oct 2004–Feb 2016 · Val Feb 2016–Jun 2017 · Test Jul 2017–Nov 2018. Baselines added: random walk, majority, momentum | split timeline | Yashwitha |
| 8 | **Exp 1: direction** | All models 0.503–0.529 accuracy (paper: 0.51–0.55). Best MLP 0.529, but p = 0.52 after correcting for 18 models → coin flip, same as the paper | `03_exp1_accuracy.png` | Yashwitha |
| 9 | **Exp 2: returns** | Validation picks Lasso with all coefficients = 0 (a constant forecast). Paper RF R² 0.70–0.78 → ours −0.06 to −0.02. Only SVM-poly > 0, by +0.004 | `04_exp2_r2_vs_rw.png` | Yashwitha |
| 10 | **"Too good" results checked** | SVM-poly 55% direction on test, but 48–51% on validation. MLP 53.5% with seed 42, but 47–52% with 5 other seeds → luck | small table | Yashwitha |
| 11 | **Profit simulation** | RF, all 3 currencies: $93.94 (no cost), $83.15 (2 bp). All 7 combos lose money with costs (paper: $100–$110). Lasso never trades: $100 | `06_cash_RandomForest.png` | Yashwitha |
| 12 | **Ours vs the paper** | Table 1 replicates (~50%); Tables 2–4 do not. Their R² 0.74 contradicts their own ~50% accuracy | 2-column table | Yashwitha |
| 13 | **Conclusion + demo** | "Without a random-walk baseline, point-in-time data and validation checks, daily FX results are easily overstated." Limitations: single split, no Diebold–Mariano test. Then live demo | — | Both |

## Demo script (~2 min)
1. Show `config.py`: pairs, period, split, the `DEVIATION` notes and `BAD_PRINT_THRESHOLD`.
2. Run `python main.py --quick` on the cached data (about 2 min, no internet needed).
3. While it runs, point at the printed split dates and the Exp 1 table.
4. Open `figures/04_exp2_r2_vs_rw.png` and `figures/06_cash_RandomForest.png`.
5. Backup: screenshots of the full-run outputs in the slides. `--quick` uses 100 trees, so its
   numbers differ slightly from the slides; say so if asked.

## Design notes
- One idea per slide; key numbers in large text.
- Keep the figure colours: GBP blue, EUR orange, JPY green.
