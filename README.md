# Foreign Currency Exchange Rate Forecast — a replication of CS229 #34

We reproduce Stanford CS229 (Spring 2019) project #34, *Foreign Currency Exchange Rate
Forecast* (J. Zhang, J. Sun, J. Zhang), on public data. We also add the baselines and checks
that make the results defensible.

**Team:** Catherine D (CS120) · Yashwitha (CS121)

**Report:** [`docs/REPORT.md`](docs/REPORT.md) · **Slides plan:** [`docs/PPT_OUTLINE.md`](docs/PPT_OUTLINE.md) · **Viva:** [`docs/VIVA_PREP.md`](docs/VIVA_PREP.md)

### Results in one paragraph
Direction prediction is a coin flip: 0.503–0.529 accuracy across all models, none significant
after correcting for multiple comparisons (the paper also found ~0.5). Return prediction does
not beat the random walk: test R² vs random walk −0.052 to +0.004, and the validation set
selects a Lasso that predicts a constant. The paper's Random Forest R² of 0.70–0.78 does not
replicate (ours: −0.06 to −0.02). Its trading simulation loses money in every currency
combination once a 2 bp cost is included ($83–$97 from $100).

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q            # 9 sanity tests
python main.py --download      # first run: fetch data, then run everything (~10 min)
python main.py                 # later runs reuse data/raw/
python main.py --quick         # 100 trees instead of 1000, for a fast live demo
python -m src.diagnose         # validation-vs-test check of the SVM-poly result
python -m src.diagnose MLP fx15_macro --seeds   # MLP result under 6 random seeds
```

`python main.py --synthetic` runs the code on fake data, for checking that it works offline.
It writes to `*_synthetic/` folders. Never report those numbers.

## What the paper did, and what we do

| | Paper | This repo | Why it differs |
|---|---|---|---|
| Pairs | USD/GBP, USD/EUR, USD/JPY | same (Yahoo `GBP=X`, `EUR=X`, `JPY=X` = foreign currency per USD) | — |
| FX data | Bloomberg, Jan 2001 – Nov 2018 | Yahoo Finance, **Dec 2003** – Nov 2018 | Bloomberg is paid. Yahoo's GBP/EUR history starts Dec 2003 |
| Market factors | S&P 500, S&P 100, Nasdaq, Dow (CRSP) | same indexes from Yahoo (`^GSPC ^OEX ^IXIC ^DJI`) | CRSP is paid |
| Economic factors | CPI change, Fed funds, NBER recession, federal debt (CRSP) | same series from FRED (`CPIAUCSL DFF USREC GFDEBTN`) | Yahoo doesn't carry them; FRED is the free standard |
| Cleaning | not described | 9 isolated Yahoo bad prices removed (5 EUR, 4 JPY, 2008) | Yahoo data error; see report §4 |
| Features | 7 → 15 FX features (+ macro) | identical definitions (`fx7`, `fx15`, `fx15_macro`) | — |
| Split | 80 / 10 / 10 chronological | same | — |
| Exp 1 | Sign prediction: Logistic, Lasso, Ridge, SVM, RF, MLP; Acc, NMSE, MAE, DS, CU, CD | same models and metrics | — |
| Exp 2 | Log-return regression, model chosen by validation R² | same, plus Linear Regression | from the original proposal |
| RF settings | 1000 trees, max depth 100 | same | — |
| Backtest | $100, switch to the currency with the lowest predicted ratio | same rule, same 7 currency combos | — |
| **Added** | — | Baselines (random walk, majority class, momentum) | the paper never measures one; without it "good" is undefined |
| **Added** | — | Macro values used only after their publication date | otherwise the model sees CPI etc. before it was released (look-ahead) |
| **Added** | — | Backtest also with 2 bp/side transaction cost, plus drawdown | the strategy trades almost daily; zero cost overstates profit |
| **Added** | — | R² vs random walk, MAE/RMSE in basis points, directional accuracy | plain R² hides whether the model beats "no change" |
| **Added** | — | Binomial significance tests with Bonferroni correction; validation and seed checks | to tell skill from luck |

The paper itself has known problems: its RF R² of ~0.74 on daily returns contradicts its ~50%
sign accuracy, and its return formula differs between report and poster. `docs/VIVA_PREP.md`
covers these. **Expect results near the random walk.** That is the normal outcome for daily FX
and agrees with Meese & Rogoff (1983).

## Outputs

| File | Paper equivalent | What it is |
|---|---|---|
| `results/data_report.json`, `summary_stats.csv` | §3.1 | rows, missing/duplicate days, flagged jumps, return statistics |
| `results/exp1_sign_avg.csv` (+ `_by_pair`) | **Table 1** | Acc, NMSE, MAE, DS, CU, CD, averaged over the 3 pairs |
| `results/exp2_return_avg.csv` (+ `_by_pair`) | **Table 2** (`val_R2`) | validation R² used for selection + test metrics |
| `results/exp2_table4_rf_by_pair.csv` | **Table 4** | Random Forest test R² per pair |
| `results/backtest_summary.csv` | **Table 3** | final cash for every model × currency combo × cost |
| `results/selected_model.txt` | §6.2.1 | which model validation R² picked |
| `results/significance_exp1.csv`, `significance_exp2_direction.csv` | — | binomial p-values, Bonferroni, validation-chosen flag |
| `results/diagnose_*.csv`, `seeds_*.csv` | — | validation-vs-test and random-seed checks |
| `figures/00_exchange_rates_before_cleaning.png` | — | raw Yahoo data, showing the 2008 bad prices |
| `figures/01_exchange_rates.png` | Figure 1 | the 3 rates with validation/test periods shaded |
| `figures/02_returns_hist.png` | — | daily return distributions |
| `figures/03_exp1_accuracy.png` | Table 1 | test accuracy vs the 0.5 coin-flip line |
| `figures/04_exp2_r2_vs_rw.png` | Table 2 | which models beat the random walk |
| `figures/05_exp2_rf_pred_vs_actual.png` | Figure 2 | RF predicted vs actual log returns |
| `figures/06_cash_*.png` | Figure 3 | cash over time, with cost and momentum benchmark |
| `figures/07_rf_importance.png` | — | what the Random Forest relied on |

## Layout

```
config.py              every setting; deviations from the paper marked DEVIATION
main.py                runs the whole pipeline
src/data_loader.py     Yahoo + FRED download and cache         (Member A)
src/preprocess.py      cleaning, data report, summary stats    (Member A)
src/features.py        paper features, macro alignment, split  (Member A, reviewed by B)
src/models.py          6 classifiers, 7 regressors, baselines  (Member B)
src/evaluate.py        paper metrics + added metrics           (Member B)
src/backtest.py        profit simulation                       (Member B)
src/plots.py           figures
src/significance.py    binomial tests, Bonferroni                (Member B)
src/diagnose.py        validation-vs-test and seed checks      (Member B)
src/synthetic.py       fake data for offline testing only
tests/                 sanity tests
docs/                  report, slide outline, viva prep, work split
```

## If a download fails

- Yahoo: open `https://finance.yahoo.com/quote/<ticker>/history`, download the CSV and save it as
  `data/raw/fx_GBP.csv`, `idx_SP500.csv` etc. It needs `date` and `close` columns.
- FRED: open `https://fred.stlouisfed.org/series/<ID>`, Download → CSV, and save it as
  `data/raw/fred_CPI_CHG.csv` etc. Keep the file's original two columns.

## Before you submit

- [x] Real data downloaded (1 Nov 2026), `data/raw/` ready to commit
- [x] Flagged jumps checked; 9 bad prints removed, decision recorded in report §4
- [x] Every number in the report copied from `results/`
- [ ] Commit and push to GitHub; add the repo link to the report appendix
- [ ] Build the slides from `docs/PPT_OUTLINE.md`
- [ ] Both members can answer everything in `docs/VIVA_PREP.md`
