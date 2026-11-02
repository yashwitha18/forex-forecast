# Work split

Proposed split; swap roles if you prefer, since both of you must be able to explain everything.

## Member A — Data, features, analysis   (Catherine D, CS120)

| Task | Files | Done |
|---|---|---|
| Set up env, `python main.py --download`, commit `data/raw/` | `data_loader.py` | ☑ |
| Read `data_report.json`, check flagged jumps on figure 01, record decision | `preprocess.py` | ☑ |
| Explain the data: 3 pairs, quote direction (foreign per USD), period, why Dec 2003 not 2001 | `config.py` | ☐ |
| Own the 15 FX features and the 8 macro factors; explain the publication-lag alignment | `features.py` | ☐ |
| EDA: figures 01–02, `summary_stats.csv` (fat tails, ~50% up days) | `plots.py` | ☐ |
| Report: Introduction, Related Work, Dataset, Preprocessing, Features | `REPORT.md` | ☐ |
| Slides 1–6 | `PPT_OUTLINE.md` | ☐ |

## Member B — Models, evaluation, backtest   (Yashwitha, CS121)

| Task | Files | Done |
|---|---|---|
| Review `features.py` for leakage; run `pytest` and explain each test | `tests/` | ☐ |
| Own the 6 classifiers / 7 regressors, validation selection, refit on train+val | `models.py` | ☐ |
| Own all metrics; be able to compute Acc, MAE, CU, CD, R² vs RW by hand | `evaluate.py` | ☐ |
| Own the profit simulation; reproduce the paper's $105.26 example by hand | `backtest.py` | ☐ |
| Results: Tables 1–4 equivalents, figures 03–07, compare to the paper's numbers | `results/` | ☑ |
| Report: Methods, Experiments, Results, Discussion, Limitations | `REPORT.md` | ☐ |
| Slides 7–13 + demo | `PPT_OUTLINE.md` | ☐ |

## Shared

- [ ] Handoff meeting once features are done: A walks B through the feature table, B walks A through the models.
- [ ] Each works on their own git branch and opens PRs to main.
- [ ] Abstract, Conclusion and the "paper vs ours" comparison written together.
- [ ] **Cross-explain drill:** each person answers *every* question in `VIVA_PREP.md`, not only their half.
- [ ] Demo rehearsed twice with `--quick` on cached data (no Wi-Fi needed), plus result screenshots in the slides as backup.

## Milestones (work back from the deadline: ________)

| Milestone | Owner | Date |
|---|---|---|
| Real data downloaded, cleaned, EDA figures | A | 1 Nov ✓ |
| Features reviewed, tests pass | A + B | |
| Full run done, results tables final | B | 1 Nov ✓ |
| Report draft (`docs/REPORT.md`) | A + B | 1 Nov ✓ — both review |
| Slides + demo rehearsed | A + B | |
