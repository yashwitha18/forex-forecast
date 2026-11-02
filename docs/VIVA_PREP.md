# Viva prep — both members answer all of these

## The problem and data

**1. What exactly do you predict?**
For each pair (GBP, EUR, JPY per 1 USD), at the close of trading day *t*:
Exp 1, whether the rate goes up on the next trading day (yes/no);
Exp 2, the next-day log return r = log(rate_{t+1} / rate_t).
Example: EUR=X goes from 0.8500 to 0.8534. r = log(1.004) ≈ 0.0040 (+0.40%), label = up.

**2. What does "EUR=X = 0.85" mean, and why does it matter?**
0.85 euros buy 1 dollar. If it rises, the dollar strengthened. The profit simulation depends
on this direction: a predicted ratio < 1 means the foreign currency is expected to strengthen,
so you hold it for a day.

**3. Why does your data start in Dec 2003, not 2001 like the paper?**
The paper used Bloomberg (paid). Yahoo's GBP and EUR history starts in Dec 2003, so all pairs
are trimmed to a common start. The end date matches the paper (Nov 2018), so the test period
(≈ mid-2017 to Nov 2018) is close to theirs (Mar 2017 – Nov 2018).

**4. Why a chronological split and no shuffling?**
Shuffling would put 2018 days in training and 2010 days in testing, so the model would learn
from the future. Real forecasting only ever has the past. That is why we use 80/10/10 in time order.

**5. What is the validation set for?**
Choosing hyperparameters (e.g. SVM's C, Lasso's alpha) and choosing the model for the backtest.
The test set is used once, at the end. After choosing, we refit on train+validation so the
model sees the most recent data before testing.

## Features

**6. List the 15 features.** (paper §3.2)
Today's log return, yesterday's log return, average log return over the last 5/10/20/60/120
days (these first 7 = `fx7`); log rate, its square, its cube, average log rate over the last
5/10/20/60/220 days.

**7. How do you know there's no leakage?**
Every rolling window ends at day *t* (`rolling(w)` uses t−w+1..t), and targets are shifted
−1. `test_no_future_leakage_in_fx_features` changes all prices after day *k* and checks that
no feature up to day *k* changes.

**8. Why the publication lag on macro data?**
CPI for March is dated 1 March but only released around mid-April. If we attached it to March
rows, the model would see information nobody had at the time (look-ahead). We attach each
value only from its realistic release date: CPI and recession flag +45 days, federal debt
+180 days, Fed funds +1 day, stock indexes +1 day. `test_macro_respects_publication_lag` checks this.

**9. Is the NBER recession indicator legitimate?**
Partly. NBER announces recessions months or years after they start, so even with a 45-day lag
the flag contains hindsight. The paper used it, so we kept it, but it is a limitation to state.
You can measure its effect by removing `RECESSION` from `MACRO_COLS`.

## Models

**10. Why Lasso and Ridge in a classification task?**
Following the paper. Our "Lasso" is L1-penalised logistic regression and our "Ridge" is
sklearn's RidgeClassifier (L2-penalised least squares on ±1 labels). Lasso can drive feature
weights to exactly 0; Ridge only shrinks them.

**11. Why scale features? Why scale the target in Exp 2?**
SVM, MLP, Lasso and Ridge are sensitive to feature scale (logp³ for JPY is ~100, returns are
~0.005). Daily returns are ~0.005, while SVR's default epsilon is 0.1. Unscaled, SVR would
treat every error as "inside the tube" and predict a constant. Tree models don't need scaling.

**12. Why does the Random Forest use 1000 trees and depth 100?**
Paper §4.1 says so. Depth 100 is effectively unlimited for ~3000 rows. With noisy daily
returns, deep trees can memorise noise, which is one reason RF may do worse than Lasso here.

**13. What are the baselines and why add them?**
Random walk (predict 0 return, i.e. tomorrow = today), historical mean, momentum (tomorrow's
return = today's), majority class, and "same direction as today". The paper cites Meese &
Rogoff (1983), who showed the random walk beats economic models, but it never reports a
baseline. Without one you can't say whether 0.53 accuracy is good.

## Metrics

**14. Work out Acc, MAE, NMSE, CU, CD.**
Actual: up up down down up. Predicted: up down down up up. As ±1: s = [1 1 −1 −1 1],
ŝ = [1 −1 −1 1 1].
Acc = 3/5 = 0.6. MAE = mean|s−ŝ| = (0+2+0+2+0)/5 = 0.8 = 2(1−Acc).
NMSE = mean((s−ŝ)²) / var(s) = (8/5) / 0.96 ≈ 1.67.
CU = correct among actual ups = 2/3. CD = correct among actual downs = 1/2.
DS = share with matching direction = 0.6 (= Acc for a sign classifier).

**15. The paper reports RF DS = 0.49 but Acc = 0.53. Possible?**
Not under the standard definition, where DS equals Acc for sign predictions. It is one of the
paper's inconsistencies.

**16. What is R² vs random walk, and why add it?**
R²_RW = 1 − SSE(model) / SSE(predict 0). Positive means the model beats "tomorrow = today".
Plain R² compares against predicting the test-set mean, which you can't know in advance.
R²_RW is the standard out-of-sample R² in return-forecasting research (Campbell & Thompson, 2008).

**17. The paper reports RF R² ≈ 0.74 on daily returns. Believable?**
No. R² = 0.74 means the model explains 74% of daily return variance, which would make
direction prediction almost perfect. Yet their own sign accuracy is ~0.5. Likely causes are
leakage, R² computed on something other than out-of-sample returns, or a mistake. Daily FX
R² is normally within about ±0.01 of 0. That is why our numbers will look nothing like theirs.

## Backtest

**18. Walk through the strategy with the paper's example.**
Predicted ratios GBP 1.00, EUR 1.02, JPY 0.95. JPY is lowest and < 1, so convert $100 to yen
for a day. If the actual ratio is 0.95: $100 / 0.95 = $105.26. If every predicted ratio ≥ 1,
stay in dollars.

**19. Why add transaction costs?**
Every day the strategy holds a foreign currency is a round trip (USD → foreign → USD), and a
round trip at 2 bp per conversion costs 0.04%. Our Random Forest trading all three currencies
held foreign currency on 83% of test days (~304 days): 1 − 0.9996^304 ≈ 11.5% of capital lost
to costs. That matches the result exactly: $93.94 without cost vs $83.15 with cost
(83.15 / 93.94 = 0.885). A zero-cost backtest of a daily strategy is not credible.

**20. The paper's EUR strategy made $110.50. Does that prove skill?**
No. It is one path over ~20 months, with no cost and no benchmark. When we tested the code on
synthetic random-walk data, which is unpredictable by construction, some models still ended at
$110–120 at zero cost. That is pure luck. Compare against the momentum benchmark and the
with-cost column.

## Results and limitations

**21. Did any model beat the random walk?**
Barely one, and not reliably. Test R² vs random walk ranges from −0.052 (Random Forest, fx15)
to +0.004 (SVM-poly, fx15_macro). +0.004 means 0.4% lower squared error than "tomorrow =
today". The model the validation set chose is Lasso with α = 0.1, which zeroes every
coefficient and predicts a constant, i.e. effectively the random walk.

**22. Did macro features help?**
Not reliably. In Exp 1 some models went up (MLP 0.522 → 0.529, SVM 0.503 → 0.518) and some
down (Lasso 0.520 → 0.505). In Exp 2 four of six learned models got slightly worse (e.g. MLP
R² vs RW −0.014 → −0.031). Monthly or quarterly data, published weeks late, says almost
nothing about tomorrow.

**23. Your best Exp 1 model got 52.9%. Isn't that above 50%?**
Over 1,102 test days, 583 correct gives a one-sided binomial p = 0.029. But we tried 18 model ×
feature-set combinations, so one of them reaching p < 0.05 is expected by chance. Bonferroni:
0.029 × 18 = 0.52, not significant. The models validation would actually have picked score
0.505–0.518, none significant.

**24. SVM-poly got 55.4% direction accuracy with p = 0.0002. Why don't you claim it?**
Validation would not have chosen it (validation R² −0.18), and choosing a model by its test
score is test-set snooping. Trained on the training data only, the same model gets 48.9%,
48.1% and 51.4% on the validation period: no skill there. Real skill should show up in both
periods. Script: `python -m src.diagnose`.

**25. What about the MLP's 53.5% in Exp 2?**
Seed luck. With random seeds 0–4 instead of 42, test direction accuracy is 47.4%, 51.6%, 49.2%,
50.2% and 49.9%. Script: `python -m src.diagnose MLP fx15_macro --seeds`.

**26. Why did you remove 9 data points? Isn't that cherry-picking?**
They are Yahoo errors, not market moves. Each is a 3–15% one-day jump on the 8th of a month in
2008 that fully reverses the next day, in EUR and JPY only; GBP is clean on those days. The rule
is written down before looking at model results (> 3% from both neighbours, neighbours agree),
removes nothing in GBP and keeps genuine jumps like Brexit. We re-ran everything without
removing them: Lasso is still selected, accuracies stay 0.49–0.53, Random Forest R² stays
negative. Only the fragile profit numbers move (EUR $104.32 → $100.75).

**27. Why does your result differ so much from the paper's Random Forest (R² 0.74)?**
Same model settings (1000 trees, depth 100), same features, same split ratios; we get
−0.06 to −0.02. Their 0.74 contradicts their own ~50% sign accuracy (see Q17). Our explanations:
R² computed on non-out-of-sample data, or features containing day t+1 information. We can't
check because their exact evaluation code isn't in the report.

**28. Limitations.**
Single train/val/test split; daily horizon only; Yahoo instead of Bloomberg (shorter history,
bad prices, August 2008 missing for EUR/JPY); the recession flag has hindsight; binomial tests
assume independent days and test direction only (no Diebold–Mariano test); flat 2 bp cost
model; no interest earned on the currency held.

**29. Future work.**
Walk-forward evaluation over many test windows, Diebold–Mariano tests against the random walk,
LSTM/GRU (the paper's own suggestion), and economically motivated inputs such as interest-rate
differentials.
