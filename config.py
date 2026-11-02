"""Central configuration. Change values here, not inside src/.

Replicates Stanford CS229 (Spring 2019) project #34, "Foreign Currency
Exchange Rate Forecast" (Zhang, Sun, Zhang). Every place where we differ
from the paper is marked  DEVIATION  with the reason.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

# ---- Currency pairs ----------------------------------------------------------
# Paper: USD/GBP, USD/EUR, USD/JPY, "all exchange rates here are USD/other
# currencies", i.e. units of foreign currency per 1 USD. On Yahoo Finance
# those are the tickers below (GBP=X is "GBP per USD", etc.).
PAIRS = {
    "GBP": "GBP=X",
    "EUR": "EUR=X",
    "JPY": "JPY=X",
    # Optional extra pair from the original proposal. Uncomment to include it
    # in Experiments 1 and 2. It is NOT a USD pair, so the profit simulation
    # (which trades USD against each currency) ignores it automatically.
    # "EURINR": "EURINR=X",
}
USD_BASED = {"GBP", "EUR", "JPY"}  # pairs the profit simulator may trade

# ---- Period -----------------------------------------------------------------
# Paper: Jan 2001 - Nov/Dec 2018 (Bloomberg).
# DEVIATION (data availability): Yahoo's GBP=X and EUR=X history starts in
# Dec 2003, so all pairs are trimmed to the latest common start date. The
# end date matches the paper so the test period (~2017-2018) matches theirs.
START_DATE = "2001-01-01"
END_DATE = "2018-12-01"       # exclusive in yfinance -> last day 2018-11-30
ALIGN_COMMON_START = True

# ---- Cleaning ---------------------------------------------------------------
# Yahoo's EUR=X and JPY=X contain isolated bad prices in 2008 (a one-day
# spike of 3-15% on the 8th of a month that reverts the next day). Rows that
# deviate from both neighbours by more than this are dropped. See
# src/preprocess.py. Applied to FX series only. Set to None to keep every row.
BAD_PRINT_THRESHOLD = 0.03

# ---- Macro / market factors (paper section 3.1) --------------------------------
# Paper: S&P 500, S&P 100, Nasdaq, Dow Jones (from CRSP).
# DEVIATION (data access): CRSP needs a paid subscription -> Yahoo indexes.
INDEXES = {"SP500": "^GSPC", "SP100": "^OEX", "NASDAQ": "^IXIC", "DJIA": "^DJI"}

# Paper: CPI rate of change, federal funds, NBER recession indicator,
# federal debt (from CRSP).
# DEVIATION (data access): Yahoo has none of these -> FRED (St. Louis Fed),
# the standard free source. No API key needed.
#   name: (FRED id, transform, publication lag in days)
# Publication lag = how long after the observation date the number is
# actually public. Using it prevents look-ahead (see src/features.py).
FRED_SERIES = {
    "CPI_CHG":   ("CPIAUCSL", "pct_change", 45),   # monthly, out mid next month
    "FEDFUNDS":  ("DFF",      "level",       1),   # daily, out next day
    "RECESSION": ("USREC",    "level",      45),   # monthly 0/1 flag
    "DEBT_CHG":  ("GFDEBTN",  "pct_change", 180),  # quarterly, ~3 months after quarter end
}
# Stock-index returns are lagged 1 trading day before use, to be safe about
# the US close vs the FX close time.
INDEX_LAG_DAYS = 1

# ---- Features (paper section 3.2) ---------------------------------------------
# fx7  : log return today, log return yesterday, SMA of log return over
#        5/10/20/60/120 days  (the paper's initial 7 features)
# fx15 : fx7 + log rate, its square, its cube, SMA of log rate over
#        5/10/20/60/220 days  (the paper's final 15 features)
# fx15_macro : fx15 + the 8 market/economic factors above
RET_SMA_WINDOWS = [5, 10, 20, 60, 120]
LOGP_SMA_WINDOWS = [5, 10, 20, 60, 220]   # 220 as written in the paper
FEATURE_SETS_EXP1 = ["fx7", "fx15", "fx15_macro"]
FEATURE_SETS_EXP2 = ["fx15", "fx15_macro"]
BACKTEST_FEATURE_SET = "fx15_macro"       # paper: returns + macro for backtest

# ---- Split (paper section 3.1) -----------------------------------------------------
# Paper: train / validation / test = 0.8 / 0.1 / 0.1, chronological.
SPLIT = (0.8, 0.1, 0.1)
# Hyperparameters are chosen on validation; the chosen model is then refit on
# train+validation before predicting test (standard practice).
REFIT_ON_TRAIN_VAL = True

RANDOM_STATE = 42

# Paper section 4.1: "number of trees in the forest is 1000 and the maximum
# depth of each tree is 100". Used as-is.
RF_PARAMS = dict(n_estimators=1000, max_depth=100, n_jobs=-1, random_state=RANDOM_STATE)
QUICK_RF_TREES = 100   # used with --quick for a fast demo run

# ---- Profit simulation (paper section 5.2) ----------------------------------------
START_CASH = 100.0
# Paper ignores transaction costs. DEVIATION (credibility): we report the
# paper's zero-cost result AND the result with a small cost per side, since
# the strategy trades almost daily. 0.0002 = 2 basis points per conversion.
COSTS_PER_SIDE = [0.0, 0.0002]

COLORS = {
    "GBP": "#2a78d6", "EUR": "#eb6834", "JPY": "#1baf7a", "EURINR": "#e87ba4",
    "model": "#2a78d6", "actual": "#2b2b2b", "benchmark": "#6b6a63",
}
