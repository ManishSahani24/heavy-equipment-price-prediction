# Methodology

This document explains *why* each step of the pipeline exists. Code references point to
`src/heavy_equipment/`.

## 1. Target and metric

The competition metric is RMSLE:

$$\text{RMSLE} = \sqrt{\frac{1}{n}\sum_i \big(\log(1+\hat p_i) - \log(1+p_i)\big)^2}$$

Defining $y = \log(1+p)$ turns this into plain RMSE on $y$. All models are therefore trained on
`log1p(TargetValue)` with an RMSE objective, early-stopped on validation RMSE, and predictions are
mapped back with `expm1` (`data.py`). This also compresses the long right tail of the price distribution.

## 2. Train/test stacking

Train and test are concatenated (`_is_test` flag) so every feature is computed identically on both.
Two rules keep this leakage-free:

- Anything that is **imputed** (medians for manufacture year, hours, model number, numeric NaNs) is
  computed on **train rows only**.
- Group statistics over the combined frame use only **non-target** columns (hours, age, dates, counts).
  This is a transductive use of the test features, not of the test labels.

## 3. Cleaning (`features.py`)

| Issue | Treatment |
|---|---|
| `ProductConfigID`, `AssetID` are numeric codes | cast to string → treated as categoricals |
| `col18`, `col19` are >99 % empty | dropped |
| `ManufactureYear` = 1000/1001 or outside 1900–2030 | NaN → `ManufactureYear_missing` flag → train median (1999) |
| `OperationalHoursMeter` = 0 | NaN → `Hours_missing` flag → train median (3,997 h) |
| missing categoricals | explicit `"Unknown"` level |

## 4. Feature engineering

| Block | Features | Intuition |
|---|---|---|
| Date | year, month, day, weekday, quarter, day-of-year, days since first sale, decade, year² , year-start/end flags | market level and seasonality |
| Age | `EquipmentAge` = sale year − manufacture year (clipped 0–65), log/sqrt/square, decade, 5 buckets, `IsNew` (≤2 y), `IsOld` (≥20 y) | depreciation is the main price driver |
| Usage | log/sqrt hours, hours per year of age, log hours × log age, log hours × sale year | wear relative to age |
| Model string | first number in `Spec_FullDescriptor` (`950FII` → 950), × age | model numbers scale with machine size within a family |
| Relative usage | hours / median hours of the same configuration or base class | is this unit heavily used *for its type*? |
| Utilisation | ordinal Low/Medium/High → 0/1/2, unknown → −1 | |
| Cross features | 11 string crosses: config × age bucket / sale decade / sale year / scale; base class × age bucket / decade; function × age bucket; category × sale year / region / vendor / scale | let trees split on interactions directly |
| Group statistics | for config, base class, function, category, region, config × age bucket: mean/median/std hours, mean/std/min/max age, count | context about the machine's peer group |
| Within-config position | percentile rank and deviation from the config median for hours, age and sale date | a newer / less-used unit of the same model should cost more |
| Numeric interactions | label code × {age, log hours, sale year} for 4 categoricals | |

## 5. Encoding (`encoding.py`)

For every categorical column (original string columns, the two ID columns and the 11 crosses):

1. **Frequency encoding** — share of rows with that value.
2. **Out-of-fold smoothed target encoding** on log-price
   $$te(c) = \frac{n_c\,\bar y_c + m\,\bar y}{n_c + m},\quad m = 10$$
   For the training rows, statistics come from the other 4 folds of a 5-fold split, so a row never
   sees its own target. Test rows use statistics from all training rows. Categories unseen in the
   fitting data fall back to the global mean.
3. **Label encoding** of the raw value (useful to tree models for low-cardinality columns).

Remaining numeric NaNs (e.g. std of singleton groups) are filled with the train median and get a
`<col>_miss` indicator. The final matrix has **270 numeric features**.

## 6. Models (`models.py`)

All level-1 models use the same 5-fold `KFold(shuffle=True, random_state=42)` split, early stopping
(300 rounds) on the held-out fold, and average the 5 fold-models' test predictions.

| Model | Key hyper-parameters |
|---|---|
| LightGBM v1 | 255 leaves, lr 0.008, row 0.8 / col 0.7 subsampling, min_child 20, L1 0.05, L2 0.2 |
| LightGBM v2 | 511 leaves, lr 0.006, row 0.7 / col 0.6, min_child 30, L1 0.1, L2 0.5 — deliberately more diverse |
| XGBoost | depth 7, lr 0.008, hist, row/col 0.8, min_child_weight 5 |
| CatBoost | depth 8, lr 0.02, L2 3, Bernoulli bootstrap 0.8 |

Hyper-parameters were tuned by hand (Optuna and other AutoML tools were not allowed).

## 7. Ensembling (`ensemble.py`)

1. **Stacking:** a `BayesianRidge` meta-model is fit on the four OOF prediction columns (same 5 folds).
   Bayesian Ridge learns near-convex weights with built-in regularisation, which suits highly correlated
   base predictions.
2. **Blending:** non-negative weights over {LGB1, LGB2, XGB, CAT, Stack}, normalised to sum to 1, are
   found by Nelder–Mead minimisation of OOF RMSLE. Final weights: Stack 0.822, LGB2 0.073, LGB1 0.068,
   CAT 0.020, XGB 0.016.
3. **Price snapping:** sale prices are quoted in round amounts, so predictions can be rounded to the
   nearest $500 and clipped to the training range. This is applied only if it lowers OOF RMSLE (it did
   not in the final run: 0.1939 vs 0.1938), so raw predictions were submitted.

## 8. Limitations and next steps

- The stacker and blend weights are fit on the same OOF predictions they are evaluated on, so the
  0.1938 blend CV is slightly optimistic. A nested split would give a cleaner estimate.
- Random K-fold ignores time. The test set's date range was not analysed against train; a time-aware
  validation scheme would be worth checking.
- Several cross features are very high-cardinality (config × sale year); pruning by feature importance
  could cut training time substantially with little loss.
- Many `iter≈9999` readings for LightGBM v1 and XGBoost show early stopping never triggered — a larger
  learning rate or more rounds may still help.
