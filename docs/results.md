# Results

## Leaderboard

| | |
|---|---|
| Competition | Heavy Equipment Selling Price Prediction Challenge (IIT Madras BS, MLP Project, May 2026 term) |
| Period | 26 May 2026 – 29 July 2026 |
| Field | 2,999 entrants · 2,593 teams · 31,093 submissions |
| **My best score** | **0.18741 RMSLE** (notebook version 23) |
| **Final rank** | **85 / 2,593 (top 3.3 %)** |
| Winning score | 0.15890 |

The private leaderboard is computed over the same rows as the public one in this competition.

## Cross-validation (5-fold, notebook version 23)

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF |
|---|---|---|---|---|---|---|
| LightGBM v1 | 0.1927 | 0.1945 | 0.1949 | 0.1957 | 0.1941 | 0.1944 |
| LightGBM v2 | 0.1927 | 0.1949 | 0.1951 | 0.1962 | 0.1943 | 0.1947 |
| XGBoost | 0.1962 | 0.1979 | 0.1982 | 0.1988 | 0.1974 | 0.1977 |
| CatBoost | 0.1973 | 0.1998 | 0.1997 | 0.2005 | 0.1988 | 0.1992 |
| Bayesian Ridge stack | 0.1919 | 0.1942 | 0.1943 | 0.1953 | 0.1935 | 0.1938 |
| **Blend** | | | | | | **0.1938** |

Blend weights: LGB1 0.068 · LGB2 0.073 · XGB 0.016 · CAT 0.020 · Stack 0.822.
Snapping predictions to a $500 grid gave 0.1939 OOF, so raw predictions were used.

Test-set predictions: mean $41,215, median $34,763, range $7,445 – $141,983.

## Submission history (selected)

| Notebook version | Leaderboard RMSLE |
|---|---|
| 5 | 0.66928 |
| 6 | 0.66918 |
| 7 | 0.19400 |
| 9 | 0.19291 |
| 10 | 0.19357 |
| 11 | 0.19393 |
| 12 | 0.19321 |
| 13 | 0.19241 |
| 14 | 0.19223 |
| 15 | 0.19206 |
| 17 | 0.19375 |
| 21 | 0.18758 |
| 22 | 0.18757 |
| **23** | **0.18741** ← best, this repository |
| 25 | 0.18757 |
| 31 | 0.19206 |
