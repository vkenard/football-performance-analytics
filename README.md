# Football Performance Analytics

## Production ML Pipeline — Dixon-Coles × LightGBM Four-Model Ensemble

> **Audience:** Data science, analytics engineering, and ML hiring managers  
> **Focus:** End-to-end ML pipeline, feature engineering, production deployment, probability calibration, Monte Carlo simulation  
> **Data:** Premier League 2021–2026 — 1,920-match dataset, model tracking live through GW36 2025/26

This project demonstrates applied data science with a production-quality ML pipeline built from scratch. The core system runs weekly: ingesting multi-source data (FPL API, Understat, football-data.co.uk, The-Odds-API), engineering 100+ features, retraining four model variants, blending predictions, and producing calibrated probability outputs with edge detection against real bookie markets. Every component has been forensically audited — **61-match data gap identified and backfilled**, a **`prob_H`/`prob_A` column-swap bug caught via known-strength fixture validation**, a **target leakage bug identified through SHAP rank analysis**, and a live GW-by-GW post-match autopsy running through the current season.

The domain is football, but the engineering patterns transfer directly: production ingestion pipelines, time-aware model validation, leakage detection, Monte Carlo simulation, calibrated probability output, and automated drift monitoring.

---

## Contents

- [Tech Stack](#tech-stack)
- [Model Architecture](#model-architecture)
- [Key Achievements](#key-achievements)
- [Notebooks](#notebooks)
- [Generated Visualisations](#generated-visualisations)
- [Fixture Intelligence](#fixture-intelligence)
- [Live Simulation Tools](#live-simulation-tools)
- [Running the Notebooks](#running-the-notebooks)
- [Data Dictionary](#data-dictionary-sample_datasetcsv)
- [Case Studies](#case-study-forensic-recruitment--performance)
- [Limitations & Future Work](#limitations--future-work)

---

## Tech Stack

| Layer | Tools |
| --- | --- |
| **Language** | Python 3.11 |
| **ML / Statistics** | scikit-learn, LightGBM, Dixon-Coles (custom implementation), Monte Carlo simulation |
| **Data engineering** | Pandas, NumPy, multi-source ingestion (REST APIs, CSV, JSON), automated weekly pipeline |
| **Probability calibration** | Platt scaling, Brier Score, reliability curves, quantile binning |
| **Visualisation** | Matplotlib (custom dark-theme charting system), radar charts, heatmaps, distribution plots |
| **Validation** | Chronological 5-fold CV, forward-chaining time splits, rolling drift monitoring |
| **Tooling** | Jupyter, Git, automated data quality gates |

---

## Model Architecture

The system implements a **four-model equal-weight blend** combining Dixon-Coles structural parameters with LightGBM gradient-boosted contextual models. Blend weights are selected via grid search over chronological cross-validation.

| Component | Role |
| --- | --- |
| **V2B** | LightGBM on base feature set: Elo, xG differentials, form, rest |
| **V2D** | LightGBM with derived DC parameters (attack/defence log-deviations, home advantage) |
| **V3B** | V3 feature set adds possession EWMA, shot quality metrics, passing accuracy under pressure |
| **V3D** | Full V3 feature set with derived interaction terms |
| **Blend framework** | Grid search over (V2B, V2D, V3B, V3D) weight combinations via chronological 5-fold CV; current production: equal-weight 0.25 each |
| **Dixon-Coles** | Per-team attack/defence strength parameters derived from expected goals; expressed as log-deviation from league mean (0 = league average; −0.87 ≈ 0.42× average rate) |
| **Dynamic Draw Multiplier** | Adjusts draw probability based on xG gap — corrects systematic under-prediction in balanced fixtures |

DC parameters are refitted on a rolling monthly basis to track squad-level changes within the season.

---

## Key Achievements

- **Target leakage detected in SHAP rank #1 feature** — `xg_diff` had importance score 461 vs 258 for the next feature. Root cause: the feature was constructed from same-match xG rather than prior-match EWMA, causing systematic prediction distortion on fixtures where last-match xG was atypical. Detected through logical validation against known-strength fixtures; resolved by replacing with a 5-match EWMA differential

- **Forensic audit of a 61-match data gap** — identified missing Understat xG records, backfilled features to restore historical parity across training windows

- **Column-swap bug detected and corrected** — raw model output had `prob_H`/`prob_A` inverted; confirmed via known-strength fixtures (strong home side showing <20% home win probability), corrected in preprocessing pipeline

- **Built Monte Carlo simulation engines** projecting player-level goal output and FPL points across remaining fixtures for all 378 active Premier League players, incorporating DC lambda per fixture, form-weighted scoring rates, availability adjustments, and position-specific scoring rules

- **Developed DC-based fixture difficulty visualisation suite** producing four heatmaps (opponent DC rating, expected goals for, expected goals against, clean sheet probability) with team badge integration, blank/double gameweek handling, and continuous DC-derived difficulty ratings replacing categorical tiers

- **Brier Skill Score improvement over naive baseline** — ensemble adds measurable signal beyond predicting the historical average for every match

- **Monotonic probability deciles** — reliability table confirms the model discriminates meaningfully between high and low-probability outcomes

- **Rolling drift detection** — 20-match Brier score monitor flags regime changes, supporting proactive re-training decisions

- **Live deployment** — model is actively generating predictions through the current Premier League season (GW36 2025/26), not a retrospective exercise

---

## Notebooks

### 1. `forward_validation_demo.ipynb`

Demonstrates time-aware model evaluation using a strict chronological 70/30 split.
Covers: 3-way accuracy, binary accuracy, Brier score, Brier Skill Score, rolling drift chart, **feature importance**.

### 2. `calibration_analysis.ipynb`

Assesses probability reliability using a quantile-binned reliability curve.
Covers: calibration curve, Brier score vs historical baseline, decile reliability table and chart.

### 4. `mc_player_projection.ipynb` *(new — fully self-contained)*

Runnable demonstration of the vectorised Monte Carlo simulation engine.
**Requires only:** `numpy`, `pandas`, `matplotlib` — no proprietary data.

Covers: Dixon-Coles fixture-level Poisson lambdas, Bernoulli availability masks, Binomial thinning for individual goals/assists, position-specific FPL scoring dispatch table (no branching), P10/P50/P90 percentile summary, three-panel dark-themed visualisation. All 300,000 simulations (6 players × 5 fixtures × 10,000 iterations) execute in a single vectorised pass with no Python loops.

![Monte Carlo FPL Projection — expected points range, distribution overlay, fixture heatmap](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/mc_projection_output.png?v=20260503)

### 3. `gw26_gamestate_and_variance_autopsy.ipynb`

Forensic multi-market post-match analysis of Gameweek 26 using real GW26 prediction data.
Covers four analytical lenses:

- **Goal-Line Accuracy**: 80% on both 2.5 and 3.5 thresholds (8/10 completed fixtures) — volumetric signal held directionally on a week where the 1X2 market collapsed
- **Territorial Dominance**: Corner Territorial Pressure Index per team, validated against actuals with four-quadrant game-state classification
- **Match Volatility Heatmap**: Four-corner matrix (High/Low pre-match model-predicted xG × High/Low predicted corners) classifying each fixture by pre-match structural volatility
- **Macro Variance Autopsy**: 2×2 Model vs Home-Win Baseline breakdown isolating the **Alpha Zone** (WHU v MUN: model called the draw the naive baseline missed) from **Structural Chaos** (8/11 games unpredictable by any rule-based system)
- **Black Swan Example (WOL v ARS 2-2)**: Wolves (attack strength −0.87) scored 2 goals against Arsenal's elite defence. Model predicted 59.2% Arsenal win; result was a draw. Pre-match DC expectancy: 2.38 total goals; actual: 4 goals (+1.62 above model). Parameters were right — the outcome was a statistical outlier.

**Notebooks 1, 2, and 4 are fully self-contained** — clone the repo and run from top to bottom with no additional setup beyond `pip install -r requirements.txt`. Notebook 3 reads proprietary match-feature data not included in the public repo; all cells have pre-rendered outputs so the analysis is fully viewable without re-running.

---

## Generated Visualisations

| Chart | Preview |
| --- | --- |
| Forward Validation Split | ![Forward Validation Split chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/forward_validation_split.png?v=20260221) |
| Drift Monitoring | ![Drift Monitoring chart showing Brier score over time](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/drift_monitoring.png?v=20260221) |
| Calibration Curve | ![Calibration Curve chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/calibration_curve.png?v=20260221) |
| Decile Reliability | ![Decile Reliability table chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/decile_reliability.png?v=20260221) |
| Feature Importance | ![Feature Importance bar chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/feature_importance.png?v=20260221) |
| GW26 Goal Expectancy | ![GW26 Goal Expectancy chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_goal_expectancy.png?v=20260221) |
| GW26 Territorial Dominance | ![GW26 Territorial Dominance chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_territorial_dominance.png?v=20260221) |
| GW26 Volatility Heatmap | ![GW26 Volatility Heatmap matrix](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_volatility_heatmap.png?v=20260221) |
| GW26 Variance Autopsy | ![GW26 Variance Autopsy breakdown chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_variance_autopsy.png?v=20260221) |
| MC Projection — Expected Points Range | ![Monte Carlo FPL Projection — expected points with P10–P90 band, distribution overlay, fixture heatmap](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/mc_projection_output.png?v=20260503) |
| James Garner Profile | ![James Garner player profile radar chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_performance_radar.png?v=20250302) |
| Garner vs Wharton vs Tielemans | ![Comparative radar: Garner vs Wharton vs Tielemans](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_cm_comparison.png?v=20250302) |
| Garner Rolling Form Arc | ![Garner rolling 5-GW form arc across 4 metrics](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_rolling_arc.png?v=20250302) |
| Everton Squad Player Radars | ![Everton squad player radar grid — all 15 qualifying players, percentile vs positional PL peers](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/everton_player_radars.png?v=20260221) |

---

## Fixture Intelligence

DC-derived fixture difficulty suite covering GW31–38 (the final eight gameweeks of 2025/26). Each chart uses continuous DC-derived ratings rather than categorical difficulty tiers, with team badge integration and blank/double gameweek handling.

| Chart | Preview |
| --- | --- |
| Opponent DC Rating | ![Fixture difficulty by opponent Dixon-Coles attack/defence rating — GW31-38](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/fixture_difficulty_ratings.png?v=20260503) |
| Expected Goals For (xGF) | ![Expected goals for per fixture, DC-derived, GW31-38](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/fixture_difficulty_xgf.png?v=20260503) |
| Expected Goals Against (xGC) | ![Expected goals against per fixture, DC-derived, GW31-38](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/fixture_difficulty_xgc.png?v=20260503) |
| Clean Sheet Probability | ![Clean sheet probability per fixture, DC-derived, GW31-38](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/fixture_difficulty_cs.png?v=20260503) |

*Generated from rolling DC lambda estimates using Poisson probability mass functions. Home advantage is parameterised into the DC system rather than applied as a flat correction. Blanks shown as grey cells; double gameweeks flagged with a border.*

---

## Live Simulation Tools

The production system includes Monte Carlo simulation engines that run at prediction time for all active players.

**Player Projection Engine**
- Simulates goal output and FPL points across remaining fixtures for all 378 active Premier League players (10,000 iterations per player per fixture)
- Inputs per simulation: DC lambda for the fixture, player's form-weighted per-90 scoring rate (5-match EWMA), availability probability (injury/suspension), position-specific FPL scoring rules (clean sheet bonuses, save points, etc.)
- Outputs: expected points, P(top-5 finish), P(blank), percentile confidence intervals — enabling probabilistic player ranking rather than point-estimate ranking

**Why this matters technically:** The engine vectorises fixture-level Poisson draws across 378 players × N remaining fixtures × 10,000 iterations without looping in Python. Player availability adjustments are applied as Bernoulli masks over the simulation matrix, and position-specific scoring is handled by a dispatch table rather than branching logic.

**See it in action:** `mc_player_projection.ipynb` is a fully runnable, self-contained demonstration of this engine — synthetic but calibrated to realistic Premier League ranges, no proprietary data required.

---

## Running the Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

Open any notebook and select **Kernel -> Restart & Run All**.

Notebooks 1, 2, and 4 (`forward_validation_demo`, `calibration_analysis`, `mc_player_projection`) are fully self-contained — no additional data required. Notebook 3 (`gw26_gamestate_and_variance_autopsy`) has pre-rendered outputs and is viewable without re-running.

---

## File Structure

```text
football-performance-analytics/
+-- README.md
+-- requirements.txt
+-- forward_validation_demo.ipynb
+-- calibration_analysis.ipynb
+-- gw26_gamestate_and_variance_autopsy.ipynb
+-- mc_player_projection.ipynb           ← vectorised Monte Carlo simulation engine demo
+-- sample_dataset.csv
+-- scripts/
    +-- player_radar_profile.py
    +-- player_form_arc.py
+-- assets/
    +-- forward_validation_split.png
    +-- drift_monitoring.png
    +-- calibration_curve.png
    +-- decile_reliability.png
    +-- feature_importance.png
    +-- mc_projection_output.png
    +-- fixture_difficulty_ratings.png
    +-- fixture_difficulty_xgf.png
    +-- fixture_difficulty_xgc.png
    +-- fixture_difficulty_cs.png
    +-- gw26_goal_expectancy.png
    +-- gw26_territorial_dominance.png
    +-- gw26_volatility_heatmap.png
    +-- gw26_variance_autopsy.png
    +-- gw26_everton_finishing_variance.png
    +-- gw26_academy_development_monitor.png
    +-- everton_process_vs_results.png
    +-- xg_inefficiency_scatter.png
    +-- garner_performance_radar.png
    +-- garner_cm_comparison.png
    +-- garner_rolling_arc.png
    +-- everton_player_radars.png
    +-- [+ individual player radars]
```

---

## Data Dictionary (`sample_dataset.csv`)

| Column | Description |
| --- | --- |
| `match_date` | ISO date of the fixture |
| `home_team` / `away_team` | Club names |
| `season` | Competition season |
| `actual_result` | Observed outcome: H / D / A |
| `predicted_result` | Model's top predicted outcome: H / D / A |
| `correct` | True / False — whether prediction matched actual result |
| `prob_H` / `prob_D` / `prob_A` | Full 3-way probability output (sums to 1.0) |
| `elo_diff` | Elo rating gap (home minus away) |
| `home_xg` / `away_xg` | Expected goals (Understat) |
| `dc_home_attack` / `dc_away_defence` | Dixon-Coles per-team strength parameters |
| `form_home_5` / `form_away_5` | Points from last 5 matches |
| `rest_days_home` / `rest_days_away` | Days since last fixture |

---

## Case Study: Forensic Recruitment & Performance

### Visualising Structural Underperformance

![Everton Process vs Results dual-axis rolling 5-game comparison chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/everton_process_vs_results.png?v=20260221)

*Dual-axis rolling 5-game comparison: actual points per game (blue, left axis) vs actual post-match Understat xG per game (orange, right axis). Shaded windows highlight periods where structural quality (actual xG) exceeded the points return — the analytical case for maintaining confidence in a squad despite a short-term points slump.*

### xG Efficiency Profile: Identifying Structural Under- and Over-performance

![xG Inefficiency Scatter plot showing team xG vs points per game](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/xg_inefficiency_scatter.png?v=20260221)

*Each point is a team's mean xG per game (x) vs mean points per game (y), coloured by how far above or below the xG-to-points regression line they sit. Teams in the **bottom-right quadrant** (High xG, Low PPG) are generating quality chances but failing to convert them to results — the structural "underperforming" profile. Teams in the **top-left quadrant** (Low xG, High PPG) are over-converting — riding form or finishing luck that is statistically unlikely to persist.*

---

## Case Study: Everton 2025/26 — Process vs Results

### GW26 Finishing Variance Context (Everton)

![Everton Finishing Variance Deep Dive showing pre-match vs in-game xG comparison](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_everton_finishing_variance.png?v=20260221)

*Real GW26 context plot: pre-match DC goal expectancy (EVE 1.06 vs BOU 0.98, model-predicted) versus actual in-game Understat xG (EVE 2.94 vs BOU 1.34, post-match), with Everton's DC attack trend over recent matches.*

**EVE 1-2 BOU (GW26) — the unlucky loss in numbers:**

The model had Everton as **46.1% favourites** going in (BOU just 26.0%). Everton generated **2.94 actual xG** against Bournemouth's 1.34 — dominated the underlying process by **+1.60 xG** — yet lost 1-2.

| Metric | Everton | Bournemouth |
| --- | --- | --- |
| Pre-match model predicted xG (DC λ) | 1.06 | 0.98 |
| Actual in-game Understat xG | **2.94** | 1.34 |
| Goals scored | 1 | **2** |
| Conversion rate (goals ÷ actual xG) | **34%** | **149%** |
| Model win probability (pre-match) | **46.1%** | 26.0% |

Bournemouth converted at 149% of their xG — extreme over-conversion. Everton at 34% — extreme under-conversion. This is the scenario where results-based analysis misleads. The underlying data shows the structural picture is the opposite of the scoreline.

*Context: Jake O'Brien (Everton) was dismissed at 69' while trailing 1-2. Everton generated the majority of their xG while chasing the game — a game state that naturally inflates attacking chances. The pre-match model probability and first-half metrics are unaffected by the red card.*

---

## Player Spotlight: James Garner (Everton)

![James Garner player profile radar showing percentile rankings vs PL midfielders](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_performance_radar.png?v=20250401)

*Percentile radar vs 121 PL midfielders with ≥900 Premier League minutes in 2025/26 (GW1–26). All metrics per 90.*

| Metric | Per 90 | Percentile (vs 121 PL midfielders) |
| --- | --- | --- |
| Defensive Contribution | 12.4 | **95th** |
| Tackles | 3.0 | **93rd** |
| Ball Recoveries | 5.2 | **79th** |
| Chance Creation (Creativity) | 22.8 | 66th |
| Overall Influence | 23.1 | **81st** |
| xG Involvements | 0.23 | 42nd |

### Comparison: Garner vs Wharton vs Tielemans

![Comparative radar Garner vs Wharton vs Tielemans all PL CM percentiles](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_cm_comparison.png?v=20250302)

*All metrics per 90, percentile vs 249 PL outfield starters (≥900 min, GW1–26).*

| Metric | Garner | Wharton | Tielemans |
| --- | --- | --- | --- |
| Defensive Contribution | **96th** | 82nd | 65th |
| Tackles | **95th** | 86th | 90th |
| Ball Recoveries | 86th | **94th** | 75th |
| Chance Creation | 81st | 80th | **91st** |
| Overall Influence | 76th | 18th | 67th |
| xG Involvements | 62nd | 69th | 63rd |

Garner uniquely combines defensive dominance with above-average creative and match-influence numbers — a rarer profile than any single metric suggests.

### Season Form Arc

![Garner rolling 5-GW form arc across 4 key metrics](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_rolling_arc.png?v=20250302)

*Rolling 5-GW average across 4 per-90 metrics. GW19 annotated (goal + assist). All 26 GWs played; 2,333 minutes.*

---

## Methodology Transfer: Development Tracking & Drift Detection

![Academy Development Monitor](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_academy_development_monitor.png?v=20260221)

*Rolling Z-score drift monitor applied to an emerging player's per-90 creativity metric versus a GW cohort baseline (Mateus Mané, Wolves, born 2007). Breakout threshold crossed by GW24 — statistically confirmed before the standout match contribution in GW26.*

The pattern is domain-agnostic and applies directly to any KPI time series:

- **Development tracking** — rolling drift monitor applied to any KPI time series detects improvement or regression before aggregate metrics catch up
- **Isolating process from luck** — Brier Score measures probability quality, not win rate; a player or team can be improving without the results to show for it
- **Longitudinal trend analysis** — chronological split discipline prevents retrospective overfitting
- **Regime change detection** — drift detection flags windows where performance patterns shift, prompting targeted review

---

## Limitations & Future Work

- **DC baseline uses goals and xG, not action-level data:** incorporating Expected Threat (xT) grid values or SPADL action sequences would improve structural team-strength estimates for press-heavy sides
- **No player-tracking or GPS data:** squad-level features use statistical proxies rather than physical output (high-intensity runs, pressing distance covered)
- **Single-league scope:** trained on Premier League only (2021–26); cross-league analysis requires division-level strength normalisation
- **Player radar axes limited to FPL-available metrics:** progressive passes, pressures, and PPDA require Opta or StatsBomb licensing for a production recruitment system

---

*Portfolio work. All model weights, proprietary feature engineering, and live prediction pipeline are withheld. Methodology and diagnostic outputs are shared for analytical review.*
