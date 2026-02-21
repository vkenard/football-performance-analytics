# Football Performance Analytics

## Hybrid Ensemble Model -- Dixon-Coles + LightGBM

> **Audience:** Academy analysts, performance staff, coaching leads  
> **Focus:** Probability calibration, longitudinal performance tracking, development diagnostics  
> **Data:** Premier League 2023-2026 (350-match curated sample, model tracking live through GW26)

This project demonstrates applied sports data science with direct relevance to academy performance analysis. The core proposition: **process quality (xG, DC parameters, expected outcomes) predicts future performance before results do** -- exactly the signal an academy analyst needs to identify development trajectories in young players and squads before the points table reflects them. Every technique here -- drift detection, probability calibration, match-state classification -- has a direct academy equivalent.

Built and maintained by a self-taught analyst. Forensic auditing over automation: this model has had a **61-match data gap identified and backfilled**, a **`prob_H`/`prob_A` column-swap bug detected via known-strength fixtures**, and a live GW-by-GW post-match autopsy running through the current season.

---

## Model Architecture

This project implements a **Dynamic Blend ensemble** combining two complementary modelling philosophies:

| Component | Role |
| --- | --- |
| **Dixon-Coles** | Per-team attack/defence strength parameters derived from expected goals. Captures structural team quality. |
| **LightGBM** | Gradient-boosted classifier incorporating form, rest, xG differentials, and Elo. Captures contextual variation. |
| **Dynamic Draw Multiplier** | Adjusts draw probability based on xG gap between teams -- avoids systematic under-prediction in balanced matches. |

The blend weight is tuned via chronological cross-validation to prevent information leakage from future matches.

---

## Key Achievements

- **Forensic audit of a 61-match data gap** -- identified missing Understat xG records, backfilled features to restore historical parity across training windows
- **Column-swap bug detected and corrected** -- raw model output had `prob_H`/`prob_A` inverted; confirmed via known-strength fixtures, corrected in preprocessing pipeline
- **Brier Skill Score improvement over naive baseline** -- ensemble adds measurable signal beyond predicting the historical average for every match
- **Monotonic probability deciles** -- reliability table confirms the model discriminates meaningfully between high and low-probability outcomes
- **Rolling drift detection** -- 20-match Brier score monitor flags regime changes, supporting proactive re-training decisions
- **Live deployment** -- model is actively generating predictions through the current Premier League season (GW26 2025/26), not a retrospective exercise

---

## Notebooks

### 1. `forward_validation_demo.ipynb`

Demonstrates time-aware model evaluation using a strict chronological 70/30 split.
Covers: 3-way accuracy, binary accuracy, Brier score, Brier Skill Score, rolling drift chart, **feature importance**.

### 2. `calibration_analysis.ipynb`

Assesses probability reliability using a quantile-binned reliability curve.
Covers: calibration curve, Brier score vs historical baseline, decile reliability table and chart.

### 3. `gw26_gamestate_and_variance_autopsy.ipynb`

Forensic multi-market post-match analysis of Gameweek 26 using real GW26 prediction data.
Covers four analytical lenses:

- **Goal-Line Accuracy**: 73% on both 2.5 and 3.5 thresholds (8/11 completed fixtures) -- volumetric signal held directionally on a week where the 1X2 market collapsed
- **Territorial Dominance**: Corner Territorial Pressure Index per team, validated against actuals with four-quadrant game-state classification
- **Match Volatility Heatmap**: Four-corner matrix (High/Low xG × High/Low corners) classifying each fixture by pre-match structural volatility
- **Macro Variance Autopsy**: 2×2 Model vs Home-Win Baseline breakdown isolating the **Alpha Zone** (WHU v MUN: model called the draw the naive baseline missed) from **Structural Chaos** (8/11 games unpredictable by any rule-based system), with **EVE v BOU explicitly classified as Finishing Variance**. Discrimination gap -1.6pp remains near-zero calibration separation.
- **Black Swan Example (WOL v ARS 2-2)**: Textbook extreme variance event where Wolves (attack strength -0.87, very weak) scored 2 goals against Arsenal's elite defense (defensive strength -0.65). Model predicted 59.2% Arsenal win with 2.38 xG total; actual result was 4 goals and a draw (+1.62 goal variance). This validates that the model correctly identified the structural quality gap -- the parameters were right; the outcome was a statistical outlier.

**Notebooks 1 and 2 are fully self-contained** -- clone the repo and run from top to bottom with no additional setup. Notebook 3 (`gw26_gamestate_and_variance_autopsy.ipynb`) reads proprietary match-feature data not included in the public repo; all cells have pre-rendered outputs so the analysis is fully viewable without re-running.

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

---

## Running the Notebooks

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook
```

Open either notebook and select **Kernel -> Restart & Run All**.

---

## File Structure

```text
football-performance-analytics/
+-- README.md
+-- forward_validation_demo.ipynb
+-- calibration_analysis.ipynb
+-- gw26_gamestate_and_variance_autopsy.ipynb
+-- sample_dataset.csv
+-- assets/
    +-- forward_validation_split.png
    +-- drift_monitoring.png
    +-- calibration_curve.png
    +-- decile_reliability.png
    +-- feature_importance.png
    +-- gw26_goal_expectancy.png
    +-- gw26_territorial_dominance.png
    +-- gw26_volatility_heatmap.png
    +-- gw26_variance_autopsy.png
    +-- gw26_everton_finishing_variance.png
    +-- gw26_academy_development_monitor.png
    +-- everton_process_vs_results.png
    +-- xg_inefficiency_scatter.png
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
| `correct` | True / False -- whether prediction matched actual result |
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

*Dual-axis rolling 5-game comparison: actual points per game (blue, left axis) vs Goal Expectancy per game (orange, right axis). Shaded windows highlight periods where structural quality (xG) exceeded results -- the analytical case for maintaining confidence in a squad despite a short-term points slump.*

### xG Efficiency Profile: Identifying Structural Under- and Over-performance

![xG Inefficiency Scatter plot showing team xG vs points per game](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/xg_inefficiency_scatter.png?v=20260221)

*Each point is a team's mean xG per game (x) vs mean points per game (y), coloured by how far above or below the xG-to-points regression line they sit. Teams in the **bottom-right quadrant** (High xG, Low PPG) are generating quality chances but failing to convert them to results -- the structural "underperforming" profile. Teams in the **top-left quadrant** (Low xG, High PPG) are over-converting -- riding form or finishing luck that is statistically unlikely to persist. The four-corner labels frame each zone for direct scouting and squad-planning language: "Structural Candidate" (bottom-right) vs "Fortunate / Regression Risk" (top-left). Applied to academy players: replace team xG with player progressive passes/shot-creating actions and the same quadrant logic identifies who is producing process indicators before results reward them.*

---

## Case Study: Everton 2024/25 -- Process vs Results

### GW26 Finishing Variance Context (Everton)

![Everton Finishing Variance Deep Dive showing pre-match vs in-game xG comparison](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_everton_finishing_variance.png?v=20260221)

*Real GW26 context plot: pre-match model xG (1.06 vs 0.98) versus actual in-game xG (2.94 vs 1.34), with Everton DC attack trend over recent matches. The match is classified as finishing variance (process right, conversion wrong), not structural attack collapse.*

In GW27-30 of the 2024/25 season, the model identified a structural disconnect between Everton's underlying metrics and their points return:

| Fixture | xG Diff (Everton minus opponent, actual) | DC Home Attack (pre-match, home team) | Result | Model Prediction |
| --- | --- | --- | --- | --- |
| Everton vs Liverpool | +0.53 | 0.886 | Draw | Draw (correct) |
| Crystal Palace vs Everton | -0.23 (Everton rel.) | -- | Everton won | Home win (wrong) |
| Everton vs Man United | +0.17 | 1.333 | Draw | Home win (close) |

Notes: xG Diff is the observed in-game xG differential from the match data. DC Home Attack is the pre-match Dixon-Coles attack parameter for the home team (not a probability), so values around ~0.8 to ~1.3 are plausible depending on opponent strength.

Key findings:

- Against Liverpool, Everton generated **+0.53 xG differential at home** -- outperforming one of the division's strongest sides in expected chances. The model captured this (42% draw probability) and was correct.
- At Crystal Palace, Everton generated **1.00 away xG vs Crystal Palace's 0.77 home xG** -- a structural mismatch invisible in the match result alone. Everton won.
- Across the window, Everton's Dixon-Coles attack parameters averaged 1.08 -- near league average, but with a consistent positive xG differential suggesting the model was slightly underrating their attacking output.

**The analytical value:** A team can show process improvement (consistent positive xG differentials, strong DC parameters) before results catch up. This is precisely the kind of signal academy analysts need -- identifying development in advance of points, not retroactively.

---

## Academy Application

![Academy Development Monitor](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_academy_development_monitor.png?v=20260221)

*Real-player development monitor built from FPL 2025/26 gameweek player files (Mateus Mané). The chart uses chances-created-per-90 and rolling Z-score versus GW cohort baseline to separate development signal from short-term noise.*

The methodology here is directly transferable to academy performance analysis:

- **Development tracking** -- replace match outcome with development KPIs (sprint load, pressing intensity, passing accuracy under pressure) and apply the same rolling drift monitor to detect improvement or regression across a season
- **Isolating process from luck** -- Brier Score measures *probability quality*, not just win rate. A player or team can be improving without the wins to show for it; this framework captures that signal
- **Longitudinal trend analysis** -- the chronological split discipline prevents retrospective overfitting, ensuring insights reflect genuine development rather than noise
- **Coaching interventions** -- drift detection flags windows where performance patterns shift, prompting targeted review rather than waiting for results to deteriorate
- **Recruitment and scouting** -- the xG differential and Dixon-Coles parameters identify players and teams that are structurally outperforming their actual results. A young forward whose team has a consistently negative xG differential (poor service) but who personally contributes above-average shot quality is structurally undervalued by results-based scouting. This framework separates individual contribution from collective outcome -- directly applicable to identifying underpriced talent at academy or senior level.

---

*Portfolio work. All model weights and proprietary feature engineering are withheld. Methodology and diagnostic outputs are shared for analytical review.*
