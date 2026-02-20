# Football Performance Analytics
### Hybrid Ensemble Model -- Dixon-Coles + LightGBM

> **Audience:** Academy analysts, performance staff, coaching leads
> **Focus:** Probability calibration, longitudinal performance tracking, development diagnostics
> **Data:** Premier League 2023-2026 (350-match curated sample, model tracking live through GW25)

---

## Model Architecture

This project implements a **Dynamic Blend ensemble** combining two complementary modelling philosophies:

| Component | Role |
|---|---|
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
- **Live deployment** -- model is actively generating predictions through the current Premier League season (GW25 2025/26), not a retrospective exercise

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
Covers: Goal-Line Accuracy (80% on both 2.5 and 3.5 thresholds, 10 completed fixtures), Territorial Dominance (corner prediction), Match Volatility Heatmap, and 1X2 Macro Variance Autopsy -- isolating an anomalous upset week from model calibration quality (+0.5pp discrimination gap).

**Notebooks 1 and 2 are fully self-contained** -- clone the repo and run from top to bottom with no additional setup. Notebook 3 (`gw26_gamestate_and_variance_autopsy.ipynb`) reads proprietary match-feature data not included in the public repo; all cells have pre-rendered outputs so the analysis is fully viewable without re-running.

---

## Generated Visualisations

| Chart | Preview |
|---|---|
| Forward Validation Split | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/forward_validation_split.png) |
| Drift Monitoring | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/drift_monitoring.png) |
| Calibration Curve | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/calibration_curve.png) |
| Decile Reliability | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/decile_reliability.png) |
| Feature Importance | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/feature_importance.png) |
| GW26 Goal Expectancy | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_goal_expectancy.png) |
| GW26 Territorial Dominance | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_territorial_dominance.png) |
| GW26 Volatility Heatmap | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_volatility_heatmap.png) |
| GW26 Variance Autopsy | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_variance_autopsy.png) |
| Everton Process vs Results | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/everton_process_vs_results.png) |
| xG Inefficiency Scatter | ![](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/xg_inefficiency_scatter.png) |

---

## Running the Notebooks

```bash
pip install pandas numpy matplotlib scikit-learn
jupyter notebook
```

Open either notebook and select **Kernel -> Restart & Run All**.

---

## File Structure

```
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
    +-- everton_process_vs_results.png
    +-- xg_inefficiency_scatter.png
```

---

## Data Dictionary (`sample_dataset.csv`)

| Column | Description |
|---|---|
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

![Everton Process vs Results](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/everton_process_vs_results.png)

*Dual-axis rolling 5-game comparison: actual points per game (blue, left axis) vs Goal Expectancy per game (orange, right axis). Shaded windows highlight periods where structural quality (xG) exceeded results -- the analytical case for maintaining confidence in a squad despite a short-term points slump.*

### Identifying the Structurally Undervalued Forward

![xG Inefficiency Scatter](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/xg_inefficiency_scatter.png)

*Each point represents a player's individual xG contribution (% of team total) plotted against their team's mean xG output. Players in the bottom-left quadrant (low team output, high personal contribution) are generating high-quality chances from low-service contexts -- the structural profile of an undervalued forward masked by collective underperformance. Highlighted points (Everton blue) represent the primary scouting target archetype.*

---

## Case Study: Everton 2024/25 -- Process vs Results

In GW27-30 of the 2024/25 season, the model identified a structural disconnect between Everton's underlying metrics and their points return:

| Fixture | xG Diff | DC Home Attack | Result | Model Prediction |
|---|---|---|---|---|
| Everton vs Liverpool | +0.53 | 0.886 | Draw | Draw (correct) |
| Crystal Palace vs Everton | -0.23 (Everton rel.) | -- | Everton won | Home win (wrong) |
| Everton vs Man United | +0.17 | 1.333 | Draw | Home win (close) |

Key findings:
- Against Liverpool, Everton generated **+0.53 xG differential at home** -- outperforming one of the division's strongest sides in expected chances. The model captured this (42% draw probability) and was correct.
- At Crystal Palace, Everton generated **1.00 away xG vs Crystal Palace's 0.77 home xG** -- a structural mismatch invisible in the match result alone. Everton won.
- Across the window, Everton's Dixon-Coles attack parameters averaged 1.08 -- near league average, but with a consistent positive xG differential suggesting the model was slightly underrating their attacking output.

**The analytical value:** A team can show process improvement (consistent positive xG differentials, strong DC parameters) before results catch up. This is precisely the kind of signal academy analysts need -- identifying development in advance of points, not retroactively.

---

## Academy Application

The methodology here is directly transferable to academy performance analysis:

- **Development tracking** -- replace match outcome with development KPIs (sprint load, pressing intensity, passing accuracy under pressure) and apply the same rolling drift monitor to detect improvement or regression across a season
- **Isolating process from luck** -- Brier Score measures *probability quality*, not just win rate. A player or team can be improving without the wins to show for it; this framework captures that signal
- **Longitudinal trend analysis** -- the chronological split discipline prevents retrospective overfitting, ensuring insights reflect genuine development rather than noise
- **Coaching interventions** -- drift detection flags windows where performance patterns shift, prompting targeted review rather than waiting for results to deteriorate
- **Recruitment and scouting** -- the xG differential and Dixon-Coles parameters identify players and teams that are structurally outperforming their actual results. A young forward whose team has a consistently negative xG differential (poor service) but who personally contributes above-average shot quality is structurally undervalued by results-based scouting. This framework separates individual contribution from collective outcome -- directly applicable to identifying underpriced talent at academy or senior level.

---

*Portfolio work. All model weights and proprietary feature engineering are withheld. Methodology and diagnostic outputs are shared for analytical review.*