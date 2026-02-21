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
| **Dixon-Coles** | Per-team attack/defence strength parameters derived from expected goals. Captures structural team quality. DC attack/defence parameters are expressed as log-deviation from league mean (0 = league average; negative = below average, e.g. −0.87 ≈ 0.42× average rate). |
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

- **Goal-Line Accuracy**: 80% on both 2.5 and 3.5 thresholds (8/10 completed fixtures) -- volumetric signal held directionally on a week where the 1X2 market collapsed
- **Territorial Dominance**: Corner Territorial Pressure Index per team, validated against actuals with four-quadrant game-state classification
- **Match Volatility Heatmap**: Four-corner matrix (High/Low **pre-match model-predicted xG** × High/Low predicted corners) classifying each fixture by pre-match structural volatility
- **Macro Variance Autopsy**: 2×2 Model vs Home-Win Baseline breakdown isolating the **Alpha Zone** (WHU v MUN: model called the draw the naive baseline missed) from **Structural Chaos** (8/11 games unpredictable by any rule-based system), with **EVE v BOU explicitly classified as Finishing Variance**. Discrimination gap +0.5pp remains near-zero calibration separation.
- **Black Swan Example (WOL v ARS 2-2)**: *(Note: this fixture was postponed and excluded from the original GW26 autopsy written Feb 16 — no actuals were available at that point. The match completed Feb 18, 2026 and is incorporated here as the full GW26 round is now complete.)* Textbook extreme variance event where Wolves (attack strength −0.87, very weak) scored 2 goals (including a Calafiori own goal, 90+4') against Arsenal's elite defence (defensive strength −0.65). Model predicted 59.2% Arsenal win with a pre-match DC goal expectancy of 2.38 total (λ_home + λ_away); actual result was 4 goals and a draw (+1.62 goals above model expectancy). This validates that the model correctly identified the structural quality gap -- the parameters were right; the outcome was a statistical outlier.

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
| James Garner Profile | ![James Garner player profile radar chart](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_performance_radar.png?v=20250302) |
| Garner vs Wharton vs Tielemans | ![Comparative radar: Garner vs Wharton vs Tielemans](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_cm_comparison.png?v=20250302) |
| Garner Rolling Form Arc | ![Garner rolling 5-GW form arc across 4 metrics](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_rolling_arc.png?v=20250302) |

---

## Running the Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

Open either notebook and select **Kernel -> Restart & Run All**.

---

## File Structure

```text
football-performance-analytics/
+-- README.md
+-- requirements.txt
+-- forward_validation_demo.ipynb
+-- calibration_analysis.ipynb
+-- gw26_gamestate_and_variance_autopsy.ipynb
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

*Dual-axis rolling 5-game comparison: actual points per game (blue, left axis) vs actual post-match Understat xG per game (orange, right axis). Shaded windows highlight periods where structural quality (actual xG) exceeded the points return -- the analytical case for maintaining confidence in a squad despite a short-term points slump.*

### xG Efficiency Profile: Identifying Structural Under- and Over-performance

![xG Inefficiency Scatter plot showing team xG vs points per game](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/xg_inefficiency_scatter.png?v=20260221)

*Each point is a team's mean xG per game (x) vs mean points per game (y), coloured by how far above or below the xG-to-points regression line they sit. Teams in the **bottom-right quadrant** (High xG, Low PPG) are generating quality chances but failing to convert them to results -- the structural "underperforming" profile. Teams in the **top-left quadrant** (Low xG, High PPG) are over-converting -- riding form or finishing luck that is statistically unlikely to persist. The four-corner labels frame each zone for direct scouting and squad-planning language: "Structural Candidate" (bottom-right) vs "Fortunate / Regression Risk" (top-left). Applied to academy players: replace team xG with player progressive passes/shot-creating actions and the same quadrant logic identifies who is producing process indicators before results reward them.*

---

## Case Study: Everton 2025/26 -- Process vs Results

### GW26 Finishing Variance Context (Everton)

![Everton Finishing Variance Deep Dive showing pre-match vs in-game xG comparison](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_everton_finishing_variance.png?v=20260221)

*Real GW26 context plot: pre-match DC goal expectancy (EVE 1.06 vs BOU 0.98, model-predicted) versus actual in-game Understat xG (EVE 2.94 vs BOU 1.34, post-match), with Everton's DC attack trend over recent matches. The match is classified as finishing variance -- process was right, conversion failed.*

**EVE 1-2 BOU (GW26) -- the unlucky loss in numbers:**

The model had Everton as **46.1% favourites** going in (BOU just 26.0%). In-game, Everton massively exceeded their pre-match expectation -- generating **2.94 actual xG** against Bournemouth's 1.34. Everton dominated the underlying process by **+1.60 xG**, yet lost 1-2.

| Metric | Everton | Bournemouth |
| --- | --- | --- |
| Pre-match model predicted xG (DC λ) | 1.06 | 0.98 |
| Actual in-game Understat xG | **2.94** | 1.34 |
| Goals scored | 1 | **2** |
| Conversion rate (goals ÷ actual xG) | **34%** | **149%** |
| Model win probability (pre-match) | **46.1%** | 26.0% |

Bournemouth converted at 149% of their xG -- extreme over-conversion. Everton converted at 34% -- extreme under-conversion. The structural quality gap, both pre-match (model favourite) and in-game (xG dominated), pointed entirely to Everton. Across the full 90 minutes the structural case for Everton is clear. One complicating factor is noted below.

This is precisely the scenario where results-based analysis misleads: a 1-2 loss looks like an Everton performance problem. The underlying data shows the opposite.

*Context: Jake O'Brien (Everton) was dismissed at 69' while the score stood at 1-2. Everton generated the majority of their 2.94 xG while reduced to ten men and chasing the game -- a game state that naturally inflates attacking chances as the opposition sits deeper. This is acknowledged as a complicating factor in the pure finishing-variance classification. The pre-match model probability (46.1% Everton) and first-half process metrics are unaffected by the red card, but the post-match xG figure should be read alongside the game state, not in isolation.*

In GW26-28 of the 2024/25 season, the model identified a structural disconnect between Everton's underlying metrics and their points return:

| Fixture | Actual xG Diff (post-match, Everton − opp.) | DC λ home (pre-match exp. goals) | Result | Model Prediction |
| --- | --- | --- | --- | --- |
| Everton vs Liverpool | +0.40 (EVE 1.0 vs LIV 0.6) | 0.654 (Everton home λ) | Draw | Draw (correct) |
| Crystal Palace vs Everton | −0.70 (EVE 0.9 vs PAL 1.6) | 1.096 (Palace home λ) | Everton won | Home win (wrong) |
| Everton vs Man United | +1.20 (EVE 1.6 vs MUN 0.4) | 0.851 (Everton home λ) | Draw | Home win (narrow miss) |

Notes: **xG Diff** is actual post-match in-game xG from Understat data (not model-predicted xG). Positive = Everton generated more expected chances. **DC λ** (lambda) is the Dixon-Coles pre-match expected goals for the home team -- a composite of both teams' attack/defence parameters and home advantage. Values below 1.0 indicate a below-average attacking expectation for the home side.

Key findings:

- Against Liverpool, Everton generated **+0.40 xG differential at home** (1.0 vs 0.6) -- competitive against one of the division's strongest sides. The model called the draw correctly (45.9% Liverpool win, 32.9% draw, 21.2% Everton win -- draw was the modal wrong-favourite outcome). A case where xG metrics and model calibration both aligned with the result.
- At Crystal Palace, **Crystal Palace generated 1.60 home xG vs Everton's 0.90** -- structural metrics favoured Palace, and the model predicted a Palace home win (42.2%). Everton converted and won. Classic over-conversion: process pointed one way, finishing variance decided the result.
- Across the two Everton home fixtures in this window, Everton's DC expected goals (λ) averaged 0.75 -- below league average, reflecting a squad under structural pressure. Yet in the Man United match their actual xG reached 1.60 against United's 0.40 (source: [Understat](https://understat.com/match/EVE/ManUnited/2024-25) -- extreme over-conversion by United, 500% of xG, consistent with finishing variance rather than structural quality), demonstrating the gap between season-level parameters and in-game execution.

**The analytical value:** A team can show process improvement (consistent positive xG differentials, strong DC parameters) before results catch up. This is precisely the kind of signal academy analysts need -- identifying development in advance of points, not retroactively.

---

## Player Spotlight: James Garner (Everton)

![James Garner player profile radar showing percentile rankings vs PL midfielders](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_performance_radar.png?v=20250401)

*Percentile radar vs 121 PL midfielders with ≥900 Premier League minutes in 2025/26 (GW1–26). All metrics per 90. Source: FPL API 2025/26 event live endpoints (GW1-26 player files), processed via `scripts/player_radar_profile.py`.*

Garner's profile is one of the most analytically interesting in the division. Across 2,333 minutes this season:

| Metric | Per 90 | Percentile (vs 121 PL midfielders) |
| --- | --- | --- |
| Defensive Contribution | 12.4 | **95th** |
| Tackles | 3.0 | **93rd** |
| Ball Recoveries | 5.2 | **79th** |
| Chance Creation (Creativity) | 22.8 | 66th |
| Overall Influence | 23.1 | **81st** |
| xG Involvements | 0.23 | 42nd |

What makes this profile compelling for recruitment analysis: Garner sits in the top 5% of all PL midfielders for defensive coverage, yet his influence (81st pct) confirms this is not a pure defensive holding role. The xGI (42nd vs midfielders) reflects a legitimate positional tradeoff -- midfielders as a group generate more xGI chances than all outfield positions pooled, placing Garner correctly as a defensive-first CM rather than a penalty-box contributor. The gap between 42nd pct xGI and 95th pct defensive contribution is not a weakness -- it is a **system fit** question.

### Comparison: Garner vs Wharton vs Tielemans

![Comparative radar Garner vs Wharton vs Tielemans all PL CM percentiles](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_cm_comparison.png?v=20250302)

*All metrics per 90, percentile vs **249 PL outfield starters** (≥900 min, GW1–26) -- note: this comparison uses the full outfield pool, not the midfielder-only pool used in the individual radar above. Garner's percentiles are higher here because forwards and wingers typically score lower on defensive metrics. Real FPL 2025/26 data.*

| Metric | Garner | Wharton | Tielemans |
| --- | --- | --- | --- |
| Defensive Contribution | **96th** | 82nd | 65th |
| Tackles | **95th** | 86th | 90th |
| Ball Recoveries | 86th | **94th** | 75th |
| Chance Creation | 81st | 80th | **91st** |
| Overall Influence | 76th | 18th | 67th |
| xG Involvements | 62nd | 69th | 63rd |

The comparison clarifies the profile distinctions. Wharton (Crystal Palace) has strong defensive numbers but sits at just the **18th percentile for overall influence** -- effective in the defensive phase but limited in his larger footprint on matches. Tielemans (Aston Villa) ranks highest for creativity (91st) but is 31 percentile points below Garner on defensive contribution. Garner uniquely combines defensive dominance with above-average creative and match-influence numbers -- a rarer profile than any single metric suggests.

### Season Form Arc

![Garner rolling 5-GW form arc across 4 key metrics](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/garner_rolling_arc.png?v=20250302)

*Rolling 5-GW average across 4 per-90 metrics. Weekly bars show individual match values; line shows smoothed form trajectory. GW19 annotated (goal + assist). All 26 GWs played; 2,333 minutes.*

The rolling arc demonstrates consistency and development trajectory -- not just snapshot ability. GW19 (goal + assist vs Leicester) is visible as the natural spike but the underlying defensive contribution and tackles metrics are elevated **across the full second half of the season** (GW14–26), not just the standout week. This is the kind of evidence-based form narrative that separates genuine development from noise -- directly applicable to the same framework used for academy monitoring.

From an academy analytics perspective: this is the same radar logic applied to development -- identifying players who are elite on process metrics (defensive coverage, pressing intensity) even when their output metrics (goals, assists) don't yet reflect it.

---

## Academy Application

![Academy Development Monitor](https://raw.githubusercontent.com/vkenard/football-performance-analytics/main/assets/gw26_academy_development_monitor.png?v=20260221)

*Real-player development monitor built from FPL 2025/26 gameweek player files (Mateus Mané, Wolverhampton Wanderers, FW/AM). The chart tracks FPL creativity per 90 -- appropriate to his attacking role -- and rolling Z-score versus GW cohort baseline to separate development signal from short-term noise. Mané (born September 2007, age 18) made his first PL appearance in GW18 and reached breakout threshold by GW24. He scored against Everton (January 7, GW21) and provided the assist for Wolves' first goal in the WOL v ARS 2-2 (GW26, Feb 18) -- the same match analysed as the Black Swan example in Notebook 3. Three sections of this portfolio track the same subject independently; the coherence is real, not constructed.*

The methodology here is directly transferable to academy performance analysis:

- **Development tracking** -- replace match outcome with development KPIs (sprint load, pressing intensity, passing accuracy under pressure) and apply the same rolling drift monitor to detect improvement or regression across a season
- **Isolating process from luck** -- Brier Score measures *probability quality*, not just win rate. A player or team can be improving without the wins to show for it; this framework captures that signal
- **Longitudinal trend analysis** -- the chronological split discipline prevents retrospective overfitting, ensuring insights reflect genuine development rather than noise
- **Coaching interventions** -- drift detection flags windows where performance patterns shift, prompting targeted review rather than waiting for results to deteriorate
- **Recruitment and scouting** -- the xG differential and Dixon-Coles parameters identify players and teams that are structurally outperforming their actual results. A young forward whose team has a consistently negative xG differential (poor service) but who personally contributes above-average shot quality is structurally undervalued by results-based scouting. This framework separates individual contribution from collective outcome -- directly applicable to identifying underpriced talent at academy or senior level.

---

*Portfolio work. All model weights and proprietary feature engineering are withheld. Methodology and diagnostic outputs are shared for analytical review.*
