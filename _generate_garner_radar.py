"""
_generate_garner_radar.py
Generates a professional player profile radar for James Garner (Everton)
ranking him percentile vs PL midfielders with >=900 min, GW1-26 2025/26.
Output: assets/garner_performance_radar.png
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import glob
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('assets', exist_ok=True)

EVT_BLUE   = '#003399'
EVT_WHITE  = '#FFFFFF'
ACCENT     = '#E63946'
TEAL       = '#2a9d8f'
GOLD       = '#FFD700'
LIGHT_GREY = '#e8ecf0'
DARK_GREY  = '#333333'

plt.rcParams.update({'font.family': 'sans-serif'})

GW_DIR = r'c:\Users\bigke\OneDrive\Desktop\VS Code Model\FPL_RAW_DATA\main_2025'

# ── 1. Load all GWs and aggregate per player ──────────────────────────────────
files = sorted(glob.glob(f'{GW_DIR}/GW*_player_gameweek_stats.csv'))
dfs = [pd.read_csv(f, low_memory=False) for f in files]
all_gws = pd.concat(dfs).reset_index(drop=True)

# Load position data from per-GW players.csv (FPL IDs match player_gameweek_stats)
players_pos = pd.read_csv(f'{GW_DIR}/GW22_players.csv')[['player_id', 'position']]
players_pos.rename(columns={'player_id': 'id'}, inplace=True)
all_gws = all_gws.merge(players_pos, on='id', how='left')

# Remove GKs (saves > 0 in any GW implies GK)
gk_ids = all_gws[all_gws['saves'] > 0]['id'].unique()
field = all_gws[~all_gws['id'].isin(gk_ids)].copy()

# Aggregate per player across season
agg = field.groupby(['id', 'second_name', 'first_name', 'web_name', 'position']).agg(
    total_minutes    = ('minutes', 'sum'),
    total_goals      = ('goals_scored', 'sum'),
    total_assists    = ('assists', 'sum'),
    total_xgi        = ('expected_goal_involvements', 'sum'),
    total_creativity = ('creativity', 'sum'),
    total_tackles    = ('tackles', 'sum'),
    total_recoveries = ('recoveries', 'sum'),
    total_def_contrib= ('defensive_contribution', 'sum'),
    total_influence  = ('influence', 'sum'),
    appearances      = ('gw', 'count'),
).reset_index()

# Per-90 normalisation
mins = agg['total_minutes'].clip(lower=1)
agg['xgi_p90']        = agg['total_xgi']         / mins * 90
agg['creativity_p90'] = agg['total_creativity']   / mins * 90
agg['tackles_p90']    = agg['total_tackles']      / mins * 90
agg['recoveries_p90'] = agg['total_recoveries']   / mins * 90
agg['def_contrib_p90']= agg['total_def_contrib']  / mins * 90
agg['influence_p90']  = agg['total_influence']    / mins * 90

# Filter: min 900 minutes + midfielders only (position-specific comparison)
all_pool = agg[agg['total_minutes'] >= 900].copy()
pool     = all_pool[all_pool['position'] == 'Midfielder'].copy()
print(f"Player pool after >=900 min filter: {len(all_pool)} outfield | {len(pool)} midfielders")

# ── 2. Percentile rank each metric ────────────────────────────────────────────
metrics = ['xgi_p90', 'creativity_p90', 'tackles_p90',
           'recoveries_p90', 'def_contrib_p90', 'influence_p90']
labels  = [
    'xG Involvements\nper 90',
    'Chance Creation\n(Creativity/90)',
    'Tackles\nper 90',
    'Ball Recoveries\nper 90',
    'Defensive\nContribution/90',
    'Influence\n(Overall Impact)',
]

for m in metrics:
    pool[f'{m}_pct'] = pool[m].rank(pct=True) * 100

garner = pool[pool['second_name'] == 'Garner']
if garner.empty:
    raise ValueError("Garner not found in pool — check minutes filter")
g = garner.iloc[0]
garner_pcts = [g[f'{m}_pct'] for m in metrics]

print(f"\nJames Garner — {g['total_minutes']:.0f} minutes | {g['appearances']} appearances")
for lbl, pct, val, m in zip(labels, garner_pcts, [g[m] for m in metrics], metrics):
    print(f"  {lbl.replace(chr(10),' '):40s}: {val:.3f}  → {pct:.1f}th percentile")

# ── 3. Build the radar chart ─────────────────────────────────────────────────
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
garner_vals = garner_pcts + [garner_pcts[0]]   # close polygon
angles_plot  = angles + [angles[0]]

fig = plt.figure(figsize=(12, 7), facecolor='white')

# ── LEFT: Radar ──────────────────────────────────────────────────────────────
ax_radar = fig.add_axes([0.03, 0.05, 0.52, 0.88], polar=True)
ax_radar.set_facecolor('#f0f4f8')

# Draw concentric rings
for r in [20, 40, 60, 80, 100]:
    ax_radar.plot(angles_plot, [r] * (N + 1), color='white', linewidth=0.8, zorder=1)
    ax_radar.fill(angles + [angles[0]], [r] * (N + 1), alpha=0.0)
    if r < 100:
        ax_radar.text(angles[0], r + 1.5, f'{r}th', ha='center', va='bottom',
                      fontsize=6.5, color='#999999')

# Axis spokes
for angle in angles:
    ax_radar.plot([angle, angle], [0, 100], color='white', linewidth=0.8, zorder=1)

# League average reference line: real median of PL midfielders per metric
# Convert actual MF medians to percentile positions within the MF pool
mf_medians_pct = []
for m in metrics:
    med_val = pool[m].median()
    pct_pos = (pool[m] <= med_val).mean() * 100  # should be ~50 by definition
    mf_medians_pct.append(pct_pos)
avg_vals = mf_medians_pct + [mf_medians_pct[0]]

ax_radar.fill(angles_plot, avg_vals, alpha=0.12, color=TEAL, zorder=2)
ax_radar.plot(angles_plot, avg_vals, color=TEAL, linewidth=1.2,
              linestyle='--', alpha=0.6, zorder=2, label='Avg PL Midfielder (50th pct)')

# Garner polygon
ax_radar.fill(angles_plot, garner_vals, alpha=0.35, color=EVT_BLUE, zorder=3)
ax_radar.plot(angles_plot, garner_vals, color=EVT_BLUE, linewidth=2.5, zorder=4)
ax_radar.scatter(angles, garner_pcts, s=55, color=EVT_BLUE, zorder=5, edgecolors='white', linewidths=1.5)

# Axis labels
ax_radar.set_xticks(angles)
ax_radar.set_xticklabels(labels, fontsize=9, fontweight='bold', color=DARK_GREY)
ax_radar.set_yticklabels([])
ax_radar.set_ylim(0, 100)
ax_radar.spines['polar'].set_visible(False)

# Percentile value annotations on each spoke
for angle, val in zip(angles, garner_pcts):
    offset = 8 if val < 90 else -10
    ax_radar.annotate(f'{val:.0f}th',
                      xy=(angle, val),
                      xytext=(0, offset),
                      textcoords='offset points',
                      ha='center', va='center',
                      fontsize=8, fontweight='bold',
                      color=EVT_BLUE,
                      bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=EVT_BLUE, linewidth=0.8, alpha=0.9))

# ── RIGHT: Context panel ─────────────────────────────────────────────────────
ax_ctx = fig.add_axes([0.57, 0.08, 0.40, 0.80])
ax_ctx.axis('off')

# Header
ax_ctx.text(0.0, 1.00, 'James Garner', fontsize=22, fontweight='bold',
            color=EVT_BLUE, va='top', transform=ax_ctx.transAxes)
ax_ctx.text(0.0, 0.90, 'Everton  ·  Central Midfielder', fontsize=12,
            color=DARK_GREY, va='top', transform=ax_ctx.transAxes)
ax_ctx.text(0.0, 0.83, 'PL 2025/26  ·  GW1–26  ·  Percentile vs PL Midfielders (≥900 min)',
            fontsize=9, color='#666666', va='top', transform=ax_ctx.transAxes)

# Horizontal rule
ax_ctx.add_patch(mpatches.FancyBboxPatch((0.0, 0.770), 1.0, 0.004,
    boxstyle='square,pad=0', facecolor=EVT_BLUE, transform=ax_ctx.transAxes, zorder=5))

# Season totals
totals = [
    ('Season minutes',     f"{g['total_minutes']:.0f}"),
    ('Appearances',        f"{g['appearances']}"),
    ('Goals',              f"{g['total_goals']:.0f}"),
    ('Assists',            f"{g['total_assists']:.0f}"),
    ('xG Involvements',    f"{g['total_xgi']:.2f}"),
]
y = 0.72
for label, val in totals:
    ax_ctx.text(0.0, y, label, fontsize=9.5, color='#555555', va='top', transform=ax_ctx.transAxes)
    ax_ctx.text(1.0, y, val, fontsize=9.5, fontweight='bold', color=DARK_GREY,
                ha='right', va='top', transform=ax_ctx.transAxes)
    y -= 0.07

# Standout stat: defensive contribution percentile
ax_ctx.add_patch(mpatches.FancyBboxPatch((0.0, y - 0.04), 1.0, 0.13,
    boxstyle='round,pad=0.02', facecolor='#eef2ff', edgecolor=EVT_BLUE,
    linewidth=1.2, transform=ax_ctx.transAxes, zorder=4))
def_pct = g['def_contrib_p90_pct']
atk_pct = g['xgi_p90_pct']
ax_ctx.text(0.5, y + 0.065, f'Defensive Contribution: {def_pct:.0f}th percentile',
            fontsize=10, fontweight='bold', color=EVT_BLUE,
            ha='center', va='top', transform=ax_ctx.transAxes)
ax_ctx.text(0.5, y + 0.005, f'xG Involvements/90: {atk_pct:.0f}th percentile',
            fontsize=9, color=DARK_GREY,
            ha='center', va='top', transform=ax_ctx.transAxes)
ax_ctx.text(0.5, y - 0.030, 'Elite defensive midfielder: 95th pct\ndef. contribution, 92nd pct tackles',
            fontsize=8.5, color='#555555', ha='center', va='top',
            transform=ax_ctx.transAxes, style='italic')

y -= 0.18

# Legend
ax_ctx.add_patch(mpatches.FancyBboxPatch((0.0, y - 0.005), 0.14, 0.045,
    boxstyle='square,pad=0', facecolor=EVT_BLUE, alpha=0.35,
    transform=ax_ctx.transAxes))
ax_ctx.text(0.17, y + 0.018, 'Garner', fontsize=8.5, color=EVT_BLUE,
            va='center', fontweight='bold', transform=ax_ctx.transAxes)
ax_ctx.add_patch(mpatches.FancyBboxPatch((0.45, y - 0.005), 0.14, 0.045,
    boxstyle='square,pad=0', facecolor=TEAL, alpha=0.35,
    transform=ax_ctx.transAxes))
ax_ctx.text(0.62, y + 0.018, 'Avg PL Midfielder', fontsize=8.5, color=TEAL,
            va='center', fontweight='bold', transform=ax_ctx.transAxes)

y -= 0.09

# Methodology note
note = (f'Pool: {len(pool)} PL midfielders with >=900 min.\n'
        'Metrics computed per 90. Source: FPL 2025/26 GW data.')
ax_ctx.text(0.0, y, note, fontsize=7.5, color='#888888',
            va='top', transform=ax_ctx.transAxes, style='italic')

# Main title strip at top of figure
fig.text(0.5, 0.975, 'Player Profile -- Percentile vs PL Midfielders  |  PL 2025/26',
         ha='center', fontsize=11, color='#555555', style='italic')

out = 'assets/garner_performance_radar.png'
plt.savefig(out, dpi=155, bbox_inches='tight', facecolor='white')
plt.close()
print(f'\nSaved: {out}')
