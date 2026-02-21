"""
_generate_garner_arc.py
Two charts:
  1. garner_cm_comparison.png  -- Radar: Garner vs Tielemans vs Wharton
  2. garner_rolling_arc.png    -- Rolling 5-GW form arc for 3 key metrics
Real FPL 2025/26 GW1-22 data. No synthetic data.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs('assets', exist_ok=True)

EVT_BLUE = '#003399'
TEAL     = '#2a9d8f'
ACCENT   = '#E63946'
GOLD     = '#e8a020'
GREY     = '#888888'
LIGHT    = '#f0f4f8'
DARK     = '#222222'

plt.rcParams.update({'font.family': 'sans-serif',
                     'figure.facecolor': 'white'})

GW_DIR = r'c:\Users\bigke\OneDrive\Desktop\VS Code Model\FPL_RAW_DATA\main_2025'

# ── Load all GWs ──────────────────────────────────────────────────────────────
files   = sorted(glob.glob(f'{GW_DIR}/GW*_player_gameweek_stats.csv'))
all_gws = pd.concat([pd.read_csv(f, low_memory=False) for f in files]).reset_index(drop=True)

gk_ids  = all_gws[all_gws['saves'] > 0]['id'].unique()
field   = all_gws[~all_gws['id'].isin(gk_ids)].copy()

# Aggregate per player
agg = field.groupby(['id', 'second_name', 'first_name']).agg(
    total_minutes     = ('minutes',                      'sum'),
    total_xgi         = ('expected_goal_involvements',   'sum'),
    total_creativity  = ('creativity',                   'sum'),
    total_tackles     = ('tackles',                      'sum'),
    total_recoveries  = ('recoveries',                   'sum'),
    total_def_contrib = ('defensive_contribution',       'sum'),
    total_influence   = ('influence',                    'sum'),
    appearances       = ('gw',                          'count'),
).reset_index()

mins = agg['total_minutes'].clip(lower=1)
agg['xgi_p90']         = agg['total_xgi']         / mins * 90
agg['creativity_p90']  = agg['total_creativity']   / mins * 90
agg['tackles_p90']     = agg['total_tackles']      / mins * 90
agg['recoveries_p90']  = agg['total_recoveries']   / mins * 90
agg['def_contrib_p90'] = agg['total_def_contrib']  / mins * 90
agg['influence_p90']   = agg['total_influence']    / mins * 90

pool = agg[agg['total_minutes'] >= 900].copy()
print(f"Pool: {len(pool)} players with ≥900 min")

metrics = ['xgi_p90','creativity_p90','tackles_p90',
           'recoveries_p90','def_contrib_p90','influence_p90']
for m in metrics:
    pool[f'{m}_pct'] = pool[m].rank(pct=True) * 100

# Pull the three players
players = {
    'Garner':    ('James Garner',       EVT_BLUE, '--'),
    'Wharton':   ('Adam Wharton',       ACCENT,   '-'),
    'Tielemans': ('Youri Tielemans',    TEAL,     '-.'),
}
player_data = {}
for surname, (fullname, colour, ls) in players.items():
    row = pool[pool['second_name'] == surname]
    if row.empty:
        print(f"WARNING: {surname} not in pool")
        continue
    r = row.iloc[0]
    player_data[surname] = {
        'label':  fullname,
        'colour': colour,
        'ls':     ls,
        'pcts':   [r[f'{m}_pct'] for m in metrics],
        'mins':   r['total_minutes'],
    }
    print(f"{fullname}: {r['total_minutes']:.0f} min | pcts: {[f'{r[f'{m}_pct']:.0f}' for m in metrics]}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHART 1 — Comparative radar                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
N      = len(metrics)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
ang_p  = angles + [angles[0]]
labels = [
    'xG Involvements\nper 90',
    'Chance Creation\n(Creativity/90)',
    'Tackles\nper 90',
    'Ball Recoveries\nper 90',
    'Defensive\nContribution/90',
    'Overall\nInfluence/90',
]

fig1 = plt.figure(figsize=(12, 6.5), facecolor='white')
ax_r  = fig1.add_axes([0.03, 0.04, 0.52, 0.88], polar=True)
ax_r.set_facecolor(LIGHT)

# Grid rings
for r in [20, 40, 60, 80, 100]:
    ax_r.plot(ang_p, [r]*(N+1), color='white', lw=0.7, zorder=1)
    if r < 100:
        ax_r.text(angles[0], r+2, f'{r}th', ha='center', va='bottom',
                  fontsize=6, color='#aaaaaa')
for ang in angles:
    ax_r.plot([ang, ang], [0, 100], color='white', lw=0.7, zorder=1)

# Each player
for surname, d in player_data.items():
    vals = d['pcts'] + [d['pcts'][0]]
    ax_r.fill(ang_p, vals, alpha=0.12, color=d['colour'], zorder=2)
    ax_r.plot(ang_p, vals, color=d['colour'], lw=2.2, zorder=3, label=d['label'])
    ax_r.scatter(angles, d['pcts'], s=40, color=d['colour'], zorder=4,
                 edgecolors='white', lw=1.2)

# Axis formatting
ax_r.set_xticks(angles)
ax_r.set_xticklabels(labels, fontsize=8.5, fontweight='bold', color=DARK)
ax_r.set_yticklabels([])
ax_r.set_ylim(0, 100)
ax_r.spines['polar'].set_visible(False)

# Right context panel
ax_ctx = fig1.add_axes([0.57, 0.06, 0.40, 0.86])
ax_ctx.axis('off')

ax_ctx.text(0.0, 1.00, 'CM Comparison — PL 2025/26',
            fontsize=16, fontweight='bold', color=DARK, va='top',
            transform=ax_ctx.transAxes)
ax_ctx.text(0.0, 0.91, 'Percentile vs 212 PL outfield starters (≥900 min, GW1–22)\n'
                        'Real FPL 2025/26 GW data · All metrics per 90',
            fontsize=8.5, color='#666666', va='top', transform=ax_ctx.transAxes)

ax_ctx.add_patch(mpatches.FancyBboxPatch(
    (0, 0.837), 1.0, 0.004, boxstyle='square,pad=0',
    facecolor=DARK, transform=ax_ctx.transAxes))

row_labels = ['xGI / 90','Creativity / 90','Tackles / 90',
              'Recoveries / 90','Def. Contrib / 90','Influence / 90']

y = 0.80
# Header row
ax_ctx.text(0.0,  y, 'Metric',    fontsize=8, color='#888', transform=ax_ctx.transAxes, fontweight='bold')
for i, (surname, d) in enumerate(player_data.items()):
    ax_ctx.text(0.52 + i*0.20, y, d['label'].split()[0],
                fontsize=8, color=d['colour'], fontweight='bold',
                ha='center', transform=ax_ctx.transAxes)
y -= 0.06

for j, (rl, m) in enumerate(zip(row_labels, metrics)):
    bg = '#f5f7ff' if j % 2 == 0 else 'white'
    ax_ctx.add_patch(mpatches.FancyBboxPatch(
        (-0.02, y-0.015), 1.04, 0.055, boxstyle='square,pad=0',
        facecolor=bg, transform=ax_ctx.transAxes, zorder=0))
    ax_ctx.text(0.0, y+0.012, rl, fontsize=8.5, color=DARK,
                transform=ax_ctx.transAxes, va='center')
    for i, (surname, d) in enumerate(player_data.items()):
        pct = d['pcts'][j]
        col = d['colour'] if pct >= 75 else (GOLD if pct >= 50 else '#aaaaaa')
        wt  = 'bold' if pct >= 75 else 'normal'
        ax_ctx.text(0.52 + i*0.20, y+0.012, f'{pct:.0f}th',
                    fontsize=9, color=col, fontweight=wt,
                    ha='center', transform=ax_ctx.transAxes, va='center')
    y -= 0.065

# Colour legend swatches
y -= 0.03
handles = [mpatches.Patch(facecolor=d['colour'], label=f"{d['label']} ({d['mins']:.0f} min)")
           for d in player_data.values()]
ax_ctx.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, 0.0),
              fontsize=8.5, frameon=False)

fig1.text(0.5, 0.975, 'Percentile values highlighted: ≥75th bold colour  ·  50–74th gold  ·  <50th grey',
          ha='center', fontsize=8, color='#888888', style='italic')

out1 = 'assets/garner_cm_comparison.png'
fig1.savefig(out1, dpi=155, bbox_inches='tight', facecolor='white')
plt.close(fig1)
print(f'Saved: {out1}')

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  CHART 2 — Garner rolling 5-GW form arc                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
garner_gw = field[field['second_name']=='Garner'].sort_values('gw').reset_index(drop=True)

# Normalise each weekly value per-90
g90 = garner_gw.copy()
mins_clip = g90['minutes'].clip(lower=1)
g90['xgi_p90']  = g90['expected_goal_involvements'] / mins_clip * 90
g90['cre_p90']  = g90['creativity']                 / mins_clip * 90
g90['def_p90']  = g90['defensive_contribution']     / mins_clip * 90
g90['tck_p90']  = g90['tackles']                    / mins_clip * 90

# Rolling 5-GW average
ROLL = 5
g90['xgi_roll']  = g90['xgi_p90'].rolling(ROLL, min_periods=2).mean()
g90['cre_roll']  = g90['cre_p90'].rolling(ROLL, min_periods=2).mean()
g90['def_roll']  = g90['def_p90'].rolling(ROLL, min_periods=2).mean()
g90['tck_roll']  = g90['tck_p90'].rolling(ROLL, min_periods=2).mean()

gws = g90['gw'].values

fig2, axes = plt.subplots(2, 2, figsize=(13, 7), facecolor='white')
fig2.subplots_adjust(hspace=0.42, wspace=0.30)

panels = [
    ('def_roll', 'def_p90',  'Defensive Contribution / 90',  EVT_BLUE, axes[0,0]),
    ('tck_roll', 'tck_p90',  'Tackles / 90',                 ACCENT,   axes[0,1]),
    ('cre_roll', 'cre_p90',  'Chance Creation (Creativity / 90)', TEAL, axes[1,0]),
    ('xgi_roll', 'xgi_p90',  'xG Involvements / 90',         GOLD,     axes[1,1]),
]

for roll_col, raw_col, title, colour, ax in panels:
    ax.set_facecolor('#f8f9fa')
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='white', linewidth=0.8)
    # Raw bars (weekly)
    ax.bar(gws, g90[raw_col], color=colour, alpha=0.20, width=0.7, label='Weekly value')
    # Rolling line
    ax.plot(gws, g90[roll_col], color=colour, lw=2.4, zorder=5,
            label=f'{ROLL}-GW rolling avg')
    ax.scatter(gws, g90[roll_col], s=28, color=colour, zorder=6,
               edgecolors='white', linewidths=0.8)
    # Season average line
    season_avg = g90[raw_col].mean()
    ax.axhline(season_avg, color=colour, lw=1.0, linestyle=':', alpha=0.55,
               label=f'Season avg ({season_avg:.2f})')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color=DARK, pad=5)
    ax.set_xlabel('Gameweek', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xlim(0.5, 22.5)
    ax.set_xticks(gws[::2])
    ax.legend(fontsize=7.5, frameon=False)
    for sp in ['top','right']:
        ax.spines[sp].set_visible(False)

# Annotate the GW19 spike (goal + assist)
for roll_col, raw_col, title, colour, ax in panels:
    gw19_rows = g90.loc[g90['gw']==19, raw_col]
    if len(gw19_rows):
        ax.annotate('GW19\nG+A', xy=(19, gw19_rows.values[0]),
                    xytext=(16.5, g90[raw_col].max()*0.88),
                    fontsize=7, color=colour, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.2))

fig2.suptitle(
    'James Garner — Rolling Form Arc  |  PL 2025/26 GW1–22\n'
    'Real FPL GW data · Per-90 values · All 22 starts (1,973 min)',
    fontsize=11, fontweight='bold', color=DARK, y=1.01
)

out2 = 'assets/garner_rolling_arc.png'
fig2.savefig(out2, dpi=155, bbox_inches='tight', facecolor='white')
plt.close(fig2)
print(f'Saved: {out2}')
