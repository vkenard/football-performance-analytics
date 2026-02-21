"""
GENERATE_PLAYER_RADAR.py
========================
Builds per-player radar charts for Everton players using real FPL API data
(26 GW player_gameweek_stats files, 2025-26 season).

Each Everton player with ≥600 minutes gets their own radar showing
percentile rank vs positional peers across the Premier League.

Outputs:
  assets/everton_player_radars.png   — grid of all qualifying Everton players
  assets/everton_player_radar_<name>.png — individual high-res per player
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy.stats import percentileofscore

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = r'c:\Users\bigke\OneDrive\Desktop\VS Code Model'
GW_GLOB     = os.path.join(BASE, 'FPL_RAW_DATA', 'main_2025', 'GW*_player_gameweek_stats.csv')
PLAYERS_CSV = os.path.join(BASE, 'FPL_PLAYERS_2025_2026.csv')
ASSETS_DIR  = os.path.join(BASE, 'football-performance-analytics', 'assets')
os.makedirs(ASSETS_DIR, exist_ok=True)

EVERTON_TEAM_CODE = 11
MIN_MINUTES       = 600   # minimum season minutes to qualify

# ── Radar axis definitions (label: raw_column) ────────────────────────────────
# We use PER-MATCH values from each GW row (FPL API supplies these as cumulative
# season totals at the snapshot date — we take the LAST snapshot per player
# to get the full-season figures, then convert to per-90 where appropriate).
#
# Per-90 axes (season_total / (total_minutes/90)):
#   xG, xA, xGI, CBI, Defensive Contribution
# Raw scaled axes (already normalised by FPL, range ~0-300 per match):
#   Creativity / 90,  Threat / 90

# Columns to pull from GW files
GW_COLS = [
    'id', 'gw', 'minutes',
    'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'goals_scored', 'assists',
    'clean_sheets', 'goals_conceded',
    'clearances_blocks_interceptions', 'tackles', 'recoveries',
    'defensive_contribution',
    'yellow_cards', 'red_cards',
    'creativity', 'threat', 'influence', 'ict_index',
    'bonus', 'bps', 'total_points',
    'saves',                    # GK specific
]

# ── Load & concatenate all GW files ───────────────────────────────────────────
print("Loading GW files…")
gw_files = sorted(glob.glob(GW_GLOB))
print(f"  Found {len(gw_files)} GW files")

frames = []
for f in gw_files:
    df = pd.read_csv(f, usecols=lambda c: c in GW_COLS)
    frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"  Raw rows: {len(raw):,}")

# ── Join with player registry for name, team, position ────────────────────────
players = pd.read_csv(PLAYERS_CSV)
players = players.rename(columns={'player_id': 'id'})

raw = raw.merge(players[['id', 'first_name', 'second_name', 'web_name', 'team_code', 'position']],
                on='id', how='left')

# ── Season aggregates per player ──────────────────────────────────────────────
# Sum counting stats across all GW appearances
SUM_COLS = [
    'minutes', 'expected_goals', 'expected_assists', 'expected_goal_involvements',
    'goals_scored', 'assists', 'clean_sheets', 'goals_conceded',
    'clearances_blocks_interceptions', 'tackles', 'recoveries',
    'defensive_contribution', 'yellow_cards', 'red_cards',
    'creativity', 'threat', 'influence', 'ict_index',
    'bonus', 'bps', 'total_points', 'saves',
]

agg = (raw
       .groupby(['id', 'web_name', 'first_name', 'second_name', 'team_code', 'position'],
                dropna=False)[SUM_COLS]
       .sum()
       .reset_index())

# ── Per-90 normalisation ───────────────────────────────────────────────────────
agg['90s'] = agg['minutes'] / 90.0
agg['90s'] = agg['90s'].replace(0, np.nan)

PER90 = {
    'xG_p90':      'expected_goals',
    'xA_p90':      'expected_assists',
    'xGI_p90':     'expected_goal_involvements',
    'CBI_p90':     'clearances_blocks_interceptions',
    'tackles_p90': 'tackles',
    'recoveries_p90': 'recoveries',
    'def_contrib_p90': 'defensive_contribution',
    'creativity_p90':  'creativity',
    'threat_p90':      'threat',
    'ict_p90':         'ict_index',
    'saves_p90':       'saves',
    'yc_p90':          'yellow_cards',
}
for new_col, raw_col in PER90.items():
    agg[new_col] = agg[raw_col] / agg['90s']

# Discipline: invert yellow cards (lower YC → better discipline score)
agg['discipline_p90'] = 1.0 / (agg['yc_p90'] + 0.1)   # +0.1 avoids div/0; higher = cleaner

# ── Filter qualifying players ─────────────────────────────────────────────────
qualified = agg[agg['minutes'] >= MIN_MINUTES].copy()
print(f"\nQualified players (≥{MIN_MINUTES} min): {len(qualified)}")

# ── Everton squad subset ───────────────────────────────────────────────────────
everton = qualified[qualified['team_code'] == EVERTON_TEAM_CODE].copy()
print(f"Everton qualifying players: {len(everton)}")
print(everton[['web_name', 'position', 'minutes']].sort_values('minutes', ascending=False).to_string(index=False))

# ── Radar configuration by position group ─────────────────────────────────────
POSITION_AXES = {
    'Goalkeeper': {
        'Saves\n/90':        'saves_p90',
        'Goals\nConceded\n/90': 'goals_conceded_p90_inv',   # inverted
        'xGC\n/90 inv':     'xgc_inv_p90',
        'Defensive\nContrib\n/90': 'def_contrib_p90',
        'Discipline':        'discipline_p90',
        'Clean Sheet\nRate': 'cs_rate',
    },
    'Defender': {
        'xG+xA\n/90':        'xGI_p90',
        'Clearances\nBlocks\nInter /90': 'CBI_p90',
        'Tackles\n/90':      'tackles_p90',
        'Defensive\nContrib\n/90': 'def_contrib_p90',
        'Discipline':        'discipline_p90',
        'Recoveries\n/90':   'recoveries_p90',
    },
    'Midfielder': {
        'xG\n/90':           'xG_p90',
        'xA\n/90':           'xA_p90',
        'Creativity\n/90':   'creativity_p90',
        'Threat\n/90':       'threat_p90',
        'Defensive\nContrib\n/90': 'def_contrib_p90',
        'Discipline':        'discipline_p90',
    },
    'Forward': {
        'xG\n/90':           'xG_p90',
        'xA\n/90':           'xA_p90',
        'Threat\n/90':       'threat_p90',
        'Creativity\n/90':   'creativity_p90',
        'Defensive\nContrib\n/90': 'def_contrib_p90',
        'Discipline':        'discipline_p90',
    },
}

# Add derived GK columns to dataframe
qualified['goals_conceded_p90_inv'] = 1.0 / (qualified['goals_conceded'] / qualified['90s'] + 0.1)
qualified['xgc_inv_p90'] = 1.0 / (qualified['expected_goals_conceded'] / qualified['90s'] + 0.1) if 'expected_goals_conceded' in qualified.columns else 0
qualified['cs_rate'] = qualified['clean_sheets'] / (qualified['minutes'] / 90.0 / 10).clip(lower=1)

everton = qualified[qualified['team_code'] == EVERTON_TEAM_CODE].copy()

# ── Radar drawing helper ───────────────────────────────────────────────────────
EVERTON_BLUE  = '#003399'
EVERTON_GOLD  = '#FFD700'
PEER_COLOUR   = '#cccccc'
BG_COLOUR     = '#0d1117'
GRID_COLOUR   = '#2a2a3a'

def draw_radar(ax, percentiles, labels, player_name, position,
               team_name='Everton', colour=EVERTON_BLUE, avg_percentiles=None):
    """Draw a single player radar on `ax`."""
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    vals = list(percentiles) + [percentiles[0]]
    avg_vals = (list(avg_percentiles) + [avg_percentiles[0]]) if avg_percentiles is not None else None

    ax.set_facecolor(BG_COLOUR)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid rings at 20th, 40th, 60th, 80th percentile
    for ring in [20, 40, 60, 80, 100]:
        ax.plot(angles, [ring] * (N + 1), color=GRID_COLOUR, lw=0.8, zorder=1)
        if ring < 100:
            ax.text(0, ring + 2, f'{ring}th', ha='center', va='bottom',
                    color='#555575', fontsize=5, zorder=2)

    # Spokes
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 100], color=GRID_COLOUR, lw=0.8, zorder=1)

    # League average shade (50th percentile ring for reference)
    if avg_vals is not None:
        ax.fill(angles, avg_vals, color='#ffffff', alpha=0.08, zorder=3)
        ax.plot(angles, avg_vals, color='#ffffff', lw=1.0, alpha=0.4, linestyle='--', zorder=4)

    # Player fill
    ax.fill(angles, vals, color=colour, alpha=0.35, zorder=5)
    ax.plot(angles, vals, color=colour, lw=2.2, zorder=6)
    ax.scatter(angles[:-1], percentiles, color=EVERTON_GOLD, s=35, zorder=7, edgecolors='white', linewidths=0.5)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7, color='white', fontweight='bold')
    ax.set_yticks([])
    ax.set_ylim(0, 110)

    # Title
    ax.set_title(f'{player_name}\n({position})', color='white',
                 fontsize=9, fontweight='bold', pad=12)


# ── Build per-position peer groups and compute percentiles ─────────────────────
def compute_percentiles(player_row, peer_df, axes_dict):
    """Return list of percentile scores for a player across all axes."""
    pcts = []
    for label, col in axes_dict.items():
        if col not in peer_df.columns or col not in player_row.index:
            pcts.append(50.0)
            continue
        peer_vals   = peer_df[col].dropna().values
        player_val  = player_row[col]
        if pd.isna(player_val) or len(peer_vals) == 0:
            pcts.append(50.0)
        else:
            pcts.append(float(percentileofscore(peer_vals, player_val, kind='rank')))
    return pcts


def compute_avg_percentiles(peer_df, axes_dict):
    """League-average percentiles (≈50th across all axes by definition)."""
    return [50.0] * len(axes_dict)


# ── Generate individual radar PNGs ─────────────────────────────────────────────
everton_sorted = everton.sort_values('minutes', ascending=False)

individual_files = []
for _, player in everton_sorted.iterrows():
    pos = player['position']
    if pos not in POSITION_AXES:
        continue

    axes_dict = POSITION_AXES[pos]
    peer_df   = qualified[qualified['position'] == pos].copy()

    # Check all required columns exist in peer_df
    available_axes = {lbl: col for lbl, col in axes_dict.items() if col in peer_df.columns}
    if len(available_axes) < 3:
        print(f"  Skipping {player['web_name']} — insufficient columns")
        continue

    labels  = list(available_axes.keys())
    pcts    = compute_percentiles(player, peer_df, available_axes)
    avg_pct = compute_avg_percentiles(peer_df, available_axes)

    fig = plt.figure(figsize=(6, 6), facecolor=BG_COLOUR)
    ax  = fig.add_subplot(111, polar=True, facecolor=BG_COLOUR)

    draw_radar(ax, pcts, labels, player['web_name'], pos, avg_percentiles=avg_pct)

    # Percentile annotation box
    ann_text = '\n'.join([f"{lbl.replace(chr(10),' ')}: {p:.0f}th"
                          for lbl, p in zip(labels, pcts)])
    fig.text(0.01, 0.01, ann_text, color='#aaaacc', fontsize=5.5,
             va='bottom', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.7))

    # Footer
    fig.text(0.5, 0.01,
             f'2025–26 PL Season  |  Peers: {pos}s with ≥{MIN_MINUTES} min  |  Data: FPL API',
             ha='center', va='bottom', color='#888899', fontsize=5.5)

    # Everton branding bar
    fig.text(0.5, 0.97, 'EVERTON FC  ·  Player Recruitment Profile',
             ha='center', va='top', color=EVERTON_GOLD, fontsize=8, fontweight='bold')

    # Legend
    legend_patches = [
        mpatches.Patch(color=EVERTON_BLUE, alpha=0.6, label='Player'),
        plt.Line2D([0], [0], color='white', lw=1.0, linestyle='--', alpha=0.5,
                   label='League avg (50th)'),
        plt.scatter([], [], c=EVERTON_GOLD, s=20, label='Axis score', edgecolors='white', linewidths=0.3),
    ]
    ax.legend(handles=legend_patches, loc='lower right',
              bbox_to_anchor=(1.30, -0.10),
              fontsize=6, facecolor='#1a1a2e', labelcolor='white',
              edgecolor='#333355', framealpha=0.8)

    safe_name = player['web_name'].replace(' ', '_').replace("'", '')
    out_path  = os.path.join(ASSETS_DIR, f'everton_player_radar_{safe_name}.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight',
                facecolor=BG_COLOUR, edgecolor='none')
    plt.close(fig)
    individual_files.append((player['web_name'], pos, player['minutes'], pcts, labels, out_path))
    print(f"  Saved: everton_player_radar_{safe_name}.png  "
          f"({player['minutes']:.0f} min, {pos})")

# ── Grid overview: all Everton players in one figure ──────────────────────────
print("\nBuilding squad overview grid…")

n_players = len(individual_files)
if n_players == 0:
    print("No qualifying Everton players — check minutes threshold.")
else:
    ncols = min(4, n_players)
    nrows = int(np.ceil(n_players / ncols))

    fig = plt.figure(figsize=(5 * ncols, 5 * nrows + 1.2), facecolor=BG_COLOUR)
    fig.suptitle('EVERTON FC  ·  2025–26 Season  ·  Player Recruitment Profiles\n'
                 f'Percentile vs positional peers (PL players ≥{MIN_MINUTES} min)  |  Data: FPL API',
                 color=EVERTON_GOLD, fontsize=12, fontweight='bold',
                 y=0.98)

    for i, (name, pos, mins, pcts, labels, _) in enumerate(individual_files):
        axes_dict = POSITION_AXES[pos]
        available_axes = {lbl: col for lbl, col in axes_dict.items()
                          if col in qualified.columns}
        peer_df = qualified[qualified['position'] == pos].copy()
        avg_pct = compute_avg_percentiles(peer_df, available_axes)

        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        draw_radar(ax, pcts, labels, f'{name}  ({int(mins)} min)', pos,
                   avg_percentiles=avg_pct)

    grid_path = os.path.join(ASSETS_DIR, 'everton_player_radars.png')
    fig.savefig(grid_path, dpi=140, bbox_inches='tight',
                facecolor=BG_COLOUR, edgecolor='none')
    plt.close(fig)
    print(f"\nSquad overview saved → assets/everton_player_radars.png")
    print(f"Individual radars saved ({len(individual_files)} files) → assets/")

print("\nDone.")
