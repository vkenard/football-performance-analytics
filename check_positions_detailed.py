import pandas as pd
import numpy as np

GW26_CSV = '../03_DATA__Match_Features_Predictions/GW26_PREDICTION_MARKET_COMPARISON.csv'
gw_raw = pd.read_csv(GW26_CSV)

TEAM_ABBR = {
    'Manchester United': 'MUN', 'Man United': 'MUN', 'Man Utd': 'MUN',
    'Manchester City': 'MCI', 'Man City': 'MCI',
    'West Ham': 'WHU', 'West Ham United': 'WHU',
    'Tottenham': 'TOT', 'Tottenham Hotspur': 'TOT', 'Spurs': 'TOT',
    'Chelsea': 'CHE', 'Arsenal': 'ARS', 'Liverpool': 'LIV',
    'Newcastle': 'NEW', 'Newcastle United': 'NEW',
    'Aston Villa': 'AVL', 'Brighton': 'BHA', 'Brighton & Hove Albion': 'BHA',
    'Fulham': 'FUL', 'Brentford': 'BRE', 'Wolves': 'WOL',
    'Wolverhampton': 'WOL', 'Wolverhampton Wanderers': 'WOL',
    'Crystal Palace': 'CRY', 'Everton': 'EVE', 'Leicester': 'LEI',
    'Leicester City': 'LEI', 'Bournemouth': 'BOU', 'AFC Bournemouth': 'BOU',
    'Nottingham Forest': 'NFO', "Nott'm Forest": 'NFO',
    'Southampton': 'SOU', 'Ipswich': 'IPS', 'Ipswich Town': 'IPS',
    'West Brom': 'WBA', 'Burnley': 'BUR', 'Sheffield United': 'SHU',
    'Luton': 'LUT', 'Luton Town': 'LUT',
}

def make_label(home, away):
    h = TEAM_ABBR.get(home, home[:3].upper())
    a = TEAM_ABBR.get(away, away[:3].upper())
    return f"{h} v {a}"

gw_raw['match_label'] = gw_raw.apply(lambda r: make_label(r['Home'], r['Away']), axis=1)

# Get completed matches
gw = gw_raw[gw_raw['actual_total_goals'].notna()].copy().reset_index(drop=True)

# Derive corner columns  
gw['pred_corners_total'] = gw['Corners_Home'] + gw['Corners_Away']

print("Checking Forest vs Wolves vs Sunderland vs Liverpool positioning:")
print("\nFor the volatility heatmap (xG vs Corners):")
for m in ['NFO v WOL', 'SUN v LIV', 'WHU v MUN']:
    match_row = gw[gw['match_label'] == m]
    if not match_row.empty:
        row = match_row.iloc[0]
        xg_total = row['Total_Goals_xG']
        corners_pred = row['pred_corners_total']
        print(f"  {m}: xG={xg_total:.2f} | Pred_Corners={corners_pred:.1f}")

print("\n\nFor the variance autopsy (LEFT/RIGHT graphs):")
for m in ['WHU v MUN', 'WOL v ARS']:
    match_row = gw[gw['match_label'] == m]
    if not match_row.empty:
        row = match_row.iloc[0]
        conviction = max(row['Blend_H'], row['Blend_D'], row['Blend_A'])
        pred = ['H','D','A'][np.argmax([row['Blend_H'], row['Blend_D'], row['Blend_A']])]
        actual = row['actual_result']
        upset = (pred != actual)
        print(f"  {m}: conviction={conviction:.3f} | predicted={pred} | actual={actual} | upset={upset}")
