import pandas as pd
import numpy as np

GW26_CSV = '../03_DATA__Match_Features_Predictions/GW26_PREDICTION_MARKET_COMPARISON.csv'
gw_raw = pd.read_csv(GW26_CSV)

# Team abbreviations
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

# Normalize probabilities
for col in ['Blend_H', 'Blend_D', 'Blend_A']:
    if gw_raw[col].max() > 1.5:
        gw_raw[col] = gw_raw[col] / 100.0

# Get completed matches only
gw = gw_raw[gw_raw['actual_total_goals'].notna()].copy().reset_index(drop=True)

# Process results
def is_upset(row):
    pred = ['H','D','A'][np.argmax([row['Blend_H'], row['Blend_D'], row['Blend_A']])]
    return pred != row['actual_result']

gw['upset'] = gw.apply(is_upset, axis=1)
gw['baseline_correct'] = (gw['actual_result'] == 'H')

print("GW26 Upsets:")
upsets = gw[gw['upset']]
for idx, row in upsets.iterrows():
    print(f"  {idx}: {row['match_label']} - Predicted: {['H','D','A'][np.argmax([row['Blend_H'], row['Blend_D'], row['Blend_A']])]} | Actual: {row['actual_result']}")

print(f"\nTotal upsets: {gw['upset'].sum()}/11")

# Check positioning issues - Forest vs Wolves vs Sunderland vs Liverpool
print("\nChecking team positioning:")
for m in ['NFO v WOL', 'SUN v LIV']:
    match_row = gw[gw['match_label'] == m]
    if not match_row.empty:
        row = match_row.iloc[0]
        print(f"  {m}: model_correct={row['blend_pick_correct']}, baseline_correct={row['baseline_correct']}, upset={row['upset']}")
