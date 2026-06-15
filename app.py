import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

# --- 1. REAL-WORLD MATCH LEDGER ---
REAL_WORLD_RESULTS = [
    {"home_team": "Netherlands", "away_team": "Japan", "home_score": 2, "away_score": 2, "status": "FT"},
    {"home_team": "Germany", "away_team": "Curaçao", "home_score": 7, "away_score": 1, "status": "FT"}
]

# --- 2. ADVANCED FEATURE DATASET ---
csv_data = """team_name,elo_rating,squad_value_m,play_style,group_assignment
Mexico,1820,240,balanced,Group A
South Africa,1610,45,defensive_counter,Group A
South Korea,1795,70,possession,Group A
Czechia,1740,64,physical,Group A
Canada,1730,190,wing_attack,Group B
Switzerland,1815,280,balanced,Group B
Qatar,1640,25,defensive_counter,Group B
Bosnia and Herzegovina,1685,85,physical,Group B
Brazil,2010,1050,attacking_fluid,Group C
Morocco,1890,350,possession,Group C
Haiti,1590,15,defensive_counter,Group C
Scotland,1710,67,physical,Group C
United States,1845,320,pressing,Group D
Paraguay,1720,65,defensive_counter,Group D
Australia,1735,68,physical,Group D
Türkiye,1780,71,attacking_fluid,Group D
Germany,1980,81,pressing,Group E
Curaçao,1520,58,balanced,Group E
Ivory Coast,1775,72,physical,Group E
Ecuador,1830,73,pressing,Group E
Netherlands,1940,620,possession,Group F
Japan,1825,290,wing_attack,Group F
Tunisia,1665,62,defensive_counter,Group F
Sweden,1805,71,balanced,Group F
Belgium,1915,75,attacking_fluid,Group G
Egypt,1745,69,defensive_counter,Group G
Iran,1715,66,balanced,Group G
New Zealand,1580,59,physical,Group G
Spain,2025,980,possession,Group H
Cabo Verde,1690,32,balanced,Group H
Saudi Arabia,1655,63,possession,Group H
Uruguay,1935,80,pressing,Group H
France,2040,1110,attacking_fluid,Group I
Senegal,1810,74,physical,Group I
Norway,1790,72,wing_attack,Group I
Iraq,1670,64,balanced,Group I
Argentina,2065,920,possession,Group J
Algeria,1730,115,balanced,Group J
Austria,1815,73,pressing,Group J
Jordan,1625,61,defensive_counter,Group J
Portugal,1990,83,attacking_fluid,Group K
Uzbekistan,1705,67,balanced,Group K
Colombia,1895,275,pressing,Group K
Congo DR,1645,63,physical,Group K
England,2015,1200,balanced,Group L
Croatia,1885,75,possession,Group L
Ghana,1700,66,wing_attack,Group L
Panama,1695,45,defensive_counter,Group L"""

@st.cache_data(ttl=1)
def load_base_data():
    return pd.read_csv(io.StringIO(csv_data))

teams_df = load_base_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
value_lookup = pd.Series(teams_df.squad_value_m.values, index=teams_df.team_name).to_dict
