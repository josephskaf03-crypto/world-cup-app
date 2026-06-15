import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

# --- 1. REAL-WORLD MATCH LEDGER (For AI Continuous Learning) ---
REAL_WORLD_RESULTS = [
    {"home_team": "Netherlands", "away_team": "Japan", "home_score": 2, "away_score": 2, "status": "FT"},
    {"home_team": "Germany", "away_team": "Curaçao", "home_score": 7, "away_score": 1, "status": "FT"}
]

# --- 2. ADVANCED FEATURE DATASET ---
# Added: Star Player Roster Value (M€) & Primary Tactical Configurations
csv_data = """team_name,elo_rating,squad_value_m,play_style,group_assignment
Mexico,1820,240,balanced,Group A
South Africa,1610,45,defensive_counter,Group A
South Korea,1795,165,possession,Group A
Czechia,1740,130,physical,Group A
Canada,1730,190,wing_attack,Group B
Switzerland,1815,280,balanced,Group B
Qatar,1640,25,defensive_counter,Group B
Bosnia and Herzegovina,1685,85,physical,Group B
Brazil,2010,1050,attacking_fluid,Group C
Morocco,1890,350,possession,Group C
Haiti,1590,15,defensive_counter,Group C
Scotland,1710,110,physical,Group C
United States,1845,320,pressing,Group D
Paraguay,1720,105,defensive_counter,Group D
Australia,1735,50,physical,Group D
Türkiye,1780,210,attacking_fluid,Group D
Germany,1980,850,pressing,Group E
Curaçao,1520,12,balanced,Group E
Ivory Coast,1775,260,physical,Group E
Ecuador,1830,195,pressing,Group E
Netherlands,1940,620,possession,Group F
Japan,1825,290,wing_attack,Group F
Tunisia,1665,40,defensive_counter,Group F
Sweden,1805,215,balanced,Group F
Belgium,1915,540,attacking_fluid,Group G
Egypt,1745,140,defensive_counter,Group G
Iran,1715,65,balanced,Group G
New Zealand,1580,22,physical,Group G
Spain,2025,980,possession,Group H
Cabo Verde,1690,32,balanced,Group H
Saudi Arabia,1655,30,possession,Group H
Uruguay,1935,420,pressing,Group H
France,2040,1110,attacking_fluid,Group I
Senegal,1810,230,physical,Group I
Norway,1790,380,wing_attack,Group I
Iraq,1670,18,balanced,Group I
Argentina,2065,920,possession,Group J
Algeria,1730,115,balanced,Group J
Austria,1815,245,pressing,Group J
Jordan,1625,14,defensive_counter,Group J
Portugal,1990,890,attacking_fluid,Group K
Uzbekistan,1705,35,balanced,Group K
Colombia,1895,275,pressing,Group K
Congo DR,1645,55,physical,Group K
England,2015,1200,balanced,Group L
Croatia,1885,295,possession,Group L
Ghana,1700,120,wing_attack,Group L
Panama,1695,45,defensive_counter,Group L"""

@st.cache_data
def load_base_data():
    return pd.read_csv(io.StringIO(csv_data))

teams_df = load_base_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
value_lookup = pd.Series(teams_df.squad_value_m.values, index=teams_df.team_name).to_dict()
style_lookup = pd.Series(teams_df.play_style.values, index=teams_df.team_name).to_dict()

# --- 3. THE AI LEARNING ENGINE (DYNAMIC RE-RATING) ---
K_FACTOR = 40
for match in REAL_WORLD_RESULTS:
    t1, t2 = match["home_team"], match["away_team"]
    s1, s2 = match["home_score"], match["away_score"]
    if t1 in elo_lookup and t2 in elo_lookup:
        elo1, elo2 = elo_lookup[t1], elo_lookup[t2]
        expected_1 = 1 / (1 + 10 ** (-(elo1 - elo2) / 400))
        actual_1 = 1.0 if s1 > s2 else (0.0 if s1 < s2 else 0.5)
        shift = K_FACTOR * (actual_1 - expected_1)
        elo_lookup[t1] += shift
        elo_lookup[t2] -= shift

groups = teams_df.groupby('group_assignment')['team_name'].apply(list).to_dict()

# --- 4. DATA-SCIENCE PROBABILITY MULTI-FEATURE ENGINE ---
def calculate_advanced_matchup(t1, t2, ref_strictness, t1_injury_impact, t2_injury_impact):
    base_elo1 = elo_lookup.get(t1, 1500)
    base_elo2 = elo_lookup.get(t2, 1500)
    
    # Feature 1: Dynamic Squad Value Modifier (Simulating Star Quality / Squad Depth)
    val1 = value_lookup.get(t1, 50)
    val2 = value_lookup.get(t2, 50)
    value_advantage = (np.log10(val1) - np.log10(val2)) * 35
    
    # Feature 2: Active Injury Impact Slashing
    elo1_effective = (base_elo1 + value_advantage) * (1 - (t1_injury_impact / 100))
    elo2_effective = (base_elo2 - value_advantage) * (1 - (t2_injury_impact / 100))
    
    # Feature 3: Tactical Play Style Advantages Matrix
    style1, style2 = style_lookup.get(t1, 'balanced'), style_lookup.get(t2, 'balanced')
    tactical_bonus = 0
    if style1 == 'pressing' and style2 == 'possession': tactical_bonus += 25
    if style1 == 'defensive_counter' and style2 == 'attacking_fluid': tactical_bonus += 20
    if style1 == 'wing_attack' and style2 == 'physical': tactical_bonus += 15
    elo1_effective += tactical_bonus

    # Feature 4: Referee Strictness Ingestion
    # Strict referees favor highly technical possession squads; disadvantage highly physical teams
    if ref_strictness == "Strict / Card Heavy":
        if style1 == 'physical': elo1_effective -= 30
        if style2 == 'physical': elo2_effective -= 30
        if style1 == 'possession': elo1_effective += 15
        if style2 == 'possession': elo2_effective += 15

    # Core Calculation
    win_prob = 1 / (1 + 10 ** (-(elo1_effective - elo2_effective) / 400))
    t1_win = win_prob * 0.76
    t2_win = (1 - win_prob) * 0.76
    tie = 1.0 - t1_win - t2_win
    return t1_win * 100, tie * 100, t2_win * 100

# --- 5. INTERFACE LAYOUT ---
tab1, tab2 = st.tabs(["🔮 Multi-Feature Predictor", "🏆 High-Accuracy Forecaster"])

with tab1:
    st.header("⚽ Deep-Feature Match Analyzer")
    st.write("Tune player availability and referee profiles before estimating game probabilities.")
    
    selected_group = st.selectbox("Select Group Context", sorted(list(groups.keys())), index=5)
    filtered_teams = groups[selected_group]
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team 1", filtered_teams, index=0)
        t1_inj = st.slider(f"🚨 {team_a} Injury Impact Level", 0, 30, 0, help="0% = Full Strength squad. 30% = Missing absolute star players.")
    with col2:
        team_b = st.selectbox("Team 2", filtered_teams, index=1)
        t2_inj = st.slider(f"🚨 {team_b} Injury Impact Level", 0, 30, 0, help="Simulate tactical losses or fatigue.")
        
    st.write("---")
    st.subheader("🏁 Match Official & Pitch Variables")
    ref_profile = st.radio("Assigned Referee Profile", ["Standard / Balanced", "Strict / Card Heavy", "Lenient / Fluid Play"], horizontal=True)

    if team_a == team_b:
        st.warning("Please choose two separate competitors.")
    else:
        t1_w, tie_p, t2_w = calculate_advanced_matchup(team_a, team_b, ref_profile, t1_inj, t2_inj)
        
        st.write("---")
        st.subheader("📊 Feature-Weighted Outcome Breakdown")
        st.write(f"**{team_a}** Win Chance: `{t1_w:.1f}%` *(Squad Value: {value_lookup[team_a]}M€ | Style: {style_lookup[team_a].replace('_',' ')})*")
        st.progress(int(t1_w))
        
        st.write(f"**Draw / Tie** Chance: `{tie_p:.1f}%`")
        st.progress(int(tie_p))
        
        st.write(f"**{team_b}** Win Chance: `{t2_w:.1f}%` *(Squad Value: {value_lookup[team_b]}M€ | Style: {style_lookup[team_b].replace('_',' ')})*")
        st.progress(int(t2_w))

with tab2:
    st.header("🏆 High-Accuracy Simulation Engine")
    st.write("Runs 1,000 algorithmic tournament iterations taking all tactical variations into account.")
    
    if st.button("🚀 Process Advanced Bracket Simulations", type="primary"):
        with st.spinner("Calculating matrices..."):
            trophy_tracker = {team: 0 for team in teams_df['team_name']}
            for run in range(1000):
                # Simulations here use baseline weights updated by the live match ledger results
                group_winners, group_runners_up = [], []
                for g_name in groups.keys():
                    standings = {team: 0 for team in groups[g_name]}
                    for i in range(len(groups[g_name])):
                        for j in range(i + 1, len(groups[g_name])):
                            t_a, t_b = groups[g_name][i], groups[g_name][j]
                            w, t, l = calculate_advanced_matchup(t_a, t_b, "Standard / Balanced", 0, 0)
                            roll = np.random.rand() * 100
                            if roll < w: standings[t_a] += 3
                            elif roll > (100 - l): standings[t_b] += 3
                            else:
                                standings[t_a] += 1
                                standings[t_b] += 1
                    sorted_s = sorted(standings.items(), key=
