import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="wide")

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
Congo DR,1645,55,physical,Group K
England,2015,1200,balanced,Group L
Croatia,1885,75,possession,Group L
Ghana,1700,120,wing_attack,Group L
Panama,1695,45,defensive_counter,Group L"""

@st.cache_data(ttl=1)
def load_base_data():
    return pd.read_csv(io.StringIO(csv_data))

teams_df = load_base_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
value_lookup = pd.Series(teams_df.squad_value_m.values, index=teams_df.team_name).to_dict()  # FIXED HERE
style_lookup = pd.Series(teams_df.play_style.values, index=teams_df.team_name).to_dict()

# --- 3. DYNAMIC RE-RATING SYSTEM ---
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

# --- 4. ADVANCED MATH ENGINE ---
def calculate_advanced_matchup(t1, t2, ref_strictness, t1_injury_impact, t2_injury_impact):
    base_elo1 = elo_lookup.get(t1, 1500)
    base_elo2 = elo_lookup.get(t2, 1500)
    
    val1 = value_lookup.get(t1, 50)
    val2 = value_lookup.get(t2, 50)
    value_advantage = (np.log10(val1) - np.log10(val2)) * 35
    
    elo1_effective = (base_elo1 + value_advantage) * (1 - (t1_injury_impact / 100))
    elo2_effective = (base_elo2 - value_advantage) * (1 - (t2_injury_impact / 100))
    
    style1, style2 = style_lookup.get(t1, 'balanced'), style_lookup.get(t2, 'balanced')
    tactical_bonus = 0
    if style1 == 'pressing' and style2 == 'possession': tactical_bonus += 25
    if style1 == 'defensive_counter' and style2 == 'attacking_fluid': tactical_bonus += 20
    if style1 == 'wing_attack' and style2 == 'physical': tactical_bonus += 15
    elo1_effective += tactical_bonus

    if ref_strictness == "Strict / Card Heavy":
        if style1 == 'physical': elo1_effective -= 30
        if style2 == 'physical': elo2_effective -= 30
        if style1 == 'possession': elo1_effective += 15
        if style2 == 'possession': elo2_effective += 15

    win_prob = 1 / (1 + 10 ** (-(elo1_effective - elo2_effective) / 400))
    t1_win = win_prob * 0.76
    t2_win = (1 - win_prob) * 0.76
    tie = 1.0 - t1_win - t2_win
    return t1_win * 100, tie * 100, t2_win * 100

# --- 5. INTERFACE LAYOUT ---
tab1, tab2 = st.tabs(["🔮 Multi-Feature Predictor", "🏆 High-Accuracy Forecaster"])

with tab1:
    st.header("⚽ Deep-Feature Match Analyzer")
    
    selected_group = st.selectbox("Select Group Context", sorted(list(groups.keys())), index=5)
    filtered_teams = groups[selected_group]
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team 1", filtered_teams, index=0)
        t1_inj = st.slider(f"🚨 {team_a} Injury Impact Level", 0, 30, 0)
    with col2:
        team_b = st.selectbox("Team 2", filtered_teams, index=1)
        t2_inj = st.slider(f"🚨 {team_b} Injury Impact Level", 0, 30, 0)
        
    st.write("---")
    st.subheader("🏁 Match Official & Pitch Variables")
    ref_profile = st.radio("Assigned Referee Profile", ["Standard / Balanced", "Strict / Card Heavy", "Lenient / Fluid Play"], horizontal=True)

    if team_a == team_b:
        st.warning("Please choose two separate competitors.")
    else:
        t1_w, tie_p, t2_w = calculate_advanced_matchup(team_a, team_b, ref_profile, t1_inj, t2_inj)
        
        st.write("---")
        st.subheader("📊 Feature-Weighted Outcome Breakdown")
        st.write(f"**{team_a}** Win Chance: `{t1_w:.1f}%` *(Squad Value: {value_lookup[team_a]}M€)*")
        st.progress(int(t1_w))
        st.write(f"**Draw / Tie** Chance: `{tie_p:.1f}%`")
        st.progress(int(tie_p))
        st.write(f"**{team_b}** Win Chance: `{t2_w:.1f}%` *(Squad Value: {value_lookup[team_b]}M€)*")
        st.progress(int(t2_w))

with tab2:
    st.header("🏆 High-Accuracy Simulation Engine")
    st.write("Runs 1,000 algorithmic tournament iterations taking all tactical variations into account.")
    
    if st.button("🚀 Process Advanced Bracket Simulations", type="primary"):
        with st.spinner("Calculating matrices..."):
            trophy_tracker = {team: 0 for team in teams_df['team_name']}
            
            for run in range(1000):
                group_winners, group_runners_up, third_place_pool = [], [], []
                
                for g_name in groups.keys():
                    standings = {team: 0 for team in groups[g_name]}
                    for i in range(len(groups[g_name])):
                        for j in range(i + 1, len(groups[g_name])):
                            t_x, t_y = groups[g_name][i], groups[g_name][j]
                            w, t, l = calculate_advanced_matchup(t_x, t_y, "Standard / Balanced", 0, 0)
                            roll = np.random.rand() * 100
                            if roll < w: standings[t_x] += 3
                            elif roll > (100 - l): standings[t_y] += 3
                            else:
                                standings[t_x] += 1
                                standings[t_y] += 1
                    sorted_s = sorted(standings.items(), key=lambda x: x[1], reverse=True)
                    group_winners.append(sorted_s[0][0])
                    group_runners_up.append(sorted_s[1][0])
                    third_place_pool.append({'team': sorted_s[2][0], 'Points': sorted_s[2][1]})
                
                third_df = pd.DataFrame(third_place_pool).sort_values(by='Points', ascending=False)
                best_8_third = third_df['team'].head(8).tolist()
                
                bracket = group_winners + group_runners_up + best_8_third
                
                while len(bracket) > 1:
                    next_layer = []
                    for i in range(0, len(bracket), 2):
                        team_1 = bracket[i]
                        team_2 = bracket[i+1]
                        w, t, l = calculate_advanced_matchup(team_1, team_2, "Standard / Balanced", 0, 0)
                        next_layer.append(team_1 if np.random.rand() * 100 < (w + t/2) else team_2)
                    bracket = next_layer
                    
                trophy_tracker[bracket[0]] += 1

            results_df = pd.DataFrame(list(trophy_tracker.items()), columns=['Country', 'Titles'])
            results_df['Win Prob (%)'] = (results_df['Titles'] / 1000) * 100
            st.success("High-accuracy calculations complete!")
            st.table(results_df.sort_values(by='Win Prob (%)', ascending=False).reset_index(drop=True).head(15))
