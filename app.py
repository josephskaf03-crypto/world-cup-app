import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

# 1. Complete, Revised Official 48-Team Dataset
csv_data = """team_name,elo_rating,form_index,play_style,group_assignment
Mexico,1820,72,balanced,Group A
South Africa,1610,65,defensive_counter,Group A
South Korea,1795,70,possession,Group A
Czechia,1740,64,physical,Group A
Canada,1730,66,wing_attack,Group B
Switzerland,1815,69,balanced,Group B
Qatar,1640,61,defensive_counter,Group B
Bosnia and Herzegovina,1685,63,physical,Group B
Brazil,2010,82,attacking_fluid,Group C
Morocco,1890,78,possession,Group C
Haiti,1590,60,defensive_counter,Group C
Scotland,1710,67,physical,Group C
United States,1845,74,pressing,Group D
Paraguay,1720,65,defensive_counter,Group D
Australia,1735,68,physical,Group D
Türkiye,1780,71,attacking_fluid,Group D
Germany,1980,81,pressing,Group E
Curaçao,1520,58,balanced,Group E
Ivory Coast,1775,72,physical,Group E
Ecuador,1830,73,pressing,Group E
Netherlands,1940,79,possession,Group F
Japan,1825,76,wing_attack,Group F
Tunisia,1665,62,defensive_counter,Group F
Sweden,1805,71,balanced,Group F
Belgium,1915,75,attacking_fluid,Group G
Egypt,1745,69,defensive_counter,Group G
Iran,1715,66,balanced,Group G
New Zealand,1580,59,physical,Group G
Spain,2025,84,possession,Group H
Cabo Verde,1690,69,balanced,Group H
Saudi Arabia,1655,63,possession,Group H
Uruguay,1935,80,pressing,Group H
France,2040,85,attacking_fluid,Group I
Senegal,1810,74,physical,Group I
Norway,1790,72,wing_attack,Group I
Iraq,1670,64,balanced,Group I
Argentina,2065,86,possession,Group J
Algeria,1730,68,balanced,Group J
Austria,1815,73,pressing,Group J
Jordan,1625,61,defensive_counter,Group J
Portugal,1990,83,attacking_fluid,Group K
Uzbekistan,1705,67,balanced,Group K
Colombia,1895,79,pressing,Group K
Congo DR,1645,63,physical,Group K
England,2015,84,balanced,Group L
Croatia,1885,75,possession,Group L
Ghana,1700,66,wing_attack,Group L
Panama,1695,65,defensive_counter,Group L"""

@st.cache_data
def load_data():
    return pd.read_csv(io.StringIO(csv_data))

teams_df = load_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
groups = teams_df.groupby('group_assignment')['team_name'].apply(list).to_dict()

# 2. Match Probability Mathematics
def calculate_probabilities(t1, t2):
    elo_1, elo_2 = elo_lookup.get(t1, 1500), elo_lookup.get(t2, 1500)
    win_prob = 1 / (1 + 10 ** (-(elo_1 - elo_2) / 400))
    t1_win = win_prob * 0.78
    t2_win = (1 - win_prob) * 0.78
    tie = 1.0 - t1_win - t2_win
    return t1_win * 100, tie * 100, t2_win * 100

def play_match(t1, t2):
    elo_1, elo_2 = elo_lookup.get(t1, 1500), elo_lookup.get(t2, 1500)
    win_prob = 1 / (1 + 10 ** (-(elo_1 - elo_2) / 400))
    roll = np.random.rand()
    if roll < win_prob * 0.8: return 3, 0
    elif roll > (1 - (1 - win_prob) * 0.8): return 0, 3
    else: return 1, 1

def play_knockout(t1, t2):
    pts1, pts2 = play_match(t1, t2)
    if pts1 == 3: return t1
    elif pts2 == 3: return t2
    else: return t1 if np.random.rand() > 0.5 else t2

# 3. Interface Navigation Tabs
tab1, tab2 = st.tabs(["🔮 Smart Match Predictor", "🏆 Full Tournament Forecaster"])

# --- TAB 1: SMART MATCH PREDICTOR ---
with tab1:
    st.header("⚽ Group Stage Match Estimator")
    st.write("Select a specific tournament group to filter the selection pool dynamically.")
    
    # Smart Filtering Step 1: User chooses the Group
    selected_group = st.selectbox("Select Tournament Group", sorted(list(groups.keys())))
    
    # Smart Filtering Step 2: Extract only the 4 teams inside that specific group
    filtered_teams = groups[selected_group]
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team 1", filtered_teams, index=0)
    with col2:
        team_b = st.selectbox("Team 2", filtered_teams, index=1)

    if team_a == team_b:
        st.warning("Select two different group competitors to generate a prediction.")
    else:
        t1_w, tie_p, t2_w = calculate_probabilities(team_a, team_b)
        
        st.write("---")
        st.subheader(f"📊 {selected_group} Win Probabilities")
        
        st.write(f"**{team_a}** Win Chance: `{t1_w:.1f}%`")
        st.progress(int(t1_w))
        
        st.write(f"**Draw / Tie** Chance: `{tie_p:.1f}%`")
        st.progress(int(tie_p))
        
        st.write(f"**{team_b}** Win Chance: `{t2_w:.1f}%`")
        st.progress(int(t2_w))
        st.write("---")

# --- TAB 2: TOURNAMENT FORECASTER ---
with tab2:
    st.header("🏆 Tournament Simulation Engine")
    st.write("Process 1,000 randomized tournament brackets using official group assignments.")
    
    if st.button("🚀 Run Live Bracket Simulations", type="primary"):
        with st.spinner("Processing brackets..."):
            trophy_tracker = {team: 0 for team in teams_df['team_name']}
            TOTAL_SIMULATIONS = 1000

            for run in range(TOTAL_SIMULATIONS):
                group_winners, group_runners_up, third_place_pool = [], [], []
                for g_name in groups.keys():
                    standings = {team: {'Points': 0} for team in groups[g_name]}
                    for i in range(len(groups[g_name])):
                        for j in range(i + 1, len(groups[g_name])):
                            t_a, t_b = groups[g_name][i], groups[g_name][j]
                            p_a, p_b = play_match(t_a, t_b)
                            standings[t_a]['Points'] += p_a
                            standings[t_b]['Points'] += p_b
                    sorted_s = sorted(standings.items(), key=lambda x: x[1]['Points'], reverse=True)
                    group_winners.append(sorted_s[0][0])
                    group_runners_up.append(sorted_s[1][0])
                    third_place_pool.append({'team': sorted_s[2][0], 'Points': sorted_s[2][1]['Points']})

                third_df = pd.DataFrame(third_place_pool).sort_values(by='Points', ascending=False)
                best_8_third = third_df['team'].head(8).tolist()
                bracket = group_winners + group_runners_up + best_8_third
                
                while len(bracket) > 1:
                    next_layer = []
                    for i in range(0, len(bracket), 2):
                        next_layer.append(play_knockout(bracket[i], bracket[i+1]))
                    bracket = next_layer
                trophy_tracker[bracket[0]] += 1

            results_df = pd.DataFrame(list(trophy_tracker.items()), columns=['Country', 'Simulated Titles'])
            results_df['Win Prob (%)'] = (results_df['Simulated Titles'] / TOTAL_SIMULATIONS) * 100
            results_df = results_df.sort_values(by='Win Prob (%)', ascending=False).reset_index(drop=True)
            
            st.success("Calculations Complete!")
            st.dataframe(results_df[results_df['Win Prob (%)'] > 0.5], use_container_width=True)
