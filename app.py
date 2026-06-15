import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import json

st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

# 1. Load baseline team statistics safely from the root directory
@st.cache_data
def load_data():
    return pd.read_csv("team_stats.csv")

teams_df = load_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
groups = teams_df.groupby('group_assignment')['team_name'].apply(list).to_dict()

# Match simulation math functions
def calculate_probabilities(t1, t2):
    elo_1, elo_2 = elo_lookup.get(t1, 1500), elo_lookup.get(t2, 1500)
    win_prob = 1 / (1 + 10 ** (-(elo_1 - elo_2) / 400))
    # Standard distribution for football probabilities including ties
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

# Create Navigation Tabs at the top of the App Screen
tab1, tab2 = st.tabs(["🔮 Predict Next Matches", "🏆 Overall Tournament Winner"])

# --- TAB 1: INDIVIDUAL MATCH ESTIMATOR ---
with tab1:
    st.header("⚽ Next Match Predictor")
    st.write("Select any two teams scheduled to play next to see the AI's calculation of the outcome probabilities.")
    
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Home / Team 1", sorted(teams_df['team_name'].tolist()), index=20) # Defaults to France
    with col2:
        team_b = st.selectbox("Away / Team 2", sorted(teams_df['team_name'].tolist()), index=39) # Defaults to Senegal

    if team_a == team_b:
        st.warning("Please choose two different teams to estimate a match outcome.")
    else:
        t1_w, tie_p, t2_w = calculate_probabilities(team_a, team_b)
        
        st.write("---")
        st.subheader("📊 Mathematical Probability Breakdown")
        
        # Display clean progress metrics for the selected match
        st.write(f"**{team_a}** Win Chance: `{t1_w:.1f}%`")
        st.progress(int(t1_w))
        
        st.write(f"**Draw / Tie** Chance: `{tie_p:.1f}%`")
        st.progress(int(tie_p))
        
        st.write(f"**{team_b}** Win Chance: `{t2_w:.1f}%`")
        st.progress(int(t2_w))
        st.write("---")

# --- TAB 2: TOURNAMENT WINNER TRACKER ---
with tab2:
    st.header("🏆 Tournament Simulation Engine")
    st.write("Simulate 1,000 full tournament brackets based on live data updates.")
    
    if st.button("🚀 Calculate Live Championship Odds", type="primary"):
        with st.spinner("Processing brackets..."):
            trophy_tracker = {team: 0 for team in teams_df['team_name']}
            TOTAL_SIMULATIONS = 1000

            for run in range(TOTAL_SIMULATIONS):
                group_winners, group_runners_up, third_place_pool = [], [], []
                for g_name in groups.keys():
                    standings = {team: {'Points': 0, 'GD': 0} for team in groups[g_name]}
                    for i in range(len(groups[g_name])):
                        for j in range(i + 1, len(groups[g_name])):
                            team_a_sim, team_b_sim = groups[g_name][i], groups[g_name][j]
                            p_a, p_b = play_match(team_a_sim, team_b_sim)
                            standings[team_a_sim]['Points'] += p_a
                            standings[team_b_sim]['Points'] += p_b
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
            results_df['Live Win Prob (%)'] = (results_df['Simulated Titles'] / TOTAL_SIMULATIONS) * 100
            results_df = results_df.sort_values(by='Live Win Prob (%)', ascending=False).reset_index(drop=True)
            
            st.success("Calculations Complete!")
            st.dataframe(results_df[results_df['Live Win Prob (%)'] > 0.5], use_container_width=True)
