import streamlit as st
import pandas as pd
import numpy as np

# Set up page styling for mobile-responsive look
st.set_page_config(page_title="2026 World Cup Predictor", page_icon="🏆", layout="centered")

st.title("🏆 World Cup 2026 Live Forecaster")
st.write("Click the button below to run 1,000 Monte Carlo simulations using real-world ELO ratings.")

# Load backend data metrics safely
@st.cache_data
def load_data():
    return pd.read_csv("team_stats.csv")

teams_df = load_data()
elo_lookup = pd.Series(teams_df.elo_rating.values, index=teams_df.team_name).to_dict()
groups = teams_df.groupby('group_assignment')['team_name'].apply(list).to_dict()

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

# UI Interaction Button
if st.button("🚀 Run Live Tournament Simulations", type="primary"):
    with st.spinner("Simulating thousands of dynamic brackets..."):
        trophy_tracker = {team: 0 for team in teams_df['team_name']}
        TOTAL_SIMULATIONS = 1000

        for run in range(TOTAL_SIMULATIONS):
            group_winners, group_runners_up, third_place_pool = [], [], []
            for g_name in groups.keys():
                standings = {team: {'Points': 0, 'GD': 0} for team in groups[g_name]}
                for i in range(len(groups[g_name])):
                    for j in range(i + 1, len(groups[g_name])):
                        team_a, team_b = groups[g_name][i], groups[g_name][j]
                        p_a, p_b = play_match(team_a, team_b)
                        standings[team_a]['Points'] += p_a
                        standings[team_b]['Points'] += p_b
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

        # Format and present results cleanly as an app interface
        results_df = pd.DataFrame(list(trophy_tracker.items()), columns=['Country Team', 'Simulated Trophies'])
        results_df['Win Probability (%)'] = (results_df['Simulated Trophies'] / TOTAL_SIMULATIONS) * 100
        results_df = results_df.sort_values(by='Win Probability (%)', ascending=False).reset_index(drop=True)
        
        st.success("Analysis Complete!")
        st.dataframe(results_df[results_df['Win Probability (%)'] > 0.5], use_container_width=True)
