import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="World Cup 2026 Knockout Predictor",
    page_icon="⚽",
    layout="centered"
)

st.markdown("<h1 style='font-size: 2.5rem; white-space: nowrap;'>FIFA World Cup 2026™ Knockout Predictor</h1>", unsafe_allow_html=True)
st.markdown("##### Real-Time Knockout Analytics Powered by Live Tournament Data")
st.markdown("---")

# 2. DATA INGESTION

@st.cache_data(ttl=60)
def load_and_train_analytics_engine():
    try:
        import time
        cache_buster = f"?v={int(time.time())}"
        
        raw_results_url = f"https://raw.githubusercontent.com/Karshin12/World-Cup-2026-Knockout-Predictor/main/results.csv{cache_buster}"
        df = pd.read_csv(raw_results_url)
     
        shootout_df = None
        try:
            raw_shootouts_url = f"https://raw.githubusercontent.com/Karshin12/World-Cup-2026-Knockout-Predictor/main/shootouts.csv{cache_buster}"
            shootout_df = pd.read_csv(raw_shootouts_url)
            shootout_df['date'] = pd.to_datetime(shootout_df['date'], format='%d-%m-%Y', errors='coerce')
            shootout_df['home_team'] = shootout_df['home_team'].astype(str).str.strip()
            shootout_df['away_team'] = shootout_df['away_team'].astype(str).str.strip()
            shootout_df['winner'] = shootout_df['winner'].astype(str).str.strip()
        except Exception:
            pass 

        df['date'] = df['date'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        
        df = df[df['date'] >= pd.to_datetime('2000-01-01')].reset_index(drop=True)
        if shootout_df is not None:
            shootout_df = shootout_df[shootout_df['date'] >= pd.to_datetime('2000-01-01')].reset_index(drop=True)

        df['home_team'] = df['home_team'].astype(str).str.strip()
        df['away_team'] = df['away_team'].astype(str).str.strip()
        df['tournament'] = df['tournament'].astype(str).str.strip()
        
        name_mappings = {
            "United States": "USA", "Cabo Verde": "Cape Verde",
            "Congo DR": "DR Congo", "DR Congo": "DR Congo"
        }
        df['home_team'] = df['home_team'].replace(name_mappings)
        df['away_team'] = df['away_team'].replace(name_mappings)
        
        if shootout_df is not None:
            shootout_df['home_team'] = shootout_df['home_team'].replace(name_mappings)
            shootout_df['away_team'] = shootout_df['away_team'].replace(name_mappings)
            shootout_df['winner'] = shootout_df['winner'].replace(name_mappings)
            
        df = df.sort_values('date').reset_index(drop=True)

        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        completed_matches = df.dropna(subset=['home_score', 'away_score'])
        
        elo_ratings = {}
        k_factor = 32
        for _, row in completed_matches.iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            if h_team not in elo_ratings: elo_ratings[h_team] = 1500.0
            if a_team not in elo_ratings: elo_ratings[a_team] = 1500.0
            
            r_h, r_a = elo_ratings[h_team], elo_ratings[a_team]
            exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
            
            if row['home_score'] > row['away_score']: act_h = 1.0
            elif row['away_score'] > row['home_score']: act_h = 0.0
            else: act_h = 0.5
                
            elo_ratings[h_team] += k_factor * (act_h - exp_h)
            elo_ratings[a_team] += k_factor * (1 - act_h - (1 - exp_h))

        # --- GOALSCORERS INGESTION BLOCK ---
        scorers_df = None
        try:
            raw_scorers_url = f"https://raw.githubusercontent.com/Karshin12/World-Cup-2026-Knockout-Predictor/main/goalscorers.csv{cache_buster}"
            scorers_df = pd.read_csv(raw_scorers_url)
            scorers_df['date'] = pd.to_datetime(scorers_df['date'], format='%Y-%m-%d', errors='coerce')
            scorers_df['home_team'] = scorers_df['home_team'].astype(str).str.strip().replace(name_mappings)
            scorers_df['away_team'] = scorers_df['away_team'].astype(str).str.strip().replace(name_mappings)
            scorers_df['team'] = scorers_df['team'].astype(str).str.strip().replace(name_mappings)
            scorers_df['scorer'] = scorers_df['scorer'].astype(str).str.strip()
        except Exception:
            pass
            
        return df, elo_ratings, shootout_df, scorers_df
    except Exception as e:
        st.error(f"Analytics Pipeline Error: {e}")
        return None, None, None, None
        
raw_data, master_elo, shootout_data, scorers_data = load_and_train_analytics_engine()

def get_official_winner(team_a, team_b, df, shootout_df=None):
    if df is None: return None
    knockout_start = pd.to_datetime('2026-06-20')
    match = df[
        (df['tournament'] == 'FIFA World Cup') & 
        (df['date'] >= knockout_start) &  
        (((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
         ((df['home_team'] == team_b) & (df['away_team'] == team_a)))
    ]
    if not match.empty:
        latest = match.iloc[-1]
        
        if pd.isna(latest['home_score']) or pd.isna(latest['away_score']):
            return None
            
        if latest['home_score'] > latest['away_score']: return latest['home_team']
        elif latest['away_score'] > latest['home_score']: return latest['away_team']

        if latest['home_score'] == latest['away_score'] and shootout_df is not None:
            so_match = shootout_df[
                (((shootout_df['home_team'] == team_a) & (shootout_df['away_team'] == team_b)) | 
                 ((shootout_df['home_team'] == team_b) & (shootout_df['away_team'] == team_a)))
            ]
            if not so_match.empty:
                return so_match.iloc[-1]['winner']
                
    return None
    
def predict_match_analytics(team_a, team_b, elo_dict, df=None):
    if not elo_dict or team_a not in elo_dict or team_b not in elo_dict:
        return 50.0, 50.0
        
    r_a = elo_dict[team_a]
    r_b = elo_dict[team_b]
  
    h2h_adjustment = 0.0
    
    if df is not None:
        past_games = df[
            (df['home_score'].notna()) & (df['away_score'].notna()) & 
            (((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
             ((df['home_team'] == team_b) & (df['away_team'] == team_a)))
        ]
        
        if not past_games.empty:
            team_a_wins = 0
            team_b_wins = 0
            
            for _, row in past_games.iterrows():
                if row['home_score'] > row['away_score']:
                    winner = row['home_team']
                elif row['away_score'] > row['home_score']:
                    winner = row['away_team']
                else:
                    winner = None
                    
                if winner == team_a:
                    team_a_wins += 1
                elif winner == team_b:
                    team_b_wins += 1
            
            h2h_adjustment = (team_a_wins - team_b_wins) * 20.0

    prob_a = 1 / (1 + 10 ** ((r_b - (r_a + h2h_adjustment)) / 400))
    prob_b = 1 - prob_a
    
    return round(prob_a * 100, 1), round(prob_b * 100, 1)

def display_probability_bar(p_home, p_away, height=18):
    st.markdown(f"""
    <div style="background-color: #ef4444; border-radius: 6px; height: {height}px; width: 100%; overflow: hidden; display: flex; margin-top: 8px; margin-bottom: 8px;">
        <div style="background-color: #3b82f6; width: {int(p_home)}%; height: 100%;"></div>
    </div>
    """, unsafe_allow_html=True)

# SCOREBOARD DISPLAY
def display_custom_match_scoreboard(team_1, team_2, match_row, shootout_df, scorers_df):
    h_score = int(match_row['home_score'].values[0]) if hasattr(match_row['home_score'], 'values') else int(match_row['home_score'])
    a_score = int(match_row['away_score'].values[0]) if hasattr(match_row['away_score'], 'values') else int(match_row['away_score'])
    match_date = match_row['date'].values[0] if hasattr(match_row['date'], 'values') else match_row['date']
    row_home_team = match_row['home_team'].values[0] if hasattr(match_row['home_team'], 'values') else match_row['home_team']

    if row_home_team == team_1:
        left_score = h_score
        right_score = a_score
    else:
        left_score = a_score
        right_score = h_score

    left_pk_text = ""
    right_pk_text = ""
    if left_score == right_score and shootout_df is not None:
        so_match = shootout_df[
            (((shootout_df['home_team'] == team_1) & (shootout_df['away_team'] == team_2)) | 
             ((shootout_df['home_team'] == team_2) & (shootout_df['away_team'] == team_1)))
        ]
        if not so_match.empty:
            so_row = so_match.iloc[-1]
            if so_row['home_team'] == team_1:
                left_pk_text = f" ({int(so_row['home_pk_score'])})"
                right_pk_text = f" ({int(so_row['away_pk_score'])})"
            else:
                left_pk_text = f" ({int(so_row['away_pk_score'])})"
                right_pk_text = f" ({int(so_row['home_pk_score'])})"

    left_scorers, right_scorers = {}, {}
    if scorers_df is not None and not scorers_df.empty:
        goals = scorers_df[
            (scorers_df['date'] == match_date) & 
            (((scorers_df['home_team'] == team_1) & (scorers_df['away_team'] == team_2)) |
             ((scorers_df['home_team'] == team_2) & (scorers_df['away_team'] == team_1)))
        ]
        for _, row in goals.sort_values('minute').iterrows():
            suffix = ""
            if str(row['own_goal']).upper() in ['TRUE', '1']: suffix += " (OG)"
            if str(row['penalty']).upper() in ['TRUE', '1']: suffix += " (P)"
            time_str = f"{int(row['minute'])}'{suffix}"
            
            if row['team'] == team_1:
                left_scorers[row['scorer']] = left_scorers.get(row['scorer'], []) + [time_str]
            else:
                right_scorers[row['scorer']] = right_scorers.get(row['scorer'], []) + [time_str]

    left_goals = [f"{player} {', '.join(times)}" for player, times in left_scorers.items()]
    right_goals = [f"{player} {', '.join(times)}" for player, times in right_scorers.items()]

    home_scorers_text = "".join([f"<div style='margin-bottom: 2px;'>{g}</div>" for g in left_goals])
    away_scorers_text = "".join([f"<div style='margin-bottom: 2px;'>{g}</div>" for g in right_goals])

    t1_col1, t1_col2 = st.columns([8, 2])
    with t1_col1:
        st.markdown(f"### {team_1}")
    with t1_col2:
        st.markdown(f"<span style='font-size: 28px; font-weight: 900;'>{left_score}{left_pk_text}</span>", unsafe_allow_html=True)

    if left_goals:
        st.markdown(f"<div style='color: #bbbbbb; font-size: 15px; line-height: 1.3;'>{home_scorers_text}</div>", unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    t2_col1, t2_col2 = st.columns([8, 2])
    with t2_col1:
        st.markdown(f"### {team_2}")
    with t2_col2:
        st.markdown(f"<span style='font-size: 28px; font-weight: 900;'>{right_score}{right_pk_text}</span>", unsafe_allow_html=True)
        
    if right_goals:
        st.markdown(f"<div style='color: #bbbbbb; font-size: 15px; line-height: 1.3;'>{away_scorers_text}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 8px;'></div>", unsafe_allow_html=True)

def display_r32_match_row(match_num, team_a, team_b, df, elo_dict, shootout_df=None, scorers_df=None):
    st.markdown(f"#### 🏟️ Match {match_num}")
    
    winner = get_official_winner(team_a, team_b, df, shootout_df)
    
    if winner:
        knockout_start = pd.to_datetime('2026-06-20')
        match_row = df[
            (df['tournament'] == 'FIFA World Cup') & 
            (df['date'] >= knockout_start) &  
            (((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
             ((df['home_team'] == team_b) & (df['away_team'] == team_a)))
        ].iloc[-1]
        
        display_custom_match_scoreboard(team_a, team_b, match_row, shootout_df, scorers_df)
        
        winner_display = winner if int(match_row['home_score']) != int(match_row['away_score']) else f"{winner} (on penalties)"
        st.success(f"🏆 **{winner_display}** advances to the next round!")
    else:
        p_a, p_b = predict_match_analytics(team_a, team_b, elo_dict)
        st.markdown(f"**{team_a} vs {team_b}**")
        st.write(f"📊 {team_a}: **{p_a}%** | {team_b}: **{p_b}%**")
        display_probability_bar(p_a, p_b)
        
    st.markdown("---")


if raw_data is not None and master_elo is not None:
    
    st.header("Round of 32 Projections")
    st.markdown("---")

    display_r32_match_row(73, "South Africa", "Canada", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(74, "Brazil", "Japan", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(75, "Germany", "Paraguay", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(76, "Netherlands", "Morocco", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(77, "Ivory Coast", "Norway", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(78, "France", "Sweden", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(79, "Mexico", "Ecuador", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(80, "England", "DR Congo", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(81, "Belgium", "Senegal", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(82, "USA", "Bosnia and Herzegovina", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(83, "Spain", "Austria", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(84, "Portugal", "Croatia", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(85, "Switzerland", "Algeria", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(86, "Australia", "Egypt", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(87, "Argentina", "Cape Verde", raw_data, master_elo, shootout_data, scorers_data)
    display_r32_match_row(88, "Colombia", "Ghana", raw_data, master_elo, shootout_data, scorers_data)

    st.header("Round of 16 Projections")
    st.markdown("---")

    r32_w1 = get_official_winner("South Africa", "Canada", raw_data, shootout_data)
    r32_w2 = get_official_winner("Netherlands", "Morocco", raw_data, shootout_data)
    t1_a = r32_w1 if r32_w1 else "Canada"
    t1_b = r32_w2 if r32_w2 else "Morocco"

    r32_w3 = get_official_winner("Germany", "Paraguay", raw_data, shootout_data)
    r32_w4 = get_official_winner("France", "Sweden", raw_data, shootout_data)
    t2_a = r32_w3 if r32_w3 else "Paraguay"
    t2_b = r32_w4 if r32_w4 else "France"

    r32_w5 = get_official_winner("Brazil", "Japan", raw_data, shootout_data)
    r32_w6 = get_official_winner("Ivory Coast", "Norway", raw_data, shootout_data)
    t3_a = r32_w5 if r32_w5 else "Brazil"
    t3_b = r32_w6 if r32_w6 else "Norway"

    r32_w7 = get_official_winner("Mexico", "Ecuador", raw_data, shootout_data)
    r32_w8 = get_official_winner("England", "DR Congo", raw_data, shootout_data)
    t4_a = r32_w7 if r32_w7 else "Mexico"
    t4_b = r32_w8 if r32_w8 else "England"

    r32_w9 = get_official_winner("Portugal", "Croatia", raw_data, shootout_data)
    r32_w10 = get_official_winner("Spain", "Austria", raw_data, shootout_data)
    t5_a = r32_w9 if r32_w9 else "Portugal"
    t5_b = r32_w10 if r32_w10 else "Spain"

    r32_w11 = get_official_winner("USA", "Bosnia and Herzegovina", raw_data, shootout_data)
    r32_w12 = get_official_winner("Belgium", "Senegal", raw_data, shootout_data)
    t6_a = r32_w11 if r32_w11 else "USA"
    t6_b = r32_w12 if r32_w12 else "Belgium"

    r32_w13 = get_official_winner("Argentina", "Cape Verde", raw_data, shootout_data)
    r32_w14 = get_official_winner("Australia", "Egypt", raw_data, shootout_data)
    t7_a = r32_w13 if r32_w13 else "Argentina"
    t7_b = r32_w14 if r32_w14 else "Egypt"

    r32_w15 = get_official_winner("Switzerland", "Algeria", raw_data, shootout_data)
    r32_w16 = get_official_winner("Colombia", "Ghana", raw_data, shootout_data)
    t8_a = r32_w15 if r32_w15 else "Switzerland"
    t8_b = r32_w16 if r32_w16 else "Colombia"

    def render_r16_row(title, team_1, team_2):
        st.markdown(f"### {title}")
        winner = get_official_winner(team_1, team_2, raw_data, shootout_data)
        if winner:
            knockout_start = pd.to_datetime('2026-06-20')
            match_row = raw_data[
                (raw_data['tournament'] == 'FIFA World Cup') & 
                (raw_data['date'] >= knockout_start) &  
                (((raw_data['home_team'] == team_1) & (raw_data['away_team'] == team_2)) | 
                 ((raw_data['home_team'] == team_2) & (raw_data['away_team'] == team_1)))
            ].iloc[-1]
            display_custom_match_scoreboard(team_1, team_2, match_row, shootout_data, scorers_data)
            return winner
        else:
            p1, p2 = predict_match_analytics(team_1, team_2, master_elo, raw_data)
            st.markdown(f"##### **{team_1} vs {team_2}**")
            st.write(f"🔮 **Prediction:** {team_1}: **{p1}%** | {team_2}: **{p2}%**")
            display_probability_bar(p1, p2)
            return team_1 if p1 >= p2 else team_2

    ma_w = render_r16_row("Round of 16: Match A", t1_a, t1_b)
    mb_w = render_r16_row("Round of 16: Match B", t2_a, t2_b)
    mc_w = render_r16_row("Round of 16: Match C", t3_a, t3_b)
    md_w = render_r16_row("Round of 16: Match D", t4_a, t4_b)
    me_w = render_r16_row("Round of 16: Match E", t5_a, t5_b)
    mf_w = render_r16_row("Round of 16: Match F", t6_a, t6_b)
    mg_w = render_r16_row("Round of 16: Match G", t7_a, t7_b)
    mh_w = render_r16_row("Round of 16: Match H", t8_a, t8_b)
    st.markdown("---")

    # QUARTERFINALS
    st.header("Quarterfinal Projections")
    st.markdown("---")

    def render_clean_interactive_qf(title, parent_w1, parent_w2, opt_list1, opt_list2, m1_label, m2_label, key_suffix):
        st.markdown(f"### {title}")
        
        is_parent1_official = get_official_winner(opt_list1[0], opt_list1[1], raw_data, shootout_data) is not None
        is_parent2_official = get_official_winner(opt_list2[0], opt_list2[1], raw_data, shootout_data) is not None
        
        lbl1 = f"Winner from {m1_label}" if is_parent1_official else f"Pick from {m1_label}"
        lbl2 = f"Winner from {m2_label}" if is_parent2_official else f"Pick from {m2_label}"
        
        col1, col2 = st.columns(2)
        with col1:
            team_1 = parent_w1 if is_parent1_official else st.selectbox(lbl1, opt_list1, key=f"qf_a_{key_suffix}")
            if is_parent1_official:
                st.markdown(f"**{lbl1}:** {team_1}")
        with col2:
            team_2 = parent_w2 if is_parent2_official else st.selectbox(lbl2, opt_list2, key=f"qf_b_{key_suffix}")
            if is_parent2_official:
                st.markdown(f"**{lbl2}:** {team_2}")
            
        winner = get_official_winner(team_1, team_2, raw_data, shootout_data)
        if winner:
            knockout_start = pd.to_datetime('2026-06-20')
            match_row = raw_data[
                (raw_data['tournament'] == 'FIFA World Cup') & 
                (raw_data['date'] >= knockout_start) &  
                (((raw_data['home_team'] == team_1) & (raw_data['away_team'] == team_2)) | 
                 ((raw_data['home_team'] == team_2) & (raw_data['away_team'] == team_1)))
            ].iloc[-1]
            display_custom_match_scoreboard(team_1, team_2, match_row, shootout_data, scorers_data)
            return winner
        else:
            p1, p2 = predict_match_analytics(team_1, team_2, master_elo, raw_data)
            st.markdown(f"##### **{team_1} vs {team_2}**")
            st.write(f"🔮 **Prediction:** {team_1}: **{p1}%** | {team_2}: **{p2}%**")
            display_probability_bar(p1, p2)
            st.markdown("---")
            return team_1 if p1 >= p2 else team_2

    q1_winner = render_clean_interactive_qf("Quarterfinal 1", ma_w, mb_w, [t1_a, t1_b], [t2_a, t2_b], "Match A", "Match B", "q1")
    q2_winner = render_clean_interactive_qf("Quarterfinal 2", me_w, mf_w, [t5_a, t5_b], [t6_a, t6_b], "Match E", "Match F", "q2")
    q3_winner = render_clean_interactive_qf("Quarterfinal 3", mc_w, md_w, [t3_a, t3_b], [t4_a, t4_b], "Match C", "Match D", "q3")
    q4_winner = render_clean_interactive_qf("Quarterfinal 4", mg_w, mh_w, [t7_a, t7_b], [t8_a, t8_b], "Match G", "Match H", "q4")

    # SEMIFINALS
    st.header("Semifinals & Third Place Playoff")
    st.markdown("---")

    # --- Semifinal 1 ---
    st.markdown("### Semifinal 1")
    is_q1_official = get_official_winner(ma_w, mb_w, raw_data, shootout_data) is not None
    is_q2_official = get_official_winner(me_w, mf_w, raw_data, shootout_data) is not None
    
    lbl_sf1_a = "Winner from QF1" if is_q1_official else "Pick from QF1"
    lbl_sf1_b = "Winner from QF2" if is_q2_official else "Pick from QF2"
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        s1_a = q1_winner if is_q1_official else st.selectbox(lbl_sf1_a, [ma_w, mb_w], key="sf1_a_sel")
        if is_q1_official:
            st.markdown(f"**{lbl_sf1_a}:** {s1_a}")
    with col_s2:
        s1_b = q2_winner if is_q2_official else st.selectbox(lbl_sf1_b, [me_w, mf_w], key="sf1_b_sel")
        if is_q2_official:
            st.markdown(f"**{lbl_sf1_b}:** {s1_b}")

    sf1_winner = get_official_winner(s1_a, s1_b, raw_data, shootout_data)
    if sf1_winner:
        knockout_start = pd.to_datetime('2026-06-20')
        match_row = raw_data[
            (raw_data['tournament'] == 'FIFA World Cup') & 
            (raw_data['date'] >= knockout_start) &  
            (((raw_data['home_team'] == s1_a) & (raw_data['away_team'] == s1_b)) | 
             ((raw_data['home_team'] == s1_b) & (raw_data['away_team'] == s1_a)))
        ].iloc[-1]
        display_custom_match_scoreboard(s1_a, s1_b, match_row, shootout_data, scorers_data)
        f_a = sf1_winner
        sf1_loser = s1_b if sf1_winner == s1_a else s1_a
    else:
        ps1, ps2 = predict_match_analytics(s1_a, s1_b, master_elo, raw_data)
        st.markdown(f"#### {s1_a} vs {s1_b}")
        st.write(f" 📊 **Prediction:** {s1_a}: **{ps1}%** | {s1_b}: **{ps2}%**")
        display_probability_bar(ps1, ps2)
        
        predicted_winner_sf1 = s1_a if ps1 >= ps2 else s1_b
        sf1_user_pick = st.selectbox(f"Who makes it to the Final?", [s1_a, s1_b], index=[s1_a, s1_b].index(predicted_winner_sf1), key="sf1_user_pick_winner")
        
        f_a = sf1_user_pick
        sf1_loser = s1_b if f_a == s1_a else s1_a
    st.markdown("---")

    # --- Semifinal 2 ---
    st.markdown("### Semifinal 2")
    is_q3_official = get_official_winner(mc_w, md_w, raw_data, shootout_data) is not None
    is_q4_official = get_official_winner(mg_w, mh_w, raw_data, shootout_data) is not None
    
    lbl_sf2_a = "Winner from QF3" if is_q3_official else "Pick from QF3"
    lbl_sf2_b = "Winner from QF4" if is_q4_official else "Pick from QF4"
    
    col_s3, col_s4 = st.columns(2)
    with col_s3:
        s2_a = q3_winner if is_q3_official else st.selectbox(lbl_sf2_a, [mc_w, md_w], key="sf2_a_sel")
        if is_q3_official:
            st.markdown(f"**{lbl_sf2_a}:** {s2_a}")
    with col_s4:
        s2_b = q4_winner if is_q4_official else st.selectbox(lbl_sf2_b, [mg_w, mh_w], key="sf2_b_sel")
        if is_q4_official:
            st.markdown(f"**{lbl_sf2_b}:** {s2_b}")

    sf2_winner = get_official_winner(s2_a, s2_b, raw_data, shootout_data)
    if sf2_winner:
        knockout_start = pd.to_datetime('2026-06-20')
        match_row = raw_data[
            (raw_data['tournament'] == 'FIFA World Cup') & 
            (raw_data['date'] >= knockout_start) &  
            (((raw_data['home_team'] == s2_a) & (raw_data['away_team'] == s2_b)) | 
             ((raw_data['home_team'] == s2_b) & (raw_data['away_team'] == s2_a)))
        ].iloc[-1]
        display_custom_match_scoreboard(s2_a, s2_b, match_row, shootout_data, scorers_data)
        f_b = sf2_winner
        sf2_loser = s2_b if sf2_winner == s2_a else s2_a
    else:
        ps3, ps4 = predict_match_analytics(s2_a, s2_b, master_elo, raw_data)
        st.markdown(f"#### {s2_a} vs {s2_b}")
        st.write(f"📊 **Prediction:** {s2_a}: **{ps3}%** | {s2_b}: **{ps4}%**")
        display_probability_bar(ps3, ps4)
        
        predicted_winner_sf2 = s2_a if ps3 >= ps4 else s2_b
        sf2_user_pick = st.selectbox(f"Who makes it to the Final?", [s2_a, s2_b], index=[s2_a, s2_b].index(predicted_winner_sf2), key="sf2_user_pick_winner")
        
        f_b = sf2_user_pick
        sf2_loser = s2_b if f_b == s2_a else s2_a
    st.markdown("---")

    # --- Third-Place Playoff ---
    st.markdown("### 🥉 Third Place Playoff")
    bronze_winner = get_official_winner(sf1_loser, sf2_loser, raw_data, shootout_data)
    if bronze_winner:
        knockout_start = pd.to_datetime('2026-06-20')
        match_row = raw_data[
            (raw_data['tournament'] == 'FIFA World Cup') & 
            (raw_data['date'] >= knockout_start) &  
            (((raw_data['home_team'] == sf1_loser) & (raw_data['away_team'] == sf2_loser)) | 
             ((raw_data['home_team'] == sf2_loser) & (raw_data['away_team'] == sf1_loser)))
        ].iloc[-1]
        display_custom_match_scoreboard(sf1_loser, sf2_loser, match_row, shootout_data, scorers_data)
    else:
        p_tp1, p_tp2 = predict_match_analytics(sf1_loser, sf2_loser, master_elo, raw_data)
        st.markdown(f"#### {sf1_loser} vs {sf2_loser}")
        st.write(f"📊 **Prediction:** {sf1_loser}: **{p_tp1}%** | {sf2_loser}: **{p_tp2}%**")
        display_probability_bar(p_tp1, p_tp2)
    st.markdown("---")

    # --- The Grand Final Showdown ---
    st.header("FIFA World Cup 2026 Final")
    st.markdown("---")
    st.markdown("### 🥇 The Final Showdown")
    
    champion = get_official_winner(f_a, f_b, raw_data, shootout_data)
    if champion:
        knockout_start = pd.to_datetime('2026-06-20')
        match_row = raw_data[
            (raw_data['tournament'] == 'FIFA World Cup') & 
            (raw_data['date'] >= knockout_start) &  
            (((raw_data['home_team'] == f_a) & (raw_data['away_team'] == f_b)) | 
             ((raw_data['home_team'] == f_b) & (raw_data['away_team'] == f_a)))
        ].iloc[-1]
        display_custom_match_scoreboard(f_a, f_b, match_row, shootout_data, scorers_data)
        st.balloons()
        st.success(f"🏆 **CHAMPIONS:** {champion} has won the 2026 FIFA World Cup!")
    else:
        st.markdown(f"### 🏆 {f_a} vs {f_b}")
        final_1, final_2 = predict_match_analytics(f_a, f_b, master_elo, raw_data)
        st.write(f"📊 {f_a}: **{final_1}%** | {f_b}: **{final_2}%**")
        display_probability_bar(final_1, final_2, height=28)
        
        higher_team = f_a if final_1 >= final_2 else f_b
        lower_team = f_b if final_1 >= final_2 else f_a
        st.info(f"🔮 **{higher_team}** is projected to edge out **{lower_team}** to capture the World Cup!")