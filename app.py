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
        # Base results dataset
        raw_results_url = "https://raw.githubusercontent.com/Karshin12/World-Cup-2026-Knockout-Predictor/main/results.csv"
        df = pd.read_csv(raw_results_url)
     
        shootout_df = None
        try:
            raw_shootouts_url = "https://raw.githubusercontent.com/Karshin12/World-Cup-2026-Knockout-Predictor/main/shootouts.csv"
            shootout_df = pd.read_csv(raw_shootouts_url)
            shootout_df['date'] = pd.to_datetime(shootout_df['date'], format='mixed', errors='coerce')
            shootout_df['home_team'] = shootout_df['home_team'].astype(str).str.strip()
            shootout_df['away_team'] = shootout_df['away_team'].astype(str).str.strip()
            shootout_df['winner'] = shootout_df['winner'].astype(str).str.strip()
        except Exception:
            pass # Keep rolling if shootouts file hasn't been created yet

        df['date'] = df['date'].astype(str).str.strip()
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['date'])
        
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
        
        # --- TRAIN ELO RATINGS ---
        elo_ratings = {}
        k_factor = 32
        for _, row in df.iterrows():
            h_team = row['home_team']
            a_team = row['away_team']
            if h_team not in elo_ratings: elo_ratings[h_team] = 1500.0
            if a_team not in elo_ratings: elo_ratings[a_team] = 1500.0
            
            r_h, r_a = elo_ratings[h_team], elo_ratings[a_team]
            exp_h = 1 / (1 + 10 ** ((r_a - r_h) / 400))
            
            # Football Elo rule: matches decided via PKs are officially rated as draws (0.5)
            if row['home_score'] > row['away_score']: act_h = 1.0
            elif row['away_score'] > row['home_score']: act_h = 0.0
            else: act_h = 0.5
                
            elo_ratings[h_team] += k_factor * (act_h - exp_h)
            elo_ratings[a_team] += k_factor * (1 - act_h - (1 - exp_h))
            
        return df, elo_ratings, shootout_df
    except Exception as e:
        st.error(f"Analytics Pipeline Error: {e}")
        return None, None, None
        
raw_data, master_elo, shootout_data = load_and_train_analytics_engine()

def get_official_winner(team_a, team_b, df, shootout_df=None):
    if df is None: return None
    match = df[
        (df['tournament'] == 'FIFA World Cup') & 
        (df['date'] >= '2026-06-20') &  
        (((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
         ((df['home_team'] == team_b) & (df['away_team'] == team_a)))
    ]
    if not match.empty:
        latest = match.iloc[-1]
        
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
    
def predict_match_analytics(team_a, team_b, elo_dict):
    """
    Industry-Level Logistic Projections using learned features.
    Outputs clean win-probability splits based on current latent variables.
    """
    if not elo_dict or team_a not in elo_dict or team_b not in elo_dict:
        return 50.0, 50.0
        
    r_a = elo_dict[team_a]
    r_b = elo_dict[team_b]

    prob_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    prob_b = 1 - prob_a
    
    return round(prob_a * 100, 1), round(prob_b * 100, 1)

def display_probability_bar(p_home, p_away, height=18):
    """Renders a unified version-safe dual color probability bar: Blue (Home) and Red (Away)."""
    st.markdown(f"""
    <div style="background-color: #ef4444; border-radius: 6px; height: {height}px; width: 100%; overflow: hidden; display: flex; margin-top: 8px; margin-bottom: 8px;">
        <div style="background-color: #3b82f6; width: {int(p_home)}%; height: 100%;"></div>
    </div>
    """, unsafe_allow_html=True)

def display_r32_match_row(match_num, team_a, team_b, df, elo_dict):
    st.markdown(f"#### 🏟️ Match {match_num}")
    
    # 1. Scan for an official winner
    winner = get_official_winner(team_a, team_b, df)
    
    if winner:
        # 2. Extract the exact match row from the data to pull scores
        match_row = df[
            (df['tournament'] == 'FIFA World Cup') & 
            (df['date'] >= '2026-06-20') &  
            (((df['home_team'] == team_a) & (df['away_team'] == team_b)) | 
             ((df['home_team'] == team_b) & (df['away_team'] == team_a)))
        ].iloc[-1]
        
        # Keep original team placement order for the matchup header
        st.markdown(f"**{team_a} vs {team_b}**")
        
        # 3. Format and display the explicit scoreline line
        st.success(
            f"🏁 **Result:** {match_row['home_team']} {int(match_row['home_score'])} - "
            f"{int(match_row['away_score'])} {match_row['away_team']}\n\n"
            f"**{winner}** advances to the next round!"
        )
    else:
        # Fallback projection layout for unplayed matchups
        p_a, p_b = predict_match_analytics(team_a, team_b, elo_dict)
        st.markdown(f"**{team_a} vs {team_b}**")
        st.write(f"📊 {team_a}: **{p_a}%** | {team_b}: **{p_b}%**")
        display_probability_bar(p_a, p_b)
        
    st.markdown("---")

# 3. INTERACTIVE PROCESSOR LAYER

if raw_data is not None and master_elo is not None:
    
    # STAGE 1: ROUND OF 32 FIXTURES

    st.header("Round of 32 Projections")
    st.markdown("---")

    display_r32_match_row(73, "South Africa", "Canada", raw_data, master_elo)
    display_r32_match_row(74, "Brazil", "Japan", raw_data, master_elo)
    display_r32_match_row(75, "Germany", "Paraguay", raw_data, master_elo)
    display_r32_match_row(76, "Netherlands", "Morocco", raw_data, master_elo)
    display_r32_match_row(77, "Ivory Coast", "Norway", raw_data, master_elo)
    display_r32_match_row(78, "France", "Sweden", raw_data, master_elo)
    display_r32_match_row(79, "Mexico", "Ecuador", raw_data, master_elo)
    display_r32_match_row(80, "England", "DR Congo", raw_data, master_elo)
    display_r32_match_row(81, "Belgium", "Senegal", raw_data, master_elo)
    display_r32_match_row(82, "USA", "Bosnia and Herzegovina", raw_data, master_elo)
    display_r32_match_row(83, "Spain", "Austria", raw_data, master_elo)
    display_r32_match_row(84, "Portugal", "Croatia", raw_data, master_elo)
    display_r32_match_row(85, "Switzerland", "Algeria", raw_data, master_elo)
    display_r32_match_row(86, "Australia", "Egypt", raw_data, master_elo)
    display_r32_match_row(87, "Argentina", "Cape Verde", raw_data, master_elo)
    display_r32_match_row(88, "Colombia", "Ghana", raw_data, master_elo)

    # STAGE 2: ROUND OF 16 CONFIGURATION
    
    st.header("Round of 16 Projections")
    st.markdown("---")

    r16_picks = {}

    # Match A
    st.markdown("### Round of 16: Match A")
    w73 = get_official_winner("South Africa", "Canada", raw_data)
    if w73: t1_a = st.selectbox("Advanced from South Africa vs Canada:", options=[w73], key="r16_a_a_locked")
    else: t1_a = st.selectbox("Who advances from South Africa vs Canada?", options=["South Africa", "Canada"], index=1, key="r16_a_a_open")
        
    w74 = get_official_winner("Brazil", "Japan", raw_data)
    if w74: t1_b = st.selectbox("Advanced from Brazil vs Japan:", options=[w74], key="r16_a_b_locked")
    else: t1_b = st.selectbox("Who advances from Brazil vs Japan?", options=["Brazil", "Japan"], key="r16_a_b_open")
        
    pa1, pa2 = predict_match_analytics(t1_a, t1_b, master_elo)
    st.write(f"🔮 **Prediction:** {t1_a}: **{pa1}%** | {t1_b}: **{pa2}%**")
    display_probability_bar(pa1, pa2)
    r16_picks["ma_w"] = st.selectbox("Who wins Match A?", options=[t1_a, t1_b], key="winner_match_a")
    st.markdown("---")

    # Match B
    st.markdown("### Round of 16: Match B")
    w75 = get_official_winner("Germany", "Paraguay", raw_data)
    if w75: t2_a = st.selectbox("Advanced from Germany vs Paraguay:", options=[w75], key="r16_b_a_locked")
    else: t2_a = st.selectbox("Who advances from Germany vs Paraguay?", options=["Germany", "Paraguay"], key="r16_b_a_open")
        
    w76 = get_official_winner("Netherlands", "Morocco", raw_data)
    if w76: t2_b = st.selectbox("Advanced from Netherlands vs Morocco:", options=[w76], key="r16_b_b_locked")
    else: t2_b = st.selectbox("Who advances from Netherlands vs Morocco?", options=["Netherlands", "Morocco"], key="r16_b_b_open")
        
    pb1, pb2 = predict_match_analytics(t2_a, t2_b, master_elo)
    st.write(f"🔮 **Prediction:** {t2_a}: **{pb1}%** | {t2_b}: **{pb2}%**")
    display_probability_bar(pb1, pb2)
    r16_picks["mb_w"] = st.selectbox("Who wins Match B?", options=[t2_a, t2_b], key="winner_match_b")
    st.markdown("---")

    # Match C
    st.markdown("### Round of 16: Match C")
    w77 = get_official_winner("Ivory Coast", "Norway", raw_data)
    if w77: t3_a = st.selectbox("Advanced from Ivory Coast vs Norway:", options=[w77], key="r16_c_a_locked")
    else: t3_a = st.selectbox("Who advances from Ivory Coast vs Norway?", options=["Ivory Coast", "Norway"], key="r16_c_a_open")
        
    w78 = get_official_winner("France", "Sweden", raw_data)
    if w78: t3_b = st.selectbox("Advanced from France vs Sweden:", options=[w78], key="r16_c_b_locked")
    else: t3_b = st.selectbox("Who advances from France vs Sweden?", options=["France", "Sweden"], key="ro16_c_b_open")
        
    pc1, pc2 = predict_match_analytics(t3_a, t3_b, master_elo)
    st.write(f"🔮 **Prediction:** {t3_a}: **{pc1}%** | {t3_b}: **{pc2}%**")
    display_probability_bar(pc1, pc2)
    r16_picks["mc_w"] = st.selectbox("Who wins Match C?", options=[t3_a, t3_b], key="winner_match_c")
    st.markdown("---")

    # Match D
    st.markdown("### Round of 16: Match D")
    w79 = get_official_winner("Mexico", "Ecuador", raw_data)
    if w79: t4_a = st.selectbox("Advanced from Mexico vs Ecuador:", options=[w79], key="r16_d_a_locked")
    else: t4_a = st.selectbox("Who advances from Mexico vs Ecuador?", options=["Mexico", "Ecuador"], key="r16_d_a_open")
        
    w80 = get_official_winner("England", "DR Congo", raw_data)
    if w80: t4_b = st.selectbox("Advanced from England vs DR Congo:", options=[w80], key="r16_d_b_locked")
    else: t4_b = st.selectbox("Who advances from England vs DR Congo?", options=["England", "DR Congo"], key="r16_d_b_open")
        
    pd1, pd2 = predict_match_analytics(t4_a, t4_b, master_elo)
    st.write(f"🔮 **Prediction:** {t4_a}: **{pd1}%** | {t4_b}: **{pd2}%**")
    display_probability_bar(pd1, pd2)
    r16_picks["md_w"] = st.selectbox("Who wins Match D?", options=[t4_a, t4_b], key="winner_match_d")
    st.markdown("---")

    # Match E
    st.markdown("### Round of 16: Match E")
    w81 = get_official_winner("Belgium", "Senegal", raw_data)
    if w81: t5_a = st.selectbox("Advanced from Belgium vs Senegal:", options=[w81], key="r16_e_a_locked")
    else: t5_a = st.selectbox("Who advances from Belgium vs Senegal?", options=["Belgium", "Senegal"], key="r16_e_a_open")
        
    w82 = get_official_winner("USA", "Bosnia and Herzegovina", raw_data)
    if w82: t5_b = st.selectbox("Advanced from USA vs Bosnia and Herzegovina:", options=[w82], key="r16_e_b_locked")
    else: t5_b = st.selectbox("Who advances from USA vs Bosnia and Herzegovina?", options=["USA", "Bosnia and Herzegovina"], key="r16_e_b_open")
        
    pe1, pe2 = predict_match_analytics(t5_a, t5_b, master_elo)
    st.write(f"🔮 **Prediction:** {t5_a}: **{pe1}%** | {t5_b}: **{pe2}%**")
    display_probability_bar(pe1, pe2)
    r16_picks["me_w"] = st.selectbox("Who wins Match E?", options=[t5_a, t5_b], key="winner_match_e")
    st.markdown("---")

    # Match F
    st.markdown("### Round of 16: Match F")
    w83 = get_official_winner("Spain", "Austria", raw_data)
    if w83: t6_a = st.selectbox("Advanced from Spain vs Austria:", options=[w83], key="r16_f_a_locked")
    else: t6_a = st.selectbox("Who advances from Spain vs Austria?", options=["Spain", "Austria"], key="r16_f_a_open")
        
    w84 = get_official_winner("Portugal", "Croatia", raw_data)
    if w84: t6_b = st.selectbox("Advanced from Portugal vs Croatia:", options=[w84], key="r16_f_b_locked")
    else: t6_b = st.selectbox("Who advances from Portugal vs Croatia?", options=["Portugal", "Croatia"], key="r16_f_b_open")
        
    pf1, pf2 = predict_match_analytics(t6_a, t6_b, master_elo)
    st.write(f"🔮 **Prediction:** {t6_a}: **{pf1}%** | {t6_b}: **{pf2}%**")
    display_probability_bar(pf1, pf2)
    r16_picks["mf_w"] = st.selectbox("Who wins Match F?", options=[t6_a, t6_b], key="winner_match_f")
    st.markdown("---")

    # Match G
    st.markdown("### Round of 16: Match G")
    w85 = get_official_winner("Switzerland", "Algeria", raw_data)
    if w85: t7_a = st.selectbox("Advanced from Switzerland vs Algeria:", options=[w85], key="r16_g_a_locked")
    else: t7_a = st.selectbox("Who advances from Switzerland vs Algeria?", options=["Switzerland", "Algeria"], key="r16_g_a_open")
        
    w86 = get_official_winner("Australia", "Egypt", raw_data)
    if w86: t7_b = st.selectbox("Advanced from Australia vs Egypt:", options=[w86], key="r16_g_b_locked")
    else: t7_b = st.selectbox("Who advances from Australia vs Egypt?", options=["Australia", "Egypt"], key="r16_g_b_open")
        
    pg1, pg2 = predict_match_analytics(t7_a, t7_b, master_elo)
    st.write(f"🔮 **Prediction:** {t7_a}: **{pg1}%** | {t7_b}: **{pg2}%**")
    display_probability_bar(pg1, pg2)
    r16_picks["mg_w"] = st.selectbox("Who wins Match G?", options=[t7_a, t7_b], key="sel_w_mg")
    st.markdown("---")

    # Match H
    st.markdown("### Round of 16: Match H")
    w87 = get_official_winner("Argentina", "Cape Verde", raw_data)
    if w87: t8_a = st.selectbox("Advanced from Argentina vs Cape Verde:", options=[w87], key="r16_h_a_locked")
    else: t8_a = st.selectbox("Who advances from Argentina vs Cape Verde?", options=["Argentina", "Cape Verde"], key="r16_h_a_open")
        
    w88 = get_official_winner("Colombia", "Ghana", raw_data)
    if w88: t8_b = st.selectbox("Advanced from Colombia vs Ghana:", options=[w88], key="r16_h_b_locked")
    else: t8_b = st.selectbox("Who advances from Colombia vs Ghana?", options=["Colombia", "Ghana"], key="r16_h_b_open")
        
    ph1, ph2 = predict_match_analytics(t8_a, t8_b, master_elo)
    st.write(f"🔮 **Prediction:** {t8_a}: **{ph1}%** | {t8_b}: **{ph2}%**")
    display_probability_bar(ph1, ph2)
    r16_picks["mh_w"] = st.selectbox("Who wins Match H?", options=[t8_a, t8_b], key="sel_w_mh")
    st.markdown("---")

    # STAGE 3: QUARTERFINALS
    st.header("Quarterfinal Projections")
    st.markdown("---")

    # Quarterfinal 1
    st.markdown("### Quarterfinal 1")
    q1_a = r16_picks["ma_w"]
    q1_b = r16_picks["mb_w"]
    
    w89 = get_official_winner(q1_a, q1_b, raw_data)
    if w89: q1_winner = st.selectbox(f"Advanced from {q1_a} vs {q1_b}:", options=[w89], key="q1_locked")
    else: q1_winner = st.selectbox(f"Who advances from {q1_a} vs {q1_b}?", options=[q1_a, q1_b], key="q1_select")
        
    pq1, pq2 = predict_match_analytics(q1_a, q1_b, master_elo)
    st.write(f"🔮 **Prediction:** {q1_a}: **{pq1}%** | {q1_b}: **{pq2}%**")
    display_probability_bar(pq1, pq2)
    st.markdown("---")

    # Quarterfinal 2
    st.markdown("### Quarterfinal 2")
    q2_a = r16_picks["mc_w"]
    q2_b = r16_picks["md_w"]
    
    w90 = get_official_winner(q2_a, q2_b, raw_data)
    if w90: q2_winner = st.selectbox(f"Advanced from {q2_a} vs {q2_b}:", options=[w90], key="q2_locked")
    else: q2_winner = st.selectbox(f"Who advances from {q2_a} vs {q2_b}?", options=[q2_a, q2_b], key="q2_select")
        
    pq3, pq4 = predict_match_analytics(q2_a, q2_b, master_elo)
    st.write(f"🔮 **Prediction:** {q2_a}: **{pq3}%** | {q2_b}: **{pq4}%**")
    display_probability_bar(pq3, pq4)
    st.markdown("---")

    # Quarterfinal 3
    st.markdown("### Quarterfinal 3")
    q3_a = r16_picks["me_w"]
    q3_b = r16_picks["mf_w"]
    
    w91 = get_official_winner(q3_a, q3_b, raw_data)
    if w91: q3_winner = st.selectbox(f"Advanced from {q3_a} vs {q3_b}:", options=[w91], key="q3_locked")
    else: q3_winner = st.selectbox(f"Who advances from {q3_a} vs {q3_b}?", options=[q3_a, q3_b], key="q3_select")
        
    pq5, pq6 = predict_match_analytics(q3_a, q3_b, master_elo)
    st.write(f"🔮 **Prediction:** {q3_a}: **{pq5}%** | {q3_b}: **{pq6}%**")
    display_probability_bar(pq5, pq6)
    st.markdown("---")

    # Quarterfinal 4
    st.markdown("### Quarterfinal 4")
    q4_a = r16_picks["mg_w"]
    q4_b = r16_picks["mh_w"]
    
    w92 = get_official_winner(q4_a, q4_b, raw_data)
    if w92: q4_winner = st.selectbox(f"Advanced from {q4_a} vs {q4_b}:", options=[w92], key="q4_locked")
    else: q4_winner = st.selectbox(f"Who advances from {q4_a} vs {q4_b}?", options=[q4_a, q4_b], key="q4_select")
        
    pq7, pq8 = predict_match_analytics(q4_a, q4_b, master_elo)
    st.write(f"🔮 **Prediction:** {q4_a}: **{pq7}%** | {q4_b}: **{pq8}%**")
    display_probability_bar(pq7, pq8)
    st.markdown("---")

    # STAGE 4: SEMIFINALS TO GRAND FINAL
    st.header("Semifinals & Third Place Playoff")
    st.markdown("---")

    # --- Semifinal 1 ---
    st.markdown("### Semifinal 1")
    w93 = get_official_winner(q1_winner, q2_winner, raw_data)
    if w93: f_a = st.selectbox(f"Advanced from Semifinal 1:", options=[w93], key="sf1_locked")
    else: f_a = st.selectbox("Who advances to the Grand Final?", options=[q1_winner, q2_winner], key="sf1_select")
        
    ps1, ps2 = predict_match_analytics(q1_winner, q2_winner, master_elo)
    st.write(f"📊 **Prediction:** {q1_winner}: **{ps1}%** | {q2_winner}: **{ps2}%**")
    display_probability_bar(ps1, ps2)
    st.markdown("---")

    # --- Semifinal 2 ---
    st.markdown("### Semifinal 2")
    w94 = get_official_winner(q3_winner, q4_winner, raw_data)
    if w94: f_b = st.selectbox(f"Advanced from Semifinal 2:", options=[w94], key="sf2_locked")
    else: f_b = st.selectbox("Who advances to the Grand Final?", options=[q3_winner, q4_winner], key="sf2_select")
        
    ps3, ps4 = predict_match_analytics(q3_winner, q4_winner, master_elo)
    st.write(f"📊 **Prediction:** {q3_winner}: **{ps3}%** | {q4_winner}: **{ps4}%**")
    display_probability_bar(ps3, ps4)
    st.markdown("---")

    # --- Third-Place Playoff ---
    st.markdown("### 🥉 Third Place Playoff")
    sf1_loser = q2_winner if f_a == q1_winner else q1_winner
    sf2_loser = q4_winner if f_b == q3_winner else q3_winner
    
    w95 = get_official_winner(sf1_loser, sf2_loser, raw_data)
    if w95: bronze_winner = st.selectbox(f"Advanced from Third Place Playoff:", options=[w95], key="tp_locked")
    else: bronze_winner = st.selectbox(f"Who wins the Third Place Playoff?", options=[sf1_loser, sf2_loser], key="tp_select")
        
    p_tp1, p_tp2 = predict_match_analytics(sf1_loser, sf2_loser, master_elo)
    st.write(f"📊 **Prediction:** {sf1_loser}: **{p_tp1}%** | {sf2_loser}: **{p_tp2}%**")
    display_probability_bar(p_tp1, p_tp2)
    st.markdown("---")

    st.header("FIFA World Cup 2026 Final")
    st.markdown("---")

    # --- The Grand Final Showdown ---
    st.markdown("### 🥇 The Final Showdown")
    st.write(f" {f_a} vs {f_b}")
    final_1, final_2 = predict_match_analytics(f_a, f_b, master_elo)

    st.write(f"📊 {f_a}: **{final_1}%** | {f_b}: **{final_2}%**")
    
    display_probability_bar(final_1, final_2, height=28)
    
    if final_1 >= final_2:
        higher_team, lower_team = f_a, f_b
    else:
        higher_team, lower_team = f_b, f_a

    st.info(f"🏆 **{higher_team}** is predicted to beat **{lower_team}** to clinch the World Cup!")
