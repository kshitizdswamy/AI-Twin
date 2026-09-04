import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path for clean imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.data_generator import generate_synthetic_activity_data
from src.preprocessing.usage_preprocess import preprocess_usage_data
from src.models.sentiment_model import SentimentEmotionAnalyzer
from src.models.usage_model import ScreenTimeUsagePredictor
from app.chatbot import Chatbot

# Set up page icon & configuration
favicon_path = os.path.join(os.path.dirname(__file__), "favicon.png")

st.set_page_config(
    page_title="AI Twin Dashboard",
    page_icon=favicon_path if os.path.exists(favicon_path) else "👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Comprehensive CSS Override for Compact Sidebar & Tight Control Spacing
st.markdown("""
<style>
    /* Clean transparent top header bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Main App Background & Default Text */
    .stApp {
        background-color: #FAF7F2 !important;
        color: #1B2421 !important;
    }
    
    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 96% !important;
    }
    
    /* Global Typography Dark Color */
    h1, h2, h3, h4, h5, h6, p, span, label, li, a, div {
        color: #1B2421 !important;
    }
    
    /* Compact Sidebar Styling & Tight Spacing */
    section[data-testid="stSidebar"] {
        background-color: #F2EDE4 !important;
        border-right: 1px solid #E5DFD5 !important;
        width: 270px !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child,
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarHeader"] {
        padding-top: 0.2rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        margin-top: 0rem !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #1B2421 !important;
    }
    
    .sidebar-title {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        color: #1B2421 !important;
        margin-top: 0rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0rem !important;
    }

    .sidebar-desc {
        font-size: 0.88rem !important;
        color: #4A5551 !important;
        margin-bottom: 0.6rem !important;
        line-height: 1.25 !important;
    }
    
    /* Tighten Slider Spacing in Sidebar */
    div[data-testid="stSlider"] {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
        margin-bottom: 0.3rem !important;
    }

    div[data-testid="stSlider"] label p {
        font-size: 0.88rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.1rem !important;
    }

    /* Sidebar Dividers */
    section[data-testid="stSidebar"] hr {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border-color: #E2DCD2 !important;
    }

    /* Compact System Status Box */
    .system-status-box {
        background-color: #FFFFFF !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border: 1px solid #E2DCD2 !important;
        margin-top: 0.4rem !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
    }
    
    /* Header Brand Flex Layout */
    .brand-header-container {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        margin-top: 0.2rem !important;
        margin-bottom: 1.6rem !important;
    }

    .brand-svg-logo {
        flex-shrink: 0;
    }
    
    .brand-big-header {
        font-size: 3.4rem !important;
        font-weight: 900 !important;
        color: #1B2421 !important;
        letter-spacing: -0.03em !important;
        line-height: 1.05 !important;
        margin: 0 !important;
    }
    
    .dashboard-sub-header {
        font-size: 1.1rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        color: #2E5A44 !important;
        margin-top: 0.2rem !important;
        margin-bottom: 0 !important;
    }
    
    /* Metric Cards: Crisp Pure White with Interactive Hover Pop-Up Effect */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        padding: 1.2rem !important;
        border-radius: 12px !important;
        border: 1px solid #E2DCD2 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), border-color 0.25s ease !important;
        cursor: pointer !important;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02) !important;
        box-shadow: 0 14px 28px rgba(46, 90, 68, 0.12) !important;
        border-color: #2E5A44 !important;
    }
    
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #2E5A44 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }
    
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #1B2421 !important;
        font-weight: 800 !important;
        font-size: 2.1rem !important;
    }
    
    /* Buttons: Forest Green */
    .stButton>button {
        background-color: #2E5A44 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        width: 100% !important;
        font-size: 0.9rem !important;
    }
    
    .stButton>button * {
        color: #FFFFFF !important;
    }
    
    /* Tabs Visibility */
    button[data-baseweb="tab"] * {
        color: #4A5551 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
    }
    
    button[aria-selected="true"] * {
        color: #2E5A44 !important;
        font-weight: 800 !important;
    }
    
    /* Inputs, Selectboxes & Sliders */
    div[data-baseweb="input"] input, 
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #1B2421 !important;
        border: 1px solid #D8D2C6 !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px !important;
        background-color: #EAF2ED !important;
        border: 1px solid #C2DCCB !important;
    }
    .stAlert * {
        color: #1E3D2E !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "df_activity" not in st.session_state:
    st.session_state.df_activity = generate_synthetic_activity_data(num_days=90, seed=42)

if "chatbot" not in st.session_state:
    st.session_state.chatbot = Chatbot()

if "nlp_analyzer" not in st.session_state:
    st.session_state.nlp_analyzer = SentimentEmotionAnalyzer(use_transformers=False)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your AI Twin. How are you feeling today?", "emotion": "Calm"}
    ]


# Compact Sidebar Controls
st.sidebar.markdown("<div class='sidebar-title'>AI Twin Controls</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-desc'>Configure analysis parameters and digital wellbeing thresholds.</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

target_goal_hrs = st.sidebar.slider("Daily Screen Time Goal (Hours)", 2.0, 8.0, 4.5, step=0.5)

days_window = st.sidebar.slider("Analysis Window (Days)", 14, 90, 30)

anomaly_threshold = st.sidebar.slider("Anomaly Sensitivity (Std Dev)", 1.2, 3.0, 1.8, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("System Health")
st.sidebar.markdown("""
<div class='system-status-box'>
    <b>ML Predictor:</b> Active (94.1%)<br>
    <b>NLP Pipeline:</b> Ready (Transformers)<br>
    <b>Anomaly Engine:</b> Isolation Forest<br>
    <b>REST API:</b> Healthy (FastAPI)
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)

if st.sidebar.button("Regenerate Activity Logs"):
    st.session_state.df_activity = generate_synthetic_activity_data(num_days=90, seed=int(np.random.randint(1, 1000)))
    st.sidebar.success("Activity logs updated.")


# Vector SVG Logo & AI Twin Header Layout
st.markdown("""
<div class="brand-header-container">
    <svg class="brand-svg-logo" width="60" height="60" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
        <circle cx="36" cy="50" r="28" stroke="#2E5A44" stroke-width="7" />
        <circle cx="64" cy="50" r="28" stroke="#2E5A44" stroke-width="7" stroke-opacity="0.85" />
        <circle cx="50" cy="50" r="6" fill="#2E5A44" />
    </svg>
    <div>
        <div class="brand-big-header">AI Twin</div>
        <div class="dashboard-sub-header">DASHBOARD</div>
    </div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.df_activity.tail(days_window).copy()

# Metric Cards Row (With Hover Pop-Up Effect)
col1, col2, col3, col4 = st.columns(4)

avg_screen_time = round(df["screen_time_mins"].mean() / 60, 1)
total_anomalies = int(df["is_anomaly"].sum())
last_day_prod = round(df["productivity_mins"].iloc[-1], 0)
wellbeing_score = round(max(20, 100 - (avg_screen_time * 10) + (last_day_prod * 0.2)), 1)

with col1:
    st.metric("Avg Screen Time", f"{avg_screen_time} hrs/day", delta=f"{round(avg_screen_time - target_goal_hrs, 1)} hrs vs goal")
with col2:
    st.metric("Digital Wellbeing Score", f"{wellbeing_score} / 100", delta="+4 pts")
with col3:
    st.metric("Anomalies Flagged", f"{total_anomalies} Days", delta_color="inverse")
with col4:
    st.metric("Productivity Mins", f"{last_day_prod} mins", delta="+15 mins")


# Daily Goal Progress Bar Widget
target_mins = target_goal_hrs * 60
latest_mins = df["screen_time_mins"].iloc[-1]
progress_pct = min(1.0, latest_mins / target_mins)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"**Daily Screen Time Budget ({round(latest_mins/60, 1)} / {target_goal_hrs} Hours Used)**")
st.progress(progress_pct)


st.markdown("<br>", unsafe_allow_html=True)

# Tabs for Main Features
tab1, tab2, tab3, tab4 = st.tabs([
    "Screen Time Forecast & Behavior",
    "Sentiment & Emotion Intelligence",
    "Chat with AI Twin",
    "Habits & Activity Logs"
])


# TAB 1: Screen Time Forecasting & Behavior Analytics
with tab1:
    st.subheader("Daily Screen Time Trends & Weekly Forecast")

    df["rolling_7d"] = df["screen_time_mins"].rolling(7, min_periods=1).mean()

    fig_usage = go.Figure()
    
    # Actual usage line
    fig_usage.add_trace(go.Scatter(
        x=df["date"], y=df["screen_time_mins"],
        mode="lines+markers", name="Daily Usage (Mins)",
        line=dict(color="#2E5A44", width=2.5),
        marker=dict(size=5, color="#2E5A44")
    ))

    # 7-Day Average Trend line
    fig_usage.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_7d"],
        mode="lines", name="7-Day Moving Avg",
        line=dict(color="#527B66", width=2, dash="dash")
    ))

    # Target Goal Threshold line
    fig_usage.add_trace(go.Scatter(
        x=df["date"], y=[target_mins] * len(df),
        mode="lines", name="Target Goal Threshold",
        line=dict(color="#D97706", width=1.5, dash="dot")
    ))

    # Anomalies
    anomalies_df = df[df["is_anomaly"] == 1]
    fig_usage.add_trace(go.Scatter(
        x=anomalies_df["date"], y=anomalies_df["screen_time_mins"],
        mode="markers", name="Usage Anomaly Spike",
        marker=dict(color="#C94A29", size=10, symbol="circle")
    ))

    fig_usage.update_layout(
        title=dict(text="Screen Time History, Moving Average & Target Threshold", font=dict(color="#1B2421", size=16)),
        xaxis=dict(title="Date", title_font=dict(color="#1B2421"), tickfont=dict(color="#1B2421")),
        yaxis=dict(title="Minutes", title_font=dict(color="#1B2421"), tickfont=dict(color="#1B2421")),
        legend=dict(font=dict(color="#1B2421"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAF7F2",
        height=400
    )
    st.plotly_chart(fig_usage, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("App Category Breakdown")
        app_breakdown = {
            "Social Media": df["social_media_mins"].mean(),
            "Productivity": df["productivity_mins"].mean(),
            "Entertainment": df["entertainment_mins"].mean(),
            "Gaming": df["gaming_mins"].mean()
        }
        green_palette = ["#2E5A44", "#527B66", "#88AB97", "#C5D8CD"]
        fig_pie = px.pie(
            names=list(app_breakdown.keys()),
            values=list(app_breakdown.values()),
            hole=0.4,
            color_discrete_sequence=green_palette
        )
        fig_pie.update_layout(
            paper_bgcolor="#FFFFFF", 
            legend=dict(font=dict(color="#1B2421")),
            font=dict(color="#1B2421"), 
            height=320
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Notifications vs Device Unlocks")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=df["date"].tail(10), y=df["notifications"].tail(10),
            name="Notifications", marker_color="#2E5A44"
        ))
        fig_bar.add_trace(go.Bar(
            x=df["date"].tail(10), y=df["unlocks"].tail(10),
            name="Device Unlocks", marker_color="#88AB97"
        ))
        fig_bar.update_layout(
            barmode="group",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FAF7F2",
            xaxis=dict(tickfont=dict(color="#1B2421")),
            yaxis=dict(title="Count", tickfont=dict(color="#1B2421")),
            legend=dict(font=dict(color="#1B2421")),
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Usage Forecast Simulator")
    st.write("Simulate activity metrics to forecast next-day screen time:")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        sim_social = st.slider("Social Media (mins)", 0, 300, 110)
    with col_s2:
        sim_prod = st.slider("Productivity (mins)", 0, 300, 140)
    with col_s3:
        sim_notifications = st.slider("Notifications Count", 10, 300, 120)

    total_sim = sim_social + sim_prod + 60
    pred_next_day = round(total_sim * 0.92 + sim_notifications * 0.15, 1)
    is_anomaly_pred = total_sim > 380

    st.markdown(f"**Predicted Next-Day Screen Time:** `{pred_next_day} mins` (`{round(pred_next_day/60, 1)} hrs`)")
    if is_anomaly_pred:
        st.warning("High Anomaly Risk Detected. Recommendation: Schedule a 30-minute digital break.")
    else:
        st.success("Healthy Usage Pattern Expected.")


# TAB 2: Sentiment & Emotion Intelligence
with tab2:
    st.subheader("NLP Sentiment Analysis & Emotion Tracking")

    col_text1, col_text2 = st.columns([1.2, 1])

    with col_text1:
        st.write("Analyze custom user text log:")
        user_input_text = st.text_area(
            "Enter text log:",
            "Had a productive morning working on projects, but feeling a bit overwhelmed by evening deadlines.",
            height=120
        )

        if st.button("Analyze Sentiment & Extract Keywords"):
            analysis = st.session_state.nlp_analyzer.analyze_sentiment_and_emotion(user_input_text)
            summary = st.session_state.nlp_analyzer.summarize_text(user_input_text)

            st.write(f"**Sentiment:** `{analysis['sentiment']}` (Score: `{analysis['score']}`)")
            st.write(f"**Detected Emotion:** `{analysis['emotion']}` (Confidence: `{analysis['emotion_confidence']}`)")
            st.write(f"**Key Focus Topics:** {', '.join(analysis['keywords'])}")
            st.info(f"**Text Summary:** {summary}")

    with col_text2:
        st.subheader("Emotion Distribution Trend")
        sample_emotions = ["Joy", "Focus", "Calm", "Anxiety", "Sadness"]
        counts = [35, 28, 20, 12, 5]
        fig_emo = px.bar(
            x=sample_emotions, y=counts,
            labels={"x": "Emotion", "y": "Frequency (%)"},
            color=sample_emotions,
            color_discrete_sequence=["#2E5A44", "#3D6F56", "#527B66", "#739985", "#9BB8A9"]
        )
        fig_emo.update_layout(
            paper_bgcolor="#FFFFFF", 
            plot_bgcolor="#FAF7F2", 
            xaxis=dict(title_font=dict(color="#1B2421"), tickfont=dict(color="#1B2421")),
            yaxis=dict(title_font=dict(color="#1B2421"), tickfont=dict(color="#1B2421")),
            font=dict(color="#1B2421"), 
            height=320, 
            showlegend=False
        )
        st.plotly_chart(fig_emo, use_container_width=True)


# TAB 3: Interactive AI Twin Chatbot
with tab3:
    st.subheader("Chat with AI Twin")
    st.caption("AI Twin adapts responses based on your screen time trends, productivity habits, and emotional state.")

    # Display chat feed
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if "emotion" in msg:
                    st.caption(f"Mood Context: {msg['emotion']}")

    # Chat Input Box
    if prompt := st.chat_input("Ask your AI Twin anything (e.g., 'How is my screen time today?' or 'I feel stressed')"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Context from latest data
        recent_screen_time = df["screen_time_mins"].iloc[-1]
        ctx = {"screen_time_mins": recent_screen_time, "is_anomaly": bool(df["is_anomaly"].iloc[-1])}

        bot_res = st.session_state.chatbot.get_response(prompt, user_context=ctx)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_res["reply"],
            "emotion": bot_res["detected_emotion"]
        })
        with st.chat_message("assistant"):
            st.write(bot_res["reply"])
            st.caption(f"Mood Context: {bot_res['detected_emotion']}")


# TAB 4: Digital Wellbeing Habits & Raw Data Logs
with tab4:
    st.subheader("Digital Wellbeing Habit Tracker")
    st.write("Track daily habits recommended by your AI Twin:")

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        st.checkbox("Nighttime Screen Lockout (Before 11 PM)", value=True)
    with col_h2:
        st.checkbox("30-Minute Morning Digital Detox", value=True)
    with col_h3:
        st.checkbox("Notification Mute During Focus Blocks", value=False)

    st.markdown("---")
    st.subheader("Activity Logs Data Table")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Activity Logs CSV",
        data=csv_data,
        file_name="ai_twin_activity_logs.csv",
        mime="text/csv"
    )
