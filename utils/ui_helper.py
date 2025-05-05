import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import base64
from datetime import datetime

def apply_custom_css():
    """
    Apply custom CSS styling to the Streamlit app.
    """
    st.markdown("""
    <style>
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stApp {
        transition: all 0.5s ease;
    }
    .dark-mode {
        --background-color: #0e1117;
        --text-color: #ffffff;
        --card-bg-color: #262730;
    }
    .light-mode {
        --background-color: #ffffff;
        --text-color: #0e1117;
        --card-bg-color: #f0f2f6;
    }
    .card {
        background-color: var(--card-bg-color);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        text-align: center;
        padding: 15px;
        border-radius: 8px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.8;
    }
    .team-red-bull {
        background: linear-gradient(90deg, #0600EF 0%, #1E00C7 100%);
        color: white;
    }
    .team-ferrari {
        background: linear-gradient(90deg, #DC0000 0%, #AE0000 100%);
        color: white;
    }
    .team-mercedes {
        background: linear-gradient(90deg, #00D2BE 0%, #00A39B 100%);
        color: black;
    }
    .team-mclaren {
        background: linear-gradient(90deg, #FF8700 0%, #D97500 100%);
        color: black;
    }
    .team-aston-martin {
        background: linear-gradient(90deg, #006F62 0%, #005A4F 100%);
        color: white;
    }
    .team-alpine {
        background: linear-gradient(90deg, #0090FF 0%, #0070C8 100%);
        color: white;
    }
    .team-williams {
        background: linear-gradient(90deg, #005AFF 0%, #0046C8 100%);
        color: white;
    }
    .team-rb {
        background: linear-gradient(90deg, #1E41FF 0%, #1A35D1 100%);
        color: white;
    }
    .team-sauber {
        background: linear-gradient(90deg, #52E252 0%, #42B542 100%);
        color: black;
    }
    .team-haas {
        background: linear-gradient(90deg, #FFFFFF 0%, #EEEEEE 100%);
        color: black;
        border: 1px solid #ddd;
    }
    .slider-label {
        font-weight: bold;
        margin-bottom: 5px;
    }
    .probability-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .probability-medium {
        color: #FFC107;
        font-weight: bold;
    }
    .probability-low {
        color: #F44336;
        font-weight: bold;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
    }
    .download-button:hover {
        background-color: #45a049;
    }
    .history-item {
        border-left: 4px solid #3366ff;
        padding-left: 15px;
        margin-bottom: 15px;
    }
    .history-date {
        font-size: 12px;
        color: #888;
    }
    .history-race {
        font-weight: bold;
        font-size: 16px;
    }
    .history-model {
        font-style: italic;
        font-size: 14px;
    }
    .model-card {
        border-left: 4px solid #3366ff;
        padding-left: 15px;
        margin-bottom: 15px;
    }
    .model-name {
        font-weight: bold;
        font-size: 18px;
    }
    .model-desc {
        font-size: 14px;
        opacity: 0.8;
    }
    </style>
    """, unsafe_allow_html=True)

def apply_theme():
    """
    Apply theme based on dark mode setting in session state.
    """
    if "dark_mode" in st.session_state and st.session_state.dark_mode:
        st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="light-mode">', unsafe_allow_html=True)

def toggle_dark_mode():
    """
    Toggle dark/light mode in session state.
    """
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    
    st.session_state.dark_mode = not st.session_state.dark_mode
    apply_theme()

def render_sidebar():
    """
    Render the sidebar navigation menu.
    """
    with st.sidebar:
        st.title("🏎️ F1 Race Predictor 2025")
        
        # Dark/Light mode toggle
        st.write("---")
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode if "dark_mode" in st.session_state else False, on_change=toggle_dark_mode)
        
        st.write("---")
        
        # Navigation
        st.subheader("Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("app.py")
        
        if st.button("📊 Driver & Constructor Analysis", use_container_width=True):
            st.switch_page("pages/driver_analysis.py")
            
        if st.button("🏁 Track Insights", use_container_width=True):
            st.switch_page("pages/track_insights.py")
            
        if st.button("📈 Model Metrics", use_container_width=True):
            st.switch_page("pages/model_metrics.py")
            
        if st.button("🔮 Scenario Simulator", use_container_width=True):
            st.switch_page("pages/scenario_simulator.py")
            
        if st.button("📝 Prediction History", use_container_width=True):
            st.switch_page("pages/prediction_history.py")
        
        st.write("---")
        
        # About section
        st.caption("F1 Race Outcome Predictor 2025")
        st.caption("Developed by Ashwinth Reddy")
        st.caption("© 2025 | v1.0.0")

def display_driver_card(driver_name, team_name, probability=None):
    """
    Display a driver card with team styling.
    
    Args:
        driver_name (str): Name of the driver
        team_name (str): Name of the team
        probability (float, optional): Podium probability. Defaults to None.
    """
    team_class = f"team-{team_name.lower().replace(' ', '-').replace('f1 team', '')}"
    
    card_content = f"""
    <div class="card {team_class}">
        <h3>{driver_name}</h3>
        <p>{team_name}</p>
    """
    
    if probability is not None:
        # Determine probability class
        prob_class = ""
        if probability >= 0.7:
            prob_class = "probability-high"
        elif probability >= 0.4:
            prob_class = "probability-medium"
        else:
            prob_class = "probability-low"
        
        card_content += f"""
        <div class="metric-label">Podium Probability</div>
        <div class="metric-value {prob_class}">{probability:.1%}</div>
        """
    
    card_content += "</div>"
    
    st.markdown(card_content, unsafe_allow_html=True)

def display_metric_card(label, value, description=None):
    """
    Display a metric card with a label and value.
    
    Args:
        label (str): Label for the metric
        value (str/float/int): Value of the metric
        description (str, optional): Additional description. Defaults to None.
    """
    card_content = f"""
    <div class="card metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    """
    
    if description:
        card_content += f"""
        <div class="metric-label">{description}</div>
        """
    
    card_content += "</div>"
    
    st.markdown(card_content, unsafe_allow_html=True)

def display_model_card(model_name, description):
    """
    Display a model card with name and description.
    
    Args:
        model_name (str): Name of the model
        description (str): Description of the model
    """
    st.markdown(f"""
    <div class="model-card">
        <div class="model-name">{model_name}</div>
        <div class="model-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)

def display_podium(top_3_drivers, include_images=True):
    """
    Display a podium with the top 3 drivers.
    
    Args:
        top_3_drivers (pd.DataFrame): DataFrame containing the top 3 drivers
        include_images (bool, optional): Whether to include driver images. Defaults to True.
    """
    cols = st.columns(3)
    positions = ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"]
    
    for i, (_, driver) in enumerate(top_3_drivers.iterrows()):
        with cols[i]:
            st.markdown(f"### {positions[i]}")
            
            if include_images:
                # Try to load driver image
                driver_name = driver['driver'].replace('. ', '_')
                image_path = f"f1-predictor-2025-main/dashboad_content/F1 2025 Season Drivers/{driver_name}.png"
                
                try:
                    image = Image.open(image_path)
                    st.image(image, width=150)
                except:
                    st.markdown(f"### {driver['driver']}")
            
            # Display team and probability
            team_class = f"team-{driver['team'].lower().replace(' ', '-').replace('f1 team', '')}"
            st.markdown(f"""
            <div class="card {team_class}">
                <div class="metric-value">{driver['probability']:.1%}</div>
                <div class="metric-label">{driver['team']}</div>
            </div>
            """, unsafe_allow_html=True)

def create_download_link(df, filename="data.csv"):
    """
    Create a download link for a DataFrame as CSV.
    
    Args:
        df (pd.DataFrame): DataFrame to download
        filename (str, optional): Filename for the download. Defaults to "data.csv".
    
    Returns:
        str: HTML link for downloading the CSV
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download CSV</a>'
    return href

def display_probability_gauge(probability, title="Podium Probability"):
    """
    Display a gauge chart for probability.
    
    Args:
        probability (float): Probability value between 0 and 1
        title (str, optional): Title for the gauge. Defaults to "Podium Probability".
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'red'},
                {'range': [40, 70], 'color': 'orange'},
                {'range': [70, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=250)
    
    st.plotly_chart(fig, use_container_width=True)

def display_race_calendar(upcoming_races_df, predictions_df=None):
    """
    Display a race calendar with upcoming races and predictions if available.
    
    Args:
        upcoming_races_df (pd.DataFrame): DataFrame containing upcoming races
        predictions_df (pd.DataFrame, optional): DataFrame containing predictions. Defaults to None.
    """
    if upcoming_races_df.empty:
        st.warning("No upcoming races data available.")
        return
    
    # Get unique races and sort by date
    unique_races = upcoming_races_df[['race_name', 'date']].drop_duplicates()
    unique_races['date'] = pd.to_datetime(unique_races['date'])
    unique_races = unique_races.sort_values('date')
    
    # Create a table for the race calendar
    calendar_data = []
    
    for _, race in unique_races.iterrows():
        race_name = race['race_name']
        race_date = race['date'].strftime('%B %d, %Y')
        
        # Get top 3 predictions if available
        top_3 = None
        if predictions_df is not None and not predictions_df.empty:
            race_predictions = predictions_df[predictions_df['race_name'] == race_name]
            if not race_predictions.empty:
                top_3 = race_predictions.nlargest(3, 'probability')['driver'].tolist()
        
        calendar_data.append({
            'Race': race_name,
            'Date': race_date,
            'Predicted Podium': ', '.join(top_3) if top_3 else 'No predictions available'
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    st.dataframe(calendar_df, use_container_width=True)