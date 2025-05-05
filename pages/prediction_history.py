import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta
import base64
import io
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Prediction History - F1 Race Outcome Predictor 2025",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
def local_css():
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
    </style>
    """, unsafe_allow_html=True)

# Apply theme based on dark mode setting
def apply_theme():
    if "dark_mode" in st.session_state and st.session_state.dark_mode:
        st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="light-mode">', unsafe_allow_html=True)

# Initialize database connection
def init_db():
    try:
        base_dir = Path(__file__).parent.parent
        db_path = os.path.join(base_dir, "data/db/f1_predictions.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_name TEXT,
            date TEXT,
            timestamp TEXT,
            model TEXT,
            data TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            last_race TEXT
        )
        """)

        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Error initializing database: {e}")
        return None

# Get prediction history from database
def get_prediction_history(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, race_name, date, timestamp, model, data FROM prediction_history ORDER BY timestamp DESC")
        rows = cursor.fetchall()

        history = []
        for row in rows:
            id, race_name, date, timestamp, model, data = row
            try:
                data_json = json.loads(data)
            except:
                data_json = []

            history.append({
                'id': id,
                'race_name': race_name,
                'date': date,
                'timestamp': timestamp,
                'model': model,
                'data': data_json
            })

        return history
    except Exception as e:
        st.error(f"Error getting prediction history: {e}")
        return []

# Get prediction history from session state if database is not available
def get_session_prediction_history():
    if "prediction_history" in st.session_state:
        return st.session_state.prediction_history
    return []

# Generate synthetic prediction history for demonstration
def generate_synthetic_history():
    races = [
        "Bahrain Grand Prix",
        "Saudi Arabian Grand Prix",
        "Miami Grand Prix",
        "Emilia Romagna Grand Prix",
        "Monaco Grand Prix",
        "Spanish Grand Prix"
    ]

    models = [
        "Stacking Ensemble",
        "Random Forest",
        "XGBoost",
        "Logistic Regression"
    ]

    drivers = [
        {"driver": "M. Verstappen", "team": "Red Bull Racing"},
        {"driver": "L. Hamilton", "team": "Mercedes"},
        {"driver": "C. Leclerc", "team": "Ferrari"},
        {"driver": "L. Norris", "team": "McLaren"},
        {"driver": "S. Perez", "team": "Red Bull Racing"},
        {"driver": "C. Sainz", "team": "Ferrari"},
        {"driver": "G. Russell", "team": "Mercedes"},
        {"driver": "O. Piastri", "team": "McLaren"},
        {"driver": "F. Alonso", "team": "Aston Martin"}
    ]

    history = []

    # Generate 20 random predictions
    for i in range(20):
        # Random race
        race = np.random.choice(races)

        # Random model
        model = np.random.choice(models)

        # Random date in the past 30 days
        days_ago = np.random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')
        date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        # Random top 3 drivers
        top3_indices = np.random.choice(len(drivers), 3, replace=False)
        top3 = []

        for idx in top3_indices:
            driver_info = drivers[idx].copy()
            driver_info['probability'] = np.random.uniform(0.6, 0.95)
            top3.append(driver_info)

        # Sort by probability
        top3 = sorted(top3, key=lambda x: x['probability'], reverse=True)

        history.append({
            'id': i + 1,
            'race_name': race,
            'date': date,
            'timestamp': timestamp,
            'model': model,
            'data': top3
        })

    # Sort by timestamp (newest first)
    history = sorted(history, key=lambda x: x['timestamp'], reverse=True)

    return history

# Create a download link for CSV
def get_csv_download_link(df, filename="prediction_history.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">Download CSV</a>'
    return href

# Main content
def main():
    # Apply CSS and theme
    local_css()
    apply_theme()

    # Page title
    st.title("📝 Prediction History")
    st.write("View and analyze your past predictions")

    # Initialize database
    conn = init_db()

    # Get prediction history
    if conn:
        history = get_prediction_history(conn)
    else:
        history = get_session_prediction_history()

    # If no history, generate synthetic data for demonstration
    if not history:
        st.info("No prediction history found. Generating synthetic data for demonstration.")
        history = generate_synthetic_history()

    # Create filters
    st.subheader("Filter Predictions")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Get unique race names
        race_names = sorted(list(set([h['race_name'] for h in history])))
        selected_races = st.multiselect("Race", race_names, default=[])

    with col2:
        # Get unique models
        models = sorted(list(set([h['model'] for h in history])))
        selected_models = st.multiselect("Model", models, default=[])

    with col3:
        # Date range
        min_date = min([datetime.strptime(h['date'], '%Y-%m-%d') for h in history])
        max_date = max([datetime.strptime(h['date'], '%Y-%m-%d') for h in history])

        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = min_date
            end_date = max_date

    # Filter history based on selections
    filtered_history = history

    if selected_races:
        filtered_history = [h for h in filtered_history if h['race_name'] in selected_races]

    if selected_models:
        filtered_history = [h for h in filtered_history if h['model'] in selected_models]

    if start_date and end_date:
        filtered_history = [
            h for h in filtered_history 
            if start_date <= datetime.strptime(h['date'], '%Y-%m-%d').date() <= end_date
        ]

    # Display history count
    st.write(f"Showing {len(filtered_history)} of {len(history)} predictions")

    # Create a DataFrame for export
    export_data = []

    for h in filtered_history:
        for i, driver_data in enumerate(h['data']):
            export_data.append({
                'Race': h['race_name'],
                'Date': h['date'],
                'Timestamp': h['timestamp'],
                'Model': h['model'],
                'Position': i + 1,
                'Driver': driver_data['driver'],
                'Team': driver_data['team'],
                'Probability': driver_data['probability']
            })

    export_df = pd.DataFrame(export_data)

    # Export button
    st.markdown(get_csv_download_link(export_df), unsafe_allow_html=True)

    # Display history
    st.subheader("Prediction History")

    # Create tabs for different views
    tab1, tab2 = st.tabs(["List View", "Analytics"])

    with tab1:
        # Display history as a list
        for h in filtered_history:
            with st.expander(f"{h['race_name']} - {h['date']}"):
                st.markdown(f"""
                <div class="history-item">
                    <div class="history-date">{h['timestamp']}</div>
                    <div class="history-race">{h['race_name']}</div>
                    <div class="history-model">Model: {h['model']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Display top 3 predictions
                cols = st.columns(3)

                for i, (col, driver_data) in enumerate(zip(cols, h['data'])):
                    with col:
                        position = ["🥇 1st", "🥈 2nd", "🥉 3rd"][i]

                        st.markdown(f"""
                        <div class="card">
                            <div class="metric-label">{position}</div>
                            <div class="metric-value">{driver_data['driver']}</div>
                            <div class="metric-label">{driver_data['team']}</div>
                            <div class="metric-label">Probability: {driver_data['probability']:.1%}</div>
                        </div>
                        """, unsafe_allow_html=True)

    with tab2:
        # Analytics view
        if export_df.empty:
            st.warning("No data available for analytics.")
        else:
            # Create visualizations

            # 1. Predictions by race
            st.subheader("Predictions by Race")
            race_counts = export_df['Race'].value_counts().reset_index()
            race_counts.columns = ['Race', 'Count']

            fig = px.bar(
                race_counts,
                x='Race',
                y='Count',
                title="Number of Predictions by Race",
                color='Race'
            )

            st.plotly_chart(fig, use_container_width=True)

            # 2. Driver prediction frequency
            st.subheader("Top Predicted Drivers")

            # Filter for only 1st place predictions
            first_place_df = export_df[export_df['Position'] == 1]
            driver_counts = first_place_df['Driver'].value_counts().reset_index()
            driver_counts.columns = ['Driver', 'Count']

            fig = px.bar(
                driver_counts.head(10),
                x='Driver',
                y='Count',
                title="Top 10 Drivers Predicted for 1st Place",
                color='Driver'
            )

            st.plotly_chart(fig, use_container_width=True)

            # 3. Average probability by model
            st.subheader("Model Confidence Comparison")

            model_probs = export_df.groupby(['Model', 'Position'])['Probability'].mean().reset_index()

            fig = px.bar(
                model_probs,
                x='Model',
                y='Probability',
                color='Position',
                barmode='group',
                title="Average Prediction Probability by Model and Position",
                labels={'Probability': 'Avg. Probability', 'Model': 'Model', 'Position': 'Position'},
                text_auto='.1%'
            )

            st.plotly_chart(fig, use_container_width=True)

            # 4. Predictions over time
            st.subheader("Prediction Activity Over Time")

            # Convert timestamp to datetime
            export_df['Timestamp'] = pd.to_datetime(export_df['Timestamp'])
            export_df['Date'] = pd.to_datetime(export_df['Date'])

            # Group by date and count predictions
            date_counts = export_df.groupby(export_df['Date'].dt.date).size().reset_index()
            date_counts.columns = ['Date', 'Count']

            fig = px.line(
                date_counts,
                x='Date',
                y='Count',
                title="Prediction Activity Over Time",
                labels={'Count': 'Number of Predictions', 'Date': 'Date'},
                markers=True
            )

            st.plotly_chart(fig, use_container_width=True)

    # Close database connection
    if conn:
        conn.close()

if __name__ == "__main__":
    main()
