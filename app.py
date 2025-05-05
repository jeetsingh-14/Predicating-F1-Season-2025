import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime
# import shap
import folium
from streamlit_folium import folium_static
from streamlit_extras.colored_header import colored_header
from pathlib import Path
import csv
import json

# Set page configuration
st.set_page_config(
    page_title="F1 Race Outcome Predictor 2025",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create directories if they don't exist
base_dir = Path(__file__).parent
os.makedirs(os.path.join(base_dir, "pages"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "utils"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "assets"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "data"), exist_ok=True)

# Initialize session state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "selected_race" not in st.session_state:
    st.session_state.selected_race = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Stacking Ensemble"
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

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
    </style>
    """, unsafe_allow_html=True)

# Apply theme based on dark mode setting
def apply_theme():
    if st.session_state.dark_mode:
        st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="light-mode">', unsafe_allow_html=True)

# Toggle dark/light mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode
    apply_theme()

# Load data functions
@st.cache_data(ttl=3600)
def load_upcoming_races():
    try:
        base_dir = Path(__file__).parent
        file_path = os.path.join(base_dir, "data", "processed", "upcoming_races.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading upcoming races: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_predictions():
    try:
        base_dir = Path(__file__).parent
        file_path = os.path.join(base_dir, "results", "2025_predictions_full.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    try:
        base_dir = Path(__file__).parent
        model_path = os.path.join(base_dir, "models", "stacking_model.pkl")
        scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize CSV for prediction logging
@st.cache_resource
def init_prediction_csv():
    try:
        # Get the base directory
        base_dir = Path(__file__).parent

        # Create data directory if it doesn't exist
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Define the CSV file path
        csv_path = os.path.join(data_dir, "prediction_history.csv")

        # Create the CSV file with headers if it doesn't exist
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['race_name', 'date', 'timestamp', 'model', 'data'])

        return csv_path
    except Exception as e:
        st.error(f"Error initializing prediction CSV: {e}")
        return None

# Function to log prediction to CSV
def log_prediction_to_csv(prediction_entry):
    try:
        csv_path = init_prediction_csv()
        if csv_path:
            with open(csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    prediction_entry['race_name'],
                    prediction_entry['date'],
                    prediction_entry['timestamp'],
                    prediction_entry['model'],
                    json.dumps(prediction_entry['top3'])
                ])
            return True
        return False
    except Exception as e:
        st.error(f"Error logging prediction to CSV: {e}")
        return False

# Function to load recent predictions from CSV
def load_recent_predictions(limit=10):
    try:
        csv_path = init_prediction_csv()
        if csv_path and os.path.exists(csv_path):
            predictions = []
            with open(csv_path, 'r', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    predictions.append(row)

            # Return the most recent predictions (up to the limit)
            return predictions[-limit:] if predictions else []
        return []
    except Exception as e:
        st.error(f"Error loading predictions from CSV: {e}")
        return []

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.title("F1 Race Predictor 2025")

        # Dark/Light mode toggle
        st.write("---")
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode, on_change=toggle_dark_mode)

        st.write("---")

        # Navigation
        st.subheader("Navigation")
        if st.button("Home", use_container_width=True):
            pass  # Already on home page

        if st.button("Driver & Constructor Analysis", use_container_width=True):
            st.switch_page("pages/driver_analysis.py")

        if st.button("Track Insights", use_container_width=True):
            st.switch_page("pages/track_insights.py")

        if st.button("Model Metrics", use_container_width=True):
            st.switch_page("pages/model_metrics.py")

        if st.button("Scenario Simulator", use_container_width=True):
            st.switch_page("pages/scenario_simulator.py")

        if st.button("Prediction History", use_container_width=True):
            st.switch_page("pages/prediction_history.py")

        st.write("---")

        # About section
        st.caption("F1 Race Outcome Predictor 2025")

# Main content
def main():
    # Apply CSS and theme
    local_css()
    apply_theme()

    # Initialize prediction CSV
    init_prediction_csv()

    # Render sidebar
    render_sidebar()

    # Main content
    colored_header(
        label="Formula 1 Race Outcome Predictor 2025",
        description="Predict podium finishers for upcoming F1 races using machine learning",
        color_name="blue-green-70",
    )

    # Load data
    upcoming_races_df = load_upcoming_races()
    predictions_df = load_predictions()

    # Get unique races
    if not upcoming_races_df.empty:
        unique_races = upcoming_races_df['race_name'].unique()

        # Race selection
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Select Race")
            selected_race = st.selectbox(
                "Choose an upcoming race:",
                options=unique_races,
                index=0 if st.session_state.selected_race is None else list(unique_races).index(st.session_state.selected_race),
                key="race_selector"
            )
            st.session_state.selected_race = selected_race

            # Get race date
            race_date = upcoming_races_df[upcoming_races_df['race_name'] == selected_race]['date'].iloc[0]
            st.write(f"Race Date: {pd.to_datetime(race_date).strftime('%B %d, %Y')}")

            # Option to upload custom race data
            st.write("---")
            st.subheader("Or Upload Custom Race Data")
            uploaded_file = st.file_uploader("Upload CSV file with race data", type=["csv"])

            if uploaded_file is not None:
                try:
                    custom_data = pd.read_csv(uploaded_file)

                    # Validate required columns
                    required_columns = ['race_name', 'date', 'circuit', 'driver', 'team', 'grid_position']
                    missing_columns = [col for col in required_columns if col not in custom_data.columns]

                    if missing_columns:
                        st.warning(f"Missing required columns: {', '.join(missing_columns)}")

                        # Offer to fill missing columns with synthetic data
                        if st.button("Fill with synthetic data"):
                            for col in missing_columns:
                                if col == 'grid_position':
                                    custom_data[col] = 0
                                elif col == 'date':
                                    custom_data[col] = datetime.now().strftime('%Y-%m-%d')
                                else:
                                    custom_data[col] = "Unknown"

                            st.success("Missing columns filled with synthetic data")
                            st.write(custom_data)
                    else:
                        st.success("Custom data loaded successfully")
                        st.write(custom_data)

                        # Use custom data instead of selected race
                        st.session_state.selected_race = custom_data['race_name'].iloc[0]

                except Exception as e:
                    st.error(f"Error loading custom data: {e}")

        with col2:
            st.subheader("Model Selection")
            selected_model = st.selectbox(
                "Choose prediction model:",
                options=["Stacking Ensemble", "Logistic Regression", "Random Forest", "XGBoost", "Deep Neural Net"],
                index=0,
                key="model_selector"
            )
            st.session_state.selected_model = selected_model

            # Model explanation
            if selected_model == "Stacking Ensemble":
                st.info("Combines multiple models for better accuracy. Uses Random Forest and Gradient Boosting as base models with Logistic Regression as meta-learner.")
            elif selected_model == "Logistic Regression":
                st.info("Simple linear model for binary classification. Good for understanding feature importance.")
            elif selected_model == "Random Forest":
                st.info("Ensemble of decision trees. Robust to overfitting and handles non-linear relationships well.")
            elif selected_model == "XGBoost":
                st.info("Gradient boosting implementation. High performance with proper tuning.")
            elif selected_model == "Deep Neural Net":
                st.info("Deep learning model with multiple layers. Can capture complex patterns but requires more data.")

    # Display podium predictions
    st.write("---")
    st.subheader("Podium Predictions")

    if not predictions_df.empty and st.session_state.selected_race is not None:
        # Filter predictions for selected race
        race_predictions = predictions_df[predictions_df['race_name'] == st.session_state.selected_race]

        if not race_predictions.empty:
            # Get top 3 drivers
            top_3 = race_predictions.nlargest(3, 'probability')

            # Display podium
            cols = st.columns(3)
            positions = ["1st Place", "2nd Place", "3rd Place"]

            for i, (_, driver) in enumerate(top_3.iterrows()):
                with cols[i]:
                    st.markdown(f"### {positions[i]}")

                    # Try to load driver image
                    driver_name = driver['driver'].replace('. ', '_')
                    base_dir = Path(__file__).parent
                    image_path = os.path.join(base_dir, "dashboad_content", "F1 2025 Season Drivers", f"{driver_name}.png")

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

                    # Add historical win rate and confidence info
                    st.markdown(f"""
                    <div class="card">
                        <div class="metric-label">Historical Win Rate</div>
                        <div class="metric-value">32%</div>
                        <div class="metric-label">Confidence Score</div>
                        <div class="metric-value">High</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Toggle between classification and probabilistic view
            st.write("---")
            view_mode = st.radio(
                "View Mode:",
                options=["Probabilistic Ranking", "Binary Classification (Podium/Non-Podium)"],
                horizontal=True
            )

            if view_mode == "Probabilistic Ranking":
                # Create bar chart of probabilities
                fig = px.bar(
                    race_predictions.nlargest(10, 'probability'),
                    x='driver',
                    y='probability',
                    color='team',
                    title=f"Top 10 Podium Probabilities - {st.session_state.selected_race}",
                    labels={'probability': 'Podium Probability', 'driver': 'Driver'},
                    text_auto='.1%'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Create binary classification view
                binary_df = race_predictions.copy()
                binary_df['podium'] = (binary_df['probability'] > 0.8).astype(int)

                fig = px.scatter(
                    binary_df,
                    x='driver',
                    y='probability',
                    color='podium',
                    size='probability',
                    title=f"Binary Podium Classification - {st.session_state.selected_race}",
                    labels={'probability': 'Confidence Score', 'driver': 'Driver', 'podium': 'Podium Finish'},
                    color_discrete_map={1: 'green', 0: 'red'},
                    hover_data=['team']
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No predictions available for {st.session_state.selected_race}")
    else:
        st.warning("Please select a race or load predictions data")

    # Feature importance visualization
    st.write("---")
    st.subheader("Feature Importance")

    # Load model for SHAP values
    model, scaler = load_model()

    if model is not None:
        # Create sample data for SHAP
        feature_names = [
            'grid_position', 'circuit_difficulty', 'driver_code', 'constructor_code', 'position_change', 
            'recent_driver_form', 'constructor_win_rate', 'circuit_familiarity',
            'driver_experience', 'constructor_experience', 'grid_importance', 
            'weather_impact', 'circuit_performance', 'time_weight',
            'team_consistency', 'driver_team_synergy', 'form_position', 
            'experience_familiarity', 'constructor_strength', 'form_experience', 
            'team_circuit', 'grid_weather'
        ]

        # Create dummy data for SHAP visualization
        X_sample = pd.DataFrame(np.random.rand(10, len(feature_names)), columns=feature_names)
        X_sample_scaled = scaler.transform(X_sample)

        # Get base model for feature importance
        if hasattr(model, 'base_estimator_'):
            base_model = model.base_estimator_
        else:
            base_model = model

        # For stacking classifier
        if hasattr(base_model, 'named_estimators_'):
            rf_model = base_model.named_estimators_['rf']

            # Plot feature importance
            importances = pd.DataFrame({
                'feature': feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            fig = px.bar(
                importances.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance",
                labels={'importance': 'Importance', 'feature': 'Feature'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Add explanation
            with st.expander("Feature Explanation"):
                st.markdown("""
                ### Key Features Explained

                - **grid_position**: Starting position on the grid
                - **recent_driver_form**: Performance in recent races
                - **constructor_win_rate**: Team's historical win percentage
                - **circuit_familiarity**: Driver's experience on this track
                - **form_position**: Interaction between recent form and grid position
                - **weather_impact**: Effect of weather conditions on performance
                - **driver_team_synergy**: How well driver performs with current team
                - **constructor_strength**: Combined measure of team capability
                - **circuit_performance**: Historical performance at this circuit
                - **grid_importance**: How important grid position is at this track
                """)
    else:
        st.warning("Model not loaded. Feature importance unavailable.")

    # Save prediction to history
    if st.session_state.selected_race is not None and not predictions_df.empty:
        race_predictions = predictions_df[predictions_df['race_name'] == st.session_state.selected_race]

        if not race_predictions.empty:
            # Add to session state
            prediction_entry = {
                'race_name': st.session_state.selected_race,
                'date': pd.to_datetime(race_predictions['date'].iloc[0]).strftime('%Y-%m-%d'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model': st.session_state.selected_model,
                'top3': race_predictions.nlargest(3, 'probability')[['driver', 'team', 'probability']].to_dict('records')
            }

            # Check if this prediction is already in history
            if not any(entry['race_name'] == prediction_entry['race_name'] and 
                      entry['model'] == prediction_entry['model'] for entry in st.session_state.prediction_history):
                st.session_state.prediction_history.append(prediction_entry)

                # Save to CSV
                if not log_prediction_to_csv(prediction_entry):
                    st.error("Error saving prediction to CSV")

    # Display recent prediction logs
    st.write("---")
    st.subheader("Recent Prediction History")

    recent_predictions = load_recent_predictions(limit=5)
    if recent_predictions:
        # Create a DataFrame for display
        history_df = pd.DataFrame(recent_predictions)

        # Format the data column to show top drivers
        def format_data(data_str):
            try:
                data = json.loads(data_str)
                return ", ".join([f"{d['driver']} ({d['probability']:.1%})" for d in data[:3]])
            except:
                return data_str

        if 'data' in history_df.columns:
            history_df['top_drivers'] = history_df['data'].apply(format_data)
            display_df = history_df[['race_name', 'model', 'timestamp', 'top_drivers']]
            display_df.columns = ['Race', 'Model Used', 'Timestamp', 'Top Drivers']
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No prediction history available yet. Make predictions to see them here.")

if __name__ == "__main__":
    main()
