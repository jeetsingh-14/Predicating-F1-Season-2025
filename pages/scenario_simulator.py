import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys
from PIL import Image
import random
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Scenario Simulator - F1 Race Outcome Predictor 2025",
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
    </style>
    """, unsafe_allow_html=True)

# Apply theme based on dark mode setting
def apply_theme():
    if "dark_mode" in st.session_state and st.session_state.dark_mode:
        st.markdown('<div class="dark-mode">', unsafe_allow_html=True)
    else:
        st.markdown('<div class="light-mode">', unsafe_allow_html=True)

# Load data functions
@st.cache_data(ttl=3600)
def load_upcoming_races():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "./data/processed/upcoming_races.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading upcoming races: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_predictions():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "results/2025_predictions_full.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

# Load model
@st.cache_resource
def load_model():
    try:
        base_dir = Path(__file__).parent.parent
        model_path = os.path.join(base_dir, "models/stacking_model.pkl")
        scaler_path = os.path.join(base_dir, "models/scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Generate synthetic feature data for a driver
def generate_driver_features(driver_name, team_name):
    # Define base feature values based on driver tier
    top_drivers = ["M. Verstappen", "L. Hamilton", "C. Leclerc", "L. Norris"]
    upper_mid_drivers = ["G. Russell", "C. Sainz", "F. Alonso", "O. Piastri", "S. Perez"]

    if driver_name in top_drivers:
        tier = "top"
    elif driver_name in upper_mid_drivers:
        tier = "upper_mid"
    else:
        tier = "lower"

    # Generate driver_code from driver_name (first initial + first 3 letters of last name)
    parts = driver_name.split('. ')
    if len(parts) == 2:
        first_initial = parts[0].lower()
        last_name = parts[1].lower()[:3]
        driver_code = first_initial + last_name
    else:
        # Fallback if name format is different
        driver_code = driver_name.lower().replace(' ', '')[:4]

    # Generate constructor_code from team_name
    constructor_code = team_name.lower().replace(' ', '_')

    # Base feature values by tier
    if tier == "top":
        base_features = {
            'grid_position': np.random.randint(1, 4),
            'circuit_difficulty': 0.6,
            'position_change': 1.5,
            'recent_driver_form': 0.85,
            'constructor_win_rate': 0.7 if team_name in ["Red Bull Racing", "Ferrari", "Mercedes"] else 0.4,
            'circuit_familiarity': 0.8,
            'driver_experience': 0.9,
            'constructor_experience': 0.85,
            'grid_importance': 0.75,
            'weather_impact': 0.6,
            'circuit_performance': 0.8,
            'time_weight': 0.9,
            'team_consistency': 0.85,
            'driver_team_synergy': 0.9,
            'driver_code': driver_code,
            'constructor_code': constructor_code
        }
    elif tier == "upper_mid":
        base_features = {
            'grid_position': np.random.randint(3, 7),
            'circuit_difficulty': 0.6,
            'position_change': 1.0,
            'recent_driver_form': 0.75,
            'constructor_win_rate': 0.6 if team_name in ["Red Bull Racing", "Ferrari", "Mercedes"] else 0.3,
            'circuit_familiarity': 0.7,
            'driver_experience': 0.8,
            'constructor_experience': 0.75,
            'grid_importance': 0.7,
            'weather_impact': 0.5,
            'circuit_performance': 0.7,
            'time_weight': 0.8,
            'team_consistency': 0.75,
            'driver_team_synergy': 0.8,
            'driver_code': driver_code,
            'constructor_code': constructor_code
        }
    else:  # lower tier
        base_features = {
            'grid_position': np.random.randint(8, 15),
            'circuit_difficulty': 0.6,
            'position_change': 0.5,
            'recent_driver_form': 0.6,
            'constructor_win_rate': 0.2,
            'circuit_familiarity': 0.6,
            'driver_experience': 0.7,
            'constructor_experience': 0.6,
            'grid_importance': 0.65,
            'weather_impact': 0.4,
            'circuit_performance': 0.5,
            'time_weight': 0.7,
            'team_consistency': 0.6,
            'driver_team_synergy': 0.7,
            'driver_code': driver_code,
            'constructor_code': constructor_code
        }

    # Add interaction features
    base_features['form_position'] = base_features['recent_driver_form'] * (1 / base_features['grid_position'])
    base_features['experience_familiarity'] = base_features['driver_experience'] * base_features['circuit_familiarity']
    base_features['constructor_strength'] = base_features['constructor_win_rate'] * base_features['constructor_experience']
    base_features['form_experience'] = base_features['recent_driver_form'] * base_features['driver_experience']
    base_features['team_circuit'] = base_features['constructor_win_rate'] * base_features['circuit_familiarity']
    base_features['grid_weather'] = base_features['grid_importance'] * base_features['weather_impact']

    return base_features

# Predict podium probability based on features
def predict_podium_probability(features, model, scaler):
    # If we don't have a real model, use a simple heuristic
    if model is None or scaler is None:
        # Simple heuristic based on grid position and constructor win rate
        grid_pos = features['grid_position']
        win_rate = features['constructor_win_rate']
        form = features['recent_driver_form']

        # Base probability decreases with grid position
        base_prob = max(0, 1 - (grid_pos / 20))

        # Adjust based on constructor win rate and driver form
        prob = base_prob * 0.4 + win_rate * 0.3 + form * 0.3

        # Add some randomness
        prob = min(1.0, max(0.0, prob + np.random.normal(0, 0.05)))

        return prob
    else:
        # Convert features to DataFrame
        feature_df = pd.DataFrame([features])

        # Scale features
        X_scaled = scaler.transform(feature_df)

        # Predict probability
        prob = model.predict_proba(X_scaled)[0, 1]

        return prob

# Main content
def main():
    # Apply CSS and theme
    local_css()
    apply_theme()

    # Page title
    st.title("🔮 Scenario Simulator")
    st.write("Simulate different race scenarios and see how they affect podium predictions")

    # Load data
    upcoming_races_df = load_upcoming_races()
    predictions_df = load_predictions()
    model, scaler = load_model()

    # What-If Engine
    st.subheader("What-If Engine")
    st.write("Adjust driver and race parameters to see how they affect podium probability")

    # Race selection
    if not upcoming_races_df.empty:
        unique_races = upcoming_races_df['race_name'].unique()

        selected_race = st.selectbox(
            "Select a race:",
            options=unique_races,
            index=0 if "selected_race" not in st.session_state else list(unique_races).index(st.session_state.selected_race)
        )

        st.session_state.selected_race = selected_race

        # Get race date
        race_date = upcoming_races_df[upcoming_races_df['race_name'] == selected_race]['date'].iloc[0]
        st.write(f"Race Date: {pd.to_datetime(race_date).strftime('%B %d, %Y')}")

        # Driver selection
        race_drivers = upcoming_races_df[upcoming_races_df['race_name'] == selected_race]
        driver_options = [f"{row['driver']} ({row['team']})" for _, row in race_drivers.iterrows()]

        selected_driver_option = st.selectbox(
            "Select a driver:",
            options=driver_options,
            index=0
        )

        # Extract driver name and team
        driver_name = selected_driver_option.split(" (")[0]
        team_name = selected_driver_option.split("(")[1].replace(")", "")

        # Get base features for the driver
        base_features = generate_driver_features(driver_name, team_name)

        # Display current podium probability
        base_probability = predict_podium_probability(base_features, model, scaler)

        # Create columns for layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Adjust Parameters")

            # Create sliders for key features
            st.markdown("<div class='slider-label'>Grid Position</div>", unsafe_allow_html=True)
            grid_position = st.slider(
                "Grid Position",
                min_value=1,
                max_value=20,
                value=int(base_features['grid_position']),
                key="grid_position_slider",
                label_visibility="collapsed"
            )

            st.markdown("<div class='slider-label'>Recent Driver Form (0-1)</div>", unsafe_allow_html=True)
            recent_form = st.slider(
                "Recent Driver Form",
                min_value=0.0,
                max_value=1.0,
                value=float(base_features['recent_driver_form']),
                step=0.01,
                key="recent_form_slider",
                label_visibility="collapsed"
            )

            st.markdown("<div class='slider-label'>Circuit Familiarity (0-1)</div>", unsafe_allow_html=True)
            circuit_familiarity = st.slider(
                "Circuit Familiarity",
                min_value=0.0,
                max_value=1.0,
                value=float(base_features['circuit_familiarity']),
                step=0.01,
                key="circuit_familiarity_slider",
                label_visibility="collapsed"
            )

            st.markdown("<div class='slider-label'>Weather Impact (0-1)</div>", unsafe_allow_html=True)
            weather_impact = st.slider(
                "Weather Impact",
                min_value=0.0,
                max_value=1.0,
                value=float(base_features['weather_impact']),
                step=0.01,
                key="weather_impact_slider",
                label_visibility="collapsed"
            )

            # Create expander for advanced parameters
            with st.expander("Advanced Parameters"):
                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("<div class='slider-label'>Driver Experience (0-1)</div>", unsafe_allow_html=True)
                    driver_experience = st.slider(
                        "Driver Experience",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(base_features['driver_experience']),
                        step=0.01,
                        key="driver_experience_slider",
                        label_visibility="collapsed"
                    )

                    st.markdown("<div class='slider-label'>Constructor Win Rate (0-1)</div>", unsafe_allow_html=True)
                    constructor_win_rate = st.slider(
                        "Constructor Win Rate",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(base_features['constructor_win_rate']),
                        step=0.01,
                        key="constructor_win_rate_slider",
                        label_visibility="collapsed"
                    )

                with col_b:
                    st.markdown("<div class='slider-label'>Team Consistency (0-1)</div>", unsafe_allow_html=True)
                    team_consistency = st.slider(
                        "Team Consistency",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(base_features['team_consistency']),
                        step=0.01,
                        key="team_consistency_slider",
                        label_visibility="collapsed"
                    )

                    st.markdown("<div class='slider-label'>Driver-Team Synergy (0-1)</div>", unsafe_allow_html=True)
                    driver_team_synergy = st.slider(
                        "Driver-Team Synergy",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(base_features['driver_team_synergy']),
                        step=0.01,
                        key="driver_team_synergy_slider",
                        label_visibility="collapsed"
                    )

            # Update features with slider values
            modified_features = base_features.copy()
            modified_features['grid_position'] = grid_position
            modified_features['recent_driver_form'] = recent_form
            modified_features['circuit_familiarity'] = circuit_familiarity
            modified_features['weather_impact'] = weather_impact

            # Update advanced parameters if expanded
            if 'driver_experience_slider' in st.session_state:
                modified_features['driver_experience'] = driver_experience
                modified_features['constructor_win_rate'] = constructor_win_rate
                modified_features['team_consistency'] = team_consistency
                modified_features['driver_team_synergy'] = driver_team_synergy

            # Recalculate interaction features
            modified_features['form_position'] = modified_features['recent_driver_form'] * (1 / modified_features['grid_position'])
            modified_features['experience_familiarity'] = modified_features['driver_experience'] * modified_features['circuit_familiarity']
            modified_features['constructor_strength'] = modified_features['constructor_win_rate'] * modified_features['constructor_experience']
            modified_features['form_experience'] = modified_features['recent_driver_form'] * modified_features['driver_experience']
            modified_features['team_circuit'] = modified_features['constructor_win_rate'] * modified_features['circuit_familiarity']
            modified_features['grid_weather'] = modified_features['grid_importance'] * modified_features['weather_impact']

            # Calculate new probability
            new_probability = predict_podium_probability(modified_features, model, scaler)

            # Calculate probability change
            probability_change = new_probability - base_probability

            # Create a DataFrame for visualization
            scenario_df = pd.DataFrame([
                {"Scenario": "Base", "Probability": base_probability},
                {"Scenario": "Modified", "Probability": new_probability}
            ])

            # Create bar chart
            fig = px.bar(
                scenario_df,
                x="Scenario",
                y="Probability",
                title=f"Podium Probability Comparison for {driver_name}",
                color="Scenario",
                color_discrete_map={"Base": "#1f77b4", "Modified": "#ff7f0e"},
                text_auto='.1%'
            )

            fig.update_layout(yaxis_range=[0, 1])

            st.plotly_chart(fig, use_container_width=True)

            # Show key changes
            st.subheader("Key Changes")

            changes = [
                {"Parameter": "Grid Position", "Base": base_features['grid_position'], "Modified": modified_features['grid_position']},
                {"Parameter": "Recent Form", "Base": f"{base_features['recent_driver_form']:.2f}", "Modified": f"{modified_features['recent_driver_form']:.2f}"},
                {"Parameter": "Circuit Familiarity", "Base": f"{base_features['circuit_familiarity']:.2f}", "Modified": f"{modified_features['circuit_familiarity']:.2f}"},
                {"Parameter": "Weather Impact", "Base": f"{base_features['weather_impact']:.2f}", "Modified": f"{modified_features['weather_impact']:.2f}"}
            ]

            changes_df = pd.DataFrame(changes)
            st.table(changes_df)

        with col2:
            # Display driver info card
            team_class = f"team-{team_name.lower().replace(' ', '-').replace('f1 team', '')}"

            st.markdown(f"""
            <div class="card {team_class}">
                <h3>{driver_name}</h3>
                <p>{team_name}</p>
            </div>
            """, unsafe_allow_html=True)

            # Display probability metrics
            st.subheader("Podium Probability")

            # Determine probability class
            prob_class = ""
            if new_probability >= 0.7:
                prob_class = "probability-high"
            elif new_probability >= 0.4:
                prob_class = "probability-medium"
            else:
                prob_class = "probability-low"

            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Current Probability</div>
                <div class="metric-value {prob_class}">{new_probability:.1%}</div>
                <div class="metric-label">Change from Base</div>
                <div class="metric-value">{probability_change:+.1%}</div>
            </div>
            """, unsafe_allow_html=True)

            # Add interpretation
            st.subheader("Interpretation")

            if new_probability >= 0.7:
                st.success(f"**Strong Podium Chance**: {driver_name} has a high probability of finishing on the podium with these parameters.")
            elif new_probability >= 0.4:
                st.warning(f"**Moderate Podium Chance**: {driver_name} has a reasonable chance of finishing on the podium, but it's not guaranteed.")
            else:
                st.error(f"**Low Podium Chance**: {driver_name} is unlikely to finish on the podium with these parameters.")

            # Add key insights
            st.subheader("Key Insights")

            insights = []

            if modified_features['grid_position'] <= 3:
                insights.append("Starting from the front row significantly increases podium chances.")
            elif modified_features['grid_position'] >= 10:
                insights.append("Starting outside the top 10 makes a podium finish challenging.")

            if modified_features['recent_driver_form'] >= 0.8:
                insights.append("Recent strong form is a positive indicator for podium finish.")
            elif modified_features['recent_driver_form'] <= 0.5:
                insights.append("Poor recent form reduces podium probability.")

            if modified_features['weather_impact'] >= 0.7:
                insights.append("Weather conditions could significantly affect race outcome.")

            if not insights:
                insights.append("No specific insights to highlight for this scenario.")

            for insight in insights:
                st.markdown(f"- {insight}")

            # Add GPT-powered summary
            st.subheader("AI Race Preview")

            # Generate a summary based on the parameters
            if new_probability >= 0.7:
                summary = f"Our model predicts {driver_name} has a {new_probability:.0%} chance of finishing on the podium at the {selected_race}. With a grid position of {grid_position}, strong recent form, and good circuit familiarity, the conditions are favorable for a strong result."
            elif new_probability >= 0.4:
                summary = f"Our model gives {driver_name} a {new_probability:.0%} chance of reaching the podium at the {selected_race}. Starting from P{grid_position} presents some challenges, but with the right strategy and race execution, a podium finish is within reach."
            else:
                summary = f"Our model suggests {driver_name} faces an uphill battle with only a {new_probability:.0%} chance of podium at the {selected_race}. Starting from P{grid_position} and with the current parameters, significant improvements or unusual race circumstances would be needed for a top-3 finish."

            st.info(summary)
    else:
        st.error("No upcoming races data available.")

    # Add note about Monte Carlo Simulator
    st.write("---")
    st.subheader("Race Outcome Simulator")
    st.info("The full Monte Carlo Race Outcome Simulator will be available in the next update. This feature will run thousands of race simulations to show the distribution of possible outcomes and estimate probabilities for various scenarios.")

if __name__ == "__main__":
    main()
