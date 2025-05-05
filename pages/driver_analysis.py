import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Driver & Constructor Analysis - F1 Race Outcome Predictor 2025",
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
def load_driver_standings():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/driver_standings.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading driver standings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_constructor_standings():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/constructor_standings.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading constructor standings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_qualifying():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/qualifying.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading qualifying data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_results():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/results.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading results data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_drivers():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/drivers.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading drivers data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_constructors():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/constructors.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading constructors data: {e}")
        return pd.DataFrame()

# Generate synthetic ELO ratings for demonstration
@st.cache_data
def generate_elo_ratings():
    drivers = [
        "M. Verstappen", "S. Perez", "C. Leclerc", "C. Sainz", 
        "L. Hamilton", "G. Russell", "L. Norris", "O. Piastri",
        "F. Alonso", "L. Stroll", "E. Ocon", "P. Gasly",
        "A. Albon", "L. Sargeant", "D. Ricciardo", "Y. Tsunoda",
        "V. Bottas", "Z. Guanyu", "K. Magnussen", "N. Hulkenberg"
    ]

    # Base ELO ratings (higher for top drivers)
    base_ratings = {
        "M. Verstappen": 1600, "S. Perez": 1450, "C. Leclerc": 1550, "C. Sainz": 1500,
        "L. Hamilton": 1580, "G. Russell": 1520, "L. Norris": 1540, "O. Piastri": 1480,
        "F. Alonso": 1530, "L. Stroll": 1400, "E. Ocon": 1420, "P. Gasly": 1430,
        "A. Albon": 1410, "L. Sargeant": 1350, "D. Ricciardo": 1440, "Y. Tsunoda": 1400,
        "V. Bottas": 1450, "Z. Guanyu": 1370, "K. Magnussen": 1390, "N. Hulkenberg": 1400
    }

    # Generate 10 races worth of ELO progression
    races = list(range(1, 11))
    data = []

    for driver in drivers:
        rating = base_ratings[driver]
        driver_ratings = [rating]

        # Add some random variation for each race
        for i in range(1, 10):
            # Top drivers tend to improve or maintain rating
            if driver in ["M. Verstappen", "L. Hamilton", "C. Leclerc", "L. Norris"]:
                change = np.random.normal(10, 15)
            # Mid-field drivers have more variation
            elif driver in ["F. Alonso", "S. Perez", "C. Sainz", "G. Russell", "O. Piastri"]:
                change = np.random.normal(0, 25)
            # Bottom drivers tend to lose rating
            else:
                change = np.random.normal(-5, 20)

            rating += change
            driver_ratings.append(rating)

        for i, race in enumerate(races):
            data.append({
                "driver": driver,
                "race": race,
                "elo_rating": driver_ratings[i]
            })

    return pd.DataFrame(data)

# Generate synthetic driver skills for radar charts
@st.cache_data
def generate_driver_skills():
    drivers = [
        "M. Verstappen", "S. Perez", "C. Leclerc", "C. Sainz", 
        "L. Hamilton", "G. Russell", "L. Norris", "O. Piastri",
        "F. Alonso", "L. Stroll", "E. Ocon", "P. Gasly",
        "A. Albon", "L. Sargeant", "D. Ricciardo", "Y. Tsunoda",
        "V. Bottas", "Z. Guanyu", "K. Magnussen", "N. Hulkenberg"
    ]

    skills = ["Qualifying", "Race Pace", "Overtaking", "Wet Weather", "Consistency"]

    data = []

    # Define skill profiles for different driver tiers
    skill_profiles = {
        "top": {
            "Qualifying": (85, 100),
            "Race Pace": (85, 100),
            "Overtaking": (80, 100),
            "Wet Weather": (80, 100),
            "Consistency": (85, 100)
        },
        "upper_mid": {
            "Qualifying": (75, 90),
            "Race Pace": (75, 90),
            "Overtaking": (70, 90),
            "Wet Weather": (70, 90),
            "Consistency": (75, 90)
        },
        "lower_mid": {
            "Qualifying": (65, 80),
            "Race Pace": (65, 80),
            "Overtaking": (60, 80),
            "Wet Weather": (60, 80),
            "Consistency": (65, 80)
        },
        "bottom": {
            "Qualifying": (50, 70),
            "Race Pace": (50, 70),
            "Overtaking": (50, 70),
            "Wet Weather": (50, 70),
            "Consistency": (50, 70)
        }
    }

    # Assign tiers to drivers
    driver_tiers = {
        "M. Verstappen": "top", "L. Hamilton": "top", "C. Leclerc": "top", "L. Norris": "top",
        "G. Russell": "upper_mid", "C. Sainz": "upper_mid", "F. Alonso": "upper_mid", "O. Piastri": "upper_mid", "S. Perez": "upper_mid",
        "P. Gasly": "lower_mid", "E. Ocon": "lower_mid", "D. Ricciardo": "lower_mid", "A. Albon": "lower_mid", "V. Bottas": "lower_mid",
        "L. Stroll": "bottom", "Y. Tsunoda": "bottom", "K. Magnussen": "bottom", "N. Hulkenberg": "bottom", "Z. Guanyu": "bottom", "L. Sargeant": "bottom"
    }

    # Generate skill values for each driver
    for driver in drivers:
        tier = driver_tiers[driver]
        profile = skill_profiles[tier]

        for skill in skills:
            min_val, max_val = profile[skill]
            value = np.random.randint(min_val, max_val + 1)

            data.append({
                "driver": driver,
                "skill": skill,
                "value": value
            })

    return pd.DataFrame(data)

# Generate synthetic qualifying data
@st.cache_data
def generate_qualifying_data():
    drivers = [
        "M. Verstappen", "S. Perez", "C. Leclerc", "C. Sainz", 
        "L. Hamilton", "G. Russell", "L. Norris", "O. Piastri",
        "F. Alonso", "L. Stroll", "E. Ocon", "P. Gasly",
        "A. Albon", "L. Sargeant", "D. Ricciardo", "Y. Tsunoda",
        "V. Bottas", "Z. Guanyu", "K. Magnussen", "N. Hulkenberg"
    ]

    constructors = {
        "M. Verstappen": "Red Bull Racing", "S. Perez": "Red Bull Racing",
        "C. Leclerc": "Ferrari", "C. Sainz": "Ferrari",
        "L. Hamilton": "Mercedes", "G. Russell": "Mercedes",
        "L. Norris": "McLaren", "O. Piastri": "McLaren",
        "F. Alonso": "Aston Martin", "L. Stroll": "Aston Martin",
        "E. Ocon": "Alpine", "P. Gasly": "Alpine",
        "A. Albon": "Williams", "L. Sargeant": "Williams",
        "D. Ricciardo": "RB", "Y. Tsunoda": "RB",
        "V. Bottas": "Sauber", "Z. Guanyu": "Sauber",
        "K. Magnussen": "Haas F1 Team", "N. Hulkenberg": "Haas F1 Team"
    }

    # Define average qualifying positions for each driver
    avg_positions = {
        "M. Verstappen": 1.5, "S. Perez": 5.5, "C. Leclerc": 3.0, "C. Sainz": 4.5,
        "L. Hamilton": 4.0, "G. Russell": 5.0, "L. Norris": 3.5, "O. Piastri": 6.0,
        "F. Alonso": 7.0, "L. Stroll": 12.0, "E. Ocon": 11.0, "P. Gasly": 10.0,
        "A. Albon": 13.0, "L. Sargeant": 18.0, "D. Ricciardo": 14.0, "Y. Tsunoda": 15.0,
        "V. Bottas": 16.0, "Z. Guanyu": 17.0, "K. Magnussen": 15.5, "N. Hulkenberg": 14.5
    }

    races = list(range(1, 6))  # Last 5 races
    data = []

    for race in races:
        # Shuffle positions around the average
        positions = {}
        for driver in drivers:
            avg_pos = avg_positions[driver]
            # Add some random variation
            pos = max(1, int(np.random.normal(avg_pos, 1.5)))
            positions[driver] = pos

        # Ensure no duplicate positions by adjusting
        used_positions = set()
        for driver in sorted(drivers, key=lambda d: positions[d]):
            pos = positions[driver]
            while pos in used_positions:
                pos += 1
            positions[driver] = pos
            used_positions.add(pos)

        # Add to dataset
        for driver in drivers:
            data.append({
                "race": race,
                "driver": driver,
                "constructor": constructors[driver],
                "position": positions[driver]
            })

    return pd.DataFrame(data)

# Main content
def main():
    # Apply CSS and theme
    local_css()
    apply_theme()

    # Page title
    st.title("📊 Driver & Constructor Performance Analysis")
    st.write("Analyze and compare driver and constructor performance metrics")

    # Load data
    drivers_df = load_drivers()
    constructors_df = load_constructors()

    # Generate synthetic data for demonstration
    elo_ratings_df = generate_elo_ratings()
    driver_skills_df = generate_driver_skills()
    qualifying_df = generate_qualifying_data()

    # Driver selection
    st.subheader("Select Drivers to Compare")

    # Get unique drivers
    unique_drivers = qualifying_df['driver'].unique()

    # Multi-select for drivers
    selected_drivers = st.multiselect(
        "Choose drivers to compare:",
        options=unique_drivers,
        default=["M. Verstappen", "L. Hamilton", "C. Leclerc"]
    )

    if not selected_drivers:
        st.warning("Please select at least one driver to analyze")
        return

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Qualifying Performance", 
        "Driver vs Constructor Influence", 
        "ELO Rating Progression",
        "Driver Skill Breakdown"
    ])

    with tab1:
        st.subheader("Average Qualifying Position (Last 5 Races)")

        # Filter qualifying data for selected drivers
        filtered_qualifying = qualifying_df[qualifying_df['driver'].isin(selected_drivers)]

        # Calculate average qualifying position per driver
        avg_qualifying = filtered_qualifying.groupby('driver')['position'].mean().reset_index()
        avg_qualifying = avg_qualifying.sort_values('position')

        # Create bar chart
        fig = px.bar(
            avg_qualifying,
            x='driver',
            y='position',
            title="Average Qualifying Position (Last 5 Races)",
            labels={'position': 'Avg. Grid Position', 'driver': 'Driver'},
            color='driver',
            text_auto='.1f'
        )
        fig.update_layout(yaxis_autorange="reversed")  # Lower position is better
        st.plotly_chart(fig, use_container_width=True)

        # Line chart showing qualifying position over races
        st.subheader("Qualifying Position Trend")

        # Create line chart
        fig = px.line(
            filtered_qualifying,
            x='race',
            y='position',
            color='driver',
            title="Qualifying Position by Race",
            labels={'position': 'Grid Position', 'race': 'Race', 'driver': 'Driver'},
            markers=True
        )
        fig.update_layout(yaxis_autorange="reversed")  # Lower position is better
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Driver vs Constructor Influence")

        # Filter qualifying data for selected drivers
        filtered_qualifying = qualifying_df[qualifying_df['driver'].isin(selected_drivers)]

        # Calculate average position by constructor
        avg_constructor = filtered_qualifying.groupby('constructor')['position'].mean().reset_index()

        # Calculate average position by driver
        avg_driver = filtered_qualifying.groupby('driver')['position'].mean().reset_index()

        # Create a combined dataframe for comparison
        comparison_data = []

        for driver in selected_drivers:
            driver_data = filtered_qualifying[filtered_qualifying['driver'] == driver]
            constructor = driver_data['constructor'].iloc[0]

            driver_avg = avg_driver[avg_driver['driver'] == driver]['position'].iloc[0]
            constructor_avg = avg_constructor[avg_constructor['constructor'] == constructor]['position'].iloc[0]

            # Calculate driver advantage/disadvantage compared to constructor average
            driver_effect = constructor_avg - driver_avg

            comparison_data.append({
                'driver': driver,
                'constructor': constructor,
                'driver_avg': driver_avg,
                'constructor_avg': constructor_avg,
                'driver_effect': driver_effect
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Create bar chart showing driver effect
        fig = px.bar(
            comparison_df,
            x='driver',
            y='driver_effect',
            title="Driver Effect on Qualifying (Compared to Constructor Average)",
            labels={'driver_effect': 'Positions Gained/Lost vs Constructor Avg', 'driver': 'Driver'},
            color='constructor',
            text_auto='.2f'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Create a grouped bar chart comparing driver and constructor averages
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=comparison_df['driver'],
            y=comparison_df['driver_avg'],
            name='Driver Average',
            marker_color='blue'
        ))

        fig.add_trace(go.Bar(
            x=comparison_df['driver'],
            y=comparison_df['constructor_avg'],
            name='Constructor Average',
            marker_color='red'
        ))

        fig.update_layout(
            title="Driver vs Constructor Average Qualifying Position",
            xaxis_title="Driver",
            yaxis_title="Average Position",
            yaxis_autorange="reversed",  # Lower position is better
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.info("""
        **Understanding Driver vs Constructor Influence:**

        - **Positive Driver Effect**: Driver qualifies better than the constructor average, suggesting the driver is outperforming the car.
        - **Negative Driver Effect**: Driver qualifies worse than the constructor average, suggesting the driver is underperforming relative to the car's potential.
        - **Near Zero**: Driver performance aligns with the constructor average.
        """)

    with tab3:
        st.subheader("ELO Rating Progression")

        # Filter ELO data for selected drivers
        filtered_elo = elo_ratings_df[elo_ratings_df['driver'].isin(selected_drivers)]

        # Create line chart
        fig = px.line(
            filtered_elo,
            x='race',
            y='elo_rating',
            color='driver',
            title="ELO Rating Progression",
            labels={'elo_rating': 'ELO Rating', 'race': 'Race', 'driver': 'Driver'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add explanation
        st.info("""
        **Understanding ELO Ratings:**

        ELO is a rating system that calculates relative skill levels between competitors. In F1:

        - **Higher Rating**: Indicates better performance over time
        - **Upward Trend**: Driver is improving or performing consistently well
        - **Downward Trend**: Driver is struggling or facing challenges
        - **Stable Rating**: Consistent performance at the same level

        ELO takes into account the quality of opponents and expected results based on previous performance.
        """)

    with tab4:
        st.subheader("Driver Skill Breakdown")

        # Create tabs for each selected driver
        if len(selected_drivers) > 0:
            driver_tabs = st.tabs(selected_drivers)

            for i, driver in enumerate(selected_drivers):
                with driver_tabs[i]:
                    # Filter skills data for this driver
                    driver_skills = driver_skills_df[driver_skills_df['driver'] == driver]

                    # Create radar chart
                    categories = driver_skills['skill'].tolist()
                    values = driver_skills['value'].tolist()

                    fig = go.Figure()

                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=driver
                    ))

                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        title=f"{driver} Skill Breakdown"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Display skill values in a table
                    st.write("Skill Ratings (0-100):")

                    # Create 3 columns
                    cols = st.columns(len(categories))

                    for j, (skill, value) in enumerate(zip(categories, values)):
                        with cols[j]:
                            st.metric(label=skill, value=value)
        else:
            st.warning("Please select at least one driver to view skill breakdown")

if __name__ == "__main__":
    main()
