import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import folium
from streamlit_folium import folium_static
import pydeck as pdk
from PIL import Image
import random
from pathlib import Path

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Track Insights - F1 Race Outcome Predictor 2025",
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
    .weather-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 10px;
        border-radius: 8px;
        background-color: var(--card-bg-color);
        margin: 5px;
    }
    .weather-icon {
        font-size: 32px;
        margin-bottom: 10px;
    }
    .weather-temp {
        font-size: 20px;
        font-weight: bold;
    }
    .weather-desc {
        font-size: 14px;
        text-align: center;
    }
    .difficulty-low {
        color: #4CAF50;
    }
    .difficulty-medium {
        color: #FFC107;
    }
    .difficulty-high {
        color: #F44336;
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
        file_path = os.path.join(base_dir, "data/processed/upcoming_races.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading upcoming races: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_circuits():
    try:
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data/processed/circuits.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading circuits data: {e}")
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

# Generate synthetic track data
@st.cache_data
def generate_track_data():
    # Define track data
    tracks = {
        "Bahrain International Circuit": {
            "location": "Sakhir, Bahrain",
            "length_km": 5.412,
            "turns": 15,
            "lap_record": "1:31.447 - Pedro de la Rosa (2005)",
            "first_gp": 2004,
            "coordinates": (26.0325, 50.5106),
            "overtaking_difficulty": "Low",
            "safety_car_probability": 0.25,
            "weather_variability": "Low",
            "tire_degradation": "High",
            "key_overtaking_zones": ["Turn 1", "Turn 4", "Turn 11"],
            "drs_zones": 3,
            "track_type": "Permanent",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Minimal",
            "image_name": "Bahrain_Bahrain_International.png"
        },
        "Jeddah Corniche Circuit": {
            "location": "Jeddah, Saudi Arabia",
            "length_km": 6.174,
            "turns": 27,
            "lap_record": "1:30.734 - Lewis Hamilton (2021)",
            "first_gp": 2021,
            "coordinates": (21.6319, 39.1044),
            "overtaking_difficulty": "Medium",
            "safety_car_probability": 0.65,
            "weather_variability": "Low",
            "tire_degradation": "Medium",
            "key_overtaking_zones": ["Turn 1", "Turn 27"],
            "drs_zones": 3,
            "track_type": "Street",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Minimal",
            "image_name": "Saudi_Arabia_Jeddah_Corniche.png"
        },
        "Miami International Autodrome": {
            "location": "Miami, Florida, USA",
            "length_km": 5.412,
            "turns": 19,
            "lap_record": "1:29.708 - Max Verstappen (2023)",
            "first_gp": 2022,
            "coordinates": (25.9581, -80.2389),
            "overtaking_difficulty": "Medium",
            "safety_car_probability": 0.45,
            "weather_variability": "Medium",
            "tire_degradation": "Medium",
            "key_overtaking_zones": ["Turn 1", "Turn 11", "Turn 17"],
            "drs_zones": 3,
            "track_type": "Street",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Minimal",
            "image_name": "USA_Miami_International.png"
        },
        "Autodromo Enzo e Dino Ferrari": {
            "location": "Imola, Italy",
            "length_km": 4.909,
            "turns": 19,
            "lap_record": "1:15.484 - Lewis Hamilton (2020)",
            "first_gp": 1980,
            "coordinates": (44.3439, 11.7167),
            "overtaking_difficulty": "High",
            "safety_car_probability": 0.35,
            "weather_variability": "Medium",
            "tire_degradation": "Medium",
            "key_overtaking_zones": ["Turn 1", "Turn 7"],
            "drs_zones": 2,
            "track_type": "Permanent",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Moderate",
            "image_name": "Italy_Imola_Internazionale_Enzo_Dino_Ferrari.png"
        },
        "Circuit de Monaco": {
            "location": "Monte Carlo, Monaco",
            "length_km": 3.337,
            "turns": 19,
            "lap_record": "1:12.909 - Lewis Hamilton (2021)",
            "first_gp": 1950,
            "coordinates": (43.7347, 7.4206),
            "overtaking_difficulty": "Very High",
            "safety_car_probability": 0.60,
            "weather_variability": "Medium",
            "tire_degradation": "Low",
            "key_overtaking_zones": ["Turn 1", "Tunnel Exit"],
            "drs_zones": 1,
            "track_type": "Street",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Significant",
            "image_name": "Monaco_Circuit_de_Monaco.png"
        },
        "Circuit de Barcelona-Catalunya": {
            "location": "Barcelona, Spain",
            "length_km": 4.675,
            "turns": 16,
            "lap_record": "1:18.149 - Max Verstappen (2021)",
            "first_gp": 1991,
            "coordinates": (41.5638, 2.2585),
            "overtaking_difficulty": "High",
            "safety_car_probability": 0.20,
            "weather_variability": "Low",
            "tire_degradation": "High",
            "key_overtaking_zones": ["Turn 1", "Turn 10"],
            "drs_zones": 2,
            "track_type": "Permanent",
            "surface_type": "Smooth asphalt",
            "elevation_change": "Moderate",
            "image_name": "Spain_Barcelona_Catalunya.png"
        }
    }

    # Add more tracks as needed

    return tracks

# Generate synthetic weather data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def generate_weather_forecast(track_name):
    # Define base weather patterns for different regions
    weather_patterns = {
        "Bahrain International Circuit": {
            "temp_range": (25, 35),
            "conditions": ["Clear", "Sunny", "Partly Cloudy"],
            "rain_probability": 0.05,
            "wind_range": (5, 15)
        },
        "Jeddah Corniche Circuit": {
            "temp_range": (25, 35),
            "conditions": ["Clear", "Sunny", "Partly Cloudy"],
            "rain_probability": 0.05,
            "wind_range": (5, 20)
        },
        "Miami International Autodrome": {
            "temp_range": (22, 32),
            "conditions": ["Partly Cloudy", "Cloudy", "Sunny", "Scattered Showers"],
            "rain_probability": 0.25,
            "wind_range": (5, 15)
        },
        "Autodromo Enzo e Dino Ferrari": {
            "temp_range": (15, 25),
            "conditions": ["Partly Cloudy", "Cloudy", "Sunny", "Light Rain"],
            "rain_probability": 0.30,
            "wind_range": (5, 15)
        },
        "Circuit de Monaco": {
            "temp_range": (18, 25),
            "conditions": ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"],
            "rain_probability": 0.20,
            "wind_range": (5, 10)
        },
        "Circuit de Barcelona-Catalunya": {
            "temp_range": (18, 28),
            "conditions": ["Sunny", "Partly Cloudy", "Cloudy"],
            "rain_probability": 0.15,
            "wind_range": (5, 15)
        }
    }

    # Default pattern if track not found
    default_pattern = {
        "temp_range": (15, 30),
        "conditions": ["Sunny", "Partly Cloudy", "Cloudy", "Light Rain"],
        "rain_probability": 0.20,
        "wind_range": (5, 15)
    }

    pattern = weather_patterns.get(track_name, default_pattern)

    # Generate 3-day forecast
    forecast = []
    today = datetime.now()

    for i in range(3):
        day = today + timedelta(days=i)

        # Temperature varies by time of day
        if i == 0:  # Today
            temps = [
                random.randint(pattern["temp_range"][0], pattern["temp_range"][1] - 5),  # Morning
                random.randint(pattern["temp_range"][0] + 5, pattern["temp_range"][1]),  # Afternoon
                random.randint(pattern["temp_range"][0], pattern["temp_range"][1] - 8)   # Evening
            ]
        else:
            # Slightly more variation for future days
            base_temp = random.randint(pattern["temp_range"][0], pattern["temp_range"][1])
            variation = random.randint(-3, 3)
            temps = [
                max(pattern["temp_range"][0], base_temp - 5 + variation),  # Morning
                min(pattern["temp_range"][1], base_temp + variation),      # Afternoon
                max(pattern["temp_range"][0], base_temp - 8 + variation)   # Evening
            ]

        # Determine if it will rain
        will_rain = random.random() < pattern["rain_probability"]

        # Select conditions
        if will_rain:
            conditions = ["Light Rain", "Scattered Showers", "Thunderstorms"]
            main_condition = random.choice(conditions)
        else:
            main_condition = random.choice(pattern["conditions"])

        # Wind speed
        wind_speed = random.randint(pattern["wind_range"][0], pattern["wind_range"][1])

        # Create day forecast
        day_forecast = {
            "date": day.strftime("%Y-%m-%d"),
            "day_name": day.strftime("%A"),
            "conditions": main_condition,
            "morning_temp": temps[0],
            "afternoon_temp": temps[1],
            "evening_temp": temps[2],
            "wind_speed": wind_speed,
            "precipitation": random.randint(0, 80) if will_rain else 0
        }

        forecast.append(day_forecast)

    return forecast

# Generate historical safety car data
@st.cache_data
def generate_safety_car_history(track_name):
    # Define base safety car probabilities and historical data
    safety_car_data = {
        "Bahrain International Circuit": {
            "probability": 0.25,
            "history": [1, 0, 1, 0, 0]  # Last 5 years (1 = safety car, 0 = no safety car)
        },
        "Jeddah Corniche Circuit": {
            "probability": 0.65,
            "history": [1, 1, 1, 0, 1]
        },
        "Miami International Autodrome": {
            "probability": 0.45,
            "history": [1, 0, 1, 0, 0]
        },
        "Autodromo Enzo e Dino Ferrari": {
            "probability": 0.35,
            "history": [0, 1, 0, 0, 1]
        },
        "Circuit de Monaco": {
            "probability": 0.60,
            "history": [1, 1, 0, 1, 1]
        },
        "Circuit de Barcelona-Catalunya": {
            "probability": 0.20,
            "history": [0, 0, 1, 0, 0]
        }
    }

    # Default data if track not found
    default_data = {
        "probability": 0.30,
        "history": [0, 1, 0, 0, 1]
    }

    return safety_car_data.get(track_name, default_data)

# Main content
def main():
    # Apply CSS and theme
    local_css()
    apply_theme()

    # Page title
    st.title("🏁 Track Insights")
    st.write("Explore detailed information about Formula 1 circuits and race conditions")

    # Load data
    upcoming_races_df = load_upcoming_races()
    track_data = generate_track_data()

    # Get unique circuits from upcoming races
    if not upcoming_races_df.empty:
        unique_circuits = upcoming_races_df['circuit'].unique()

        # Track selection
        selected_circuit = st.selectbox(
            "Select a circuit:",
            options=unique_circuits,
            index=0 if "selected_circuit" not in st.session_state else list(unique_circuits).index(st.session_state.selected_circuit)
        )

        st.session_state.selected_circuit = selected_circuit

        # Get associated race
        race_info = upcoming_races_df[upcoming_races_df['circuit'] == selected_circuit].iloc[0]
        race_name = race_info['race_name']
        race_date = pd.to_datetime(race_info['date'])

        st.subheader(f"{race_name} - {race_date.strftime('%B %d, %Y')}")

        # Check if we have data for this track
        if selected_circuit in track_data:
            track_info = track_data[selected_circuit]

            # Create two columns for layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # Display track image
                try:
                    base_dir = Path(__file__).parent.parent
                    image_path = os.path.join(base_dir, "dashboad_content", "F1 Race Tracks", track_info['image_name'])
                    image = Image.open(image_path)
                    st.image(image, caption=f"{selected_circuit} Layout", use_column_width=True)
                except Exception as e:
                    st.warning(f"Track layout image not available: {e}")

                # Track map
                st.subheader("Interactive Track Map")

                # Create a folium map centered at the track coordinates
                m = folium.Map(location=track_info["coordinates"], zoom_start=14)

                # Add a marker for the track
                folium.Marker(
                    location=track_info["coordinates"],
                    popup=selected_circuit,
                    icon=folium.Icon(color="red", icon="flag", prefix="fa")
                ).add_to(m)

                # Add a circle to highlight the area
                folium.Circle(
                    location=track_info["coordinates"],
                    radius=1000,  # 1km radius
                    color="#3186cc",
                    fill=True,
                    fill_color="#3186cc"
                ).add_to(m)

                # Display the map
                folium_static(m)

                # Alternative 3D map with pydeck
                st.subheader("3D Track View")

                # Create a deck.gl map
                view_state = pdk.ViewState(
                    latitude=track_info["coordinates"][0],
                    longitude=track_info["coordinates"][1],
                    zoom=14,
                    pitch=45,
                    bearing=0
                )

                # Create a layer for the track
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=[{"position": [track_info["coordinates"][1], track_info["coordinates"][0]], "size": 100}],
                    get_position="position",
                    get_color=[255, 0, 0],
                    get_radius="size",
                    pickable=True
                )

                # Render the map
                st.pydeck_chart(pdk.Deck(
                    map_style="mapbox://styles/mapbox/satellite-streets-v11",
                    initial_view_state=view_state,
                    layers=[layer]
                ))

            with col2:
                # Track details card
                st.subheader("Track Details")

                st.markdown(f"""
                <div class="card">
                    <p><strong>Location:</strong> {track_info['location']}</p>
                    <p><strong>Length:</strong> {track_info['length_km']} km</p>
                    <p><strong>Turns:</strong> {track_info['turns']}</p>
                    <p><strong>Lap Record:</strong> {track_info['lap_record']}</p>
                    <p><strong>First Grand Prix:</strong> {track_info['first_gp']}</p>
                    <p><strong>Track Type:</strong> {track_info['track_type']}</p>
                    <p><strong>Surface:</strong> {track_info['surface_type']}</p>
                    <p><strong>Elevation Change:</strong> {track_info['elevation_change']}</p>
                    <p><strong>DRS Zones:</strong> {track_info['drs_zones']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Overtaking difficulty
                st.subheader("Race Characteristics")

                # Determine CSS class for difficulty
                difficulty_class = ""
                if track_info['overtaking_difficulty'] == "Low":
                    difficulty_class = "difficulty-low"
                elif track_info['overtaking_difficulty'] == "Medium":
                    difficulty_class = "difficulty-medium"
                else:
                    difficulty_class = "difficulty-high"

                st.markdown(f"""
                <div class="card">
                    <p><strong>Overtaking Difficulty:</strong> <span class="{difficulty_class}">{track_info['overtaking_difficulty']}</span></p>
                    <p><strong>Safety Car Probability:</strong> {track_info['safety_car_probability'] * 100:.0f}%</p>
                    <p><strong>Tire Degradation:</strong> {track_info['tire_degradation']}</p>
                    <p><strong>Weather Variability:</strong> {track_info['weather_variability']}</p>
                    <p><strong>Key Overtaking Zones:</strong> {', '.join(track_info['key_overtaking_zones'])}</p>
                </div>
                """, unsafe_allow_html=True)

                # Safety car history
                safety_car_data = generate_safety_car_history(selected_circuit)

                st.subheader("Safety Car History")

                # Calculate percentage
                sc_percentage = sum(safety_car_data['history']) / len(safety_car_data['history']) * 100

                # Create a bar chart for safety car history
                years = list(range(2020, 2025))
                sc_data = pd.DataFrame({
                    'Year': years,
                    'Safety Car': safety_car_data['history']
                })

                fig = px.bar(
                    sc_data,
                    x='Year',
                    y='Safety Car',
                    title=f"Safety Car Deployments (Last 5 Years): {sc_percentage:.0f}%",
                    labels={'Safety Car': 'Deployment', 'Year': 'Year'},
                    color='Safety Car',
                    color_discrete_map={0: 'lightgrey', 1: 'red'}
                )

                st.plotly_chart(fig, use_container_width=True)

            # Weather forecast section
            st.subheader("Weather Forecast")

            # Generate weather forecast
            weather_forecast = generate_weather_forecast(selected_circuit)

            # Display weather cards
            weather_cols = st.columns(len(weather_forecast))

            for i, day_forecast in enumerate(weather_forecast):
                with weather_cols[i]:
                    # Determine weather icon
                    weather_icon = "☀️"  # Default sunny
                    if "Rain" in day_forecast['conditions'] or "Showers" in day_forecast['conditions']:
                        weather_icon = "🌧️"
                    elif "Thunderstorms" in day_forecast['conditions']:
                        weather_icon = "⛈️"
                    elif "Cloudy" in day_forecast['conditions']:
                        if "Partly" in day_forecast['conditions']:
                            weather_icon = "⛅"
                        else:
                            weather_icon = "☁️"

                    # Create weather card
                    st.markdown(f"""
                    <div class="weather-card">
                        <div class="weather-day">{day_forecast['day_name']}</div>
                        <div class="weather-icon">{weather_icon}</div>
                        <div class="weather-temp">{day_forecast['afternoon_temp']}°C</div>
                        <div class="weather-desc">{day_forecast['conditions']}</div>
                        <div class="weather-desc">Wind: {day_forecast['wind_speed']} km/h</div>
                        <div class="weather-desc">Precip: {day_forecast['precipitation']}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Historical performance section
            st.subheader("Historical Performance by Team")

            # Generate synthetic team performance data for this track
            teams = ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren", "Aston Martin", "Alpine", "Williams", "RB", "Sauber", "Haas F1 Team"]

            # Create random win percentages that sum to 100%
            win_percentages = np.random.dirichlet(np.ones(len(teams)) * 0.5) * 100

            # Bias towards top teams
            win_percentages[0] *= 3  # Red Bull
            win_percentages[1] *= 2  # Ferrari
            win_percentages[2] *= 2  # Mercedes

            # Normalize to 100%
            win_percentages = win_percentages / win_percentages.sum() * 100

            # Create dataframe
            team_performance = pd.DataFrame({
                'Team': teams,
                'Win Percentage': win_percentages
            })

            # Sort by win percentage
            team_performance = team_performance.sort_values('Win Percentage', ascending=False)

            # Create pie chart
            fig = px.pie(
                team_performance,
                values='Win Percentage',
                names='Team',
                title=f"Historical Win Percentage at {selected_circuit}",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

            # Add track-specific strategy insights
            st.subheader("Strategy Insights")

            # Generate random but realistic strategy insights
            pit_stop_range = random.choice(["1-2", "2-3", "1-3"])
            optimal_tire_strategy = random.choice([
                "Medium-Hard", "Soft-Hard", "Soft-Medium-Soft", 
                "Medium-Hard-Soft", "Soft-Medium-Medium"
            ])

            undercut_effectiveness = random.choice(["Low", "Medium", "High", "Very High"])

            # Determine CSS class for undercut effectiveness
            undercut_class = ""
            if undercut_effectiveness == "Low":
                undercut_class = "difficulty-low"
            elif undercut_effectiveness == "Medium":
                undercut_class = "difficulty-medium"
            else:
                undercut_class = "difficulty-high"

            st.markdown(f"""
            <div class="card">
                <p><strong>Expected Pit Stops:</strong> {pit_stop_range}</p>
                <p><strong>Optimal Tire Strategy:</strong> {optimal_tire_strategy}</p>
                <p><strong>Undercut Effectiveness:</strong> <span class="{undercut_class}">{undercut_effectiveness}</span></p>
                <p><strong>Start Importance:</strong> {'High' if track_info['overtaking_difficulty'] in ['High', 'Very High'] else 'Medium'}</p>
                <p><strong>Qualifying Importance:</strong> {'High' if track_info['overtaking_difficulty'] in ['High', 'Very High'] else 'Medium'}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning(f"Detailed information for {selected_circuit} is not available.")
    else:
        st.error("No upcoming races data available.")

if __name__ == "__main__":
    main()
