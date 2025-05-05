import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st
from datetime import datetime

@st.cache_data(ttl=3600)
def load_upcoming_races():
    """
    Load upcoming races data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing upcoming races data
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data", "processed", "upcoming_races.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading upcoming races: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_predictions():
    """
    Load predictions data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing predictions data
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "results", "2025_predictions_full.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_model():
    """
    Load model and scaler from pickle files.

    Returns:
        tuple: (model, scaler) tuple containing the loaded model and scaler
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        model_path = os.path.join(base_dir, "models", "stacking_model.pkl")
        scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data(ttl=3600)
def load_driver_standings():
    """
    Load driver standings data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing driver standings data
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data", "processed", "driver_standings.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading driver standings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_constructor_standings():
    """
    Load constructor standings data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing constructor standings data
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data", "processed", "constructor_standings.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading constructor standings: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_circuits():
    """
    Load circuits data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing circuits data
    """
    try:
        from pathlib import Path
        base_dir = Path(__file__).parent.parent
        file_path = os.path.join(base_dir, "data", "processed", "circuits.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading circuits data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_results():
    """
    Load race results data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing race results data
    """
    try:
        return pd.read_csv("f1-predictor-2025-main/data/processed/results.csv")
    except Exception as e:
        st.error(f"Error loading results data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_qualifying():
    """
    Load qualifying data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing qualifying data
    """
    try:
        return pd.read_csv("f1-predictor-2025-main/data/processed/qualifying.csv")
    except Exception as e:
        st.error(f"Error loading qualifying data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_drivers():
    """
    Load drivers data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing drivers data
    """
    try:
        return pd.read_csv("f1-predictor-2025-main/data/processed/drivers.csv")
    except Exception as e:
        st.error(f"Error loading drivers data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_constructors():
    """
    Load constructors data from CSV file.

    Returns:
        pd.DataFrame: DataFrame containing constructors data
    """
    try:
        return pd.read_csv("f1-predictor-2025-main/data/processed/constructors.csv")
    except Exception as e:
        st.error(f"Error loading constructors data: {e}")
        return pd.DataFrame()

def get_unique_races(df):
    """
    Get unique race names from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing race_name column

    Returns:
        list: List of unique race names
    """
    if df.empty or 'race_name' not in df.columns:
        return []
    return sorted(df['race_name'].unique())

def get_unique_drivers(df):
    """
    Get unique drivers from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing driver column

    Returns:
        list: List of unique drivers
    """
    if df.empty or 'driver' not in df.columns:
        return []
    return sorted(df['driver'].unique())

def get_unique_teams(df):
    """
    Get unique teams from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing team column

    Returns:
        list: List of unique teams
    """
    if df.empty or 'team' not in df.columns:
        return []
    return sorted(df['team'].unique())

def get_driver_team_mapping(df):
    """
    Create a mapping of drivers to teams.

    Args:
        df (pd.DataFrame): DataFrame containing driver and team columns

    Returns:
        dict: Dictionary mapping drivers to teams
    """
    if df.empty or 'driver' not in df.columns or 'team' not in df.columns:
        return {}

    mapping = {}
    for _, row in df.iterrows():
        mapping[row['driver']] = row['team']

    return mapping

def get_race_date(df, race_name):
    """
    Get the date of a specific race.

    Args:
        df (pd.DataFrame): DataFrame containing race_name and date columns
        race_name (str): Name of the race

    Returns:
        str: Date of the race in string format
    """
    if df.empty or 'race_name' not in df.columns or 'date' not in df.columns:
        return None

    race_data = df[df['race_name'] == race_name]
    if race_data.empty:
        return None

    return race_data['date'].iloc[0]

def get_top_drivers_for_race(df, race_name, n=3):
    """
    Get the top N drivers for a specific race based on probability.

    Args:
        df (pd.DataFrame): DataFrame containing race_name, driver, and probability columns
        race_name (str): Name of the race
        n (int): Number of top drivers to return

    Returns:
        pd.DataFrame: DataFrame containing the top N drivers for the race
    """
    if df.empty or 'race_name' not in df.columns or 'driver' not in df.columns or 'probability' not in df.columns:
        return pd.DataFrame()

    race_data = df[df['race_name'] == race_name]
    if race_data.empty:
        return pd.DataFrame()

    return race_data.nlargest(n, 'probability')
