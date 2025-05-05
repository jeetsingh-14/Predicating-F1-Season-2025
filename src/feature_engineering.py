# src/feature_engineering.py

import pandas as pd
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def parallel_feature_engineering(df, feature_funcs):
    """Run feature engineering functions in parallel"""
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(lambda func: func(df), feature_funcs))
    return pd.concat(results, axis=1)

def feature_engineering_historical():
    print("🚀 Feature engineering: historical data...")
    df = pd.read_csv('data/processed/historical_clean.csv', low_memory=False)

    # Basic feature engineering
    df['raceYear'] = pd.to_datetime(df['date'], errors='coerce').dt.year
    df['grid_position'] = pd.to_numeric(df['grid'], errors='coerce').fillna(10)
    df['finish_position'] = pd.to_numeric(df['position'], errors='coerce').fillna(10)
    df['position_change'] = df['grid_position'] - df['finish_position']
    df['podium'] = df['finish_position'].apply(lambda x: 1 if x in [1, 2, 3] else 0)

    # Circuit difficulty mapping
    circuit_difficulty = {
        'monaco': 5, 'spa-francorchamps': 4, 'silverstone': 4, 'monza': 3, 'suzuka': 4,
        'bahrain': 3, 'jeddah': 4, 'miami': 4, 'imola': 4, 'catalunya': 3, 'montreal': 3,
        'red_bull_ring': 3, 'hungaroring': 3, 'zandvoort': 4, 'baku': 4, 'marina_bay': 5,
        'cota': 3, 'rodriguez': 3, 'interlagos': 4, 'las_vegas': 4, 'losail': 4, 'yas_marina': 3
    }
    df['circuit_difficulty'] = df['circuitRef'].str.lower().map(circuit_difficulty).fillna(3)

    # Encoding
    df['driver_code'] = df['driverRef'].astype('category').cat.codes
    df['constructor_code'] = df['constructorRef'].astype('category').cat.codes

    # Parallel feature engineering functions
    def calculate_driver_form(df):
        df = df.sort_values(['driverRef', 'date'])
        return pd.DataFrame({
            'recent_driver_form': df.groupby('driverRef')['finish_position']
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .fillna(df['finish_position'].mean())
        })

    def calculate_constructor_stats(df):
        constructor_wins = df[df['finish_position'] == 1].groupby('constructorRef').size()
        constructor_total = df.groupby('constructorRef').size()
        constructor_win_rate = (constructor_wins / constructor_total).fillna(0)
        return pd.DataFrame({
            'constructor_win_rate': df['constructorRef'].map(constructor_win_rate),
            'constructor_form': df.groupby('constructorRef')['finish_position']
            .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            .fillna(df['finish_position'].mean())
        })

    def calculate_experience_metrics(df):
        return pd.DataFrame({
            'circuit_familiarity': df.groupby(['driverRef', 'circuitRef']).cumcount() + 1,
            'driver_experience': df.groupby('driverRef').cumcount() + 1,
            'constructor_experience': df.groupby('constructorRef').cumcount() + 1
        })

    feature_funcs = [calculate_driver_form, calculate_constructor_stats, calculate_experience_metrics]
    additional_features = parallel_feature_engineering(df, feature_funcs)
    df = pd.concat([df, additional_features], axis=1)

    # Grid position importance
    qualifying_tracks = ['monaco', 'hungaroring', 'red_bull_ring', 'silverstone']
    df['grid_importance'] = df['circuitRef'].str.lower().isin(qualifying_tracks).astype(float) * 0.5 + 1.0

    df['weather_impact'] = 1.0  # Placeholder

    # Save
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/historical_engineered.csv', index=False)
    print("✅ Historical data engineered and saved!")

    return df

def feature_engineering_upcoming(historical_df):
    print("🚀 Feature engineering: upcoming races...")
    df = pd.read_csv('data/processed/upcoming_races.csv')

    # ✅ SAFETY CHECK: Add missing critical columns
    if 'driver' not in df.columns:
        print("⚠️ 'driver' column missing. Filling with placeholder.")
        df['driver'] = 'Placeholder Driver'

    if 'team' not in df.columns:
        print("⚠️ 'team' column missing. Filling with placeholder.")
        df['team'] = 'Placeholder Team'

    if 'grid_position' not in df.columns:
        print("⚠️ 'grid_position' column missing. Filling with default value 10.")
        df['grid_position'] = 10

    # Encoding
    df['driver_code'] = df['driver'].astype('category').cat.codes
    df['constructor_code'] = df['team'].astype('category').cat.codes
    df['position_change'] = 0
    df['podium'] = 0

    # Circuit mapping
    race_to_circuit = {
        'Bahrain Grand Prix': 'bahrain',
        'Saudi Arabian Grand Prix': 'jeddah',
        'Miami Grand Prix': 'miami',
        'Emilia Romagna Grand Prix': 'imola',
        'Monaco Grand Prix': 'monaco',
        'Spanish Grand Prix': 'catalunya',
        'Canadian Grand Prix': 'montreal',
        'Austrian Grand Prix': 'red_bull_ring',
        'British Grand Prix': 'silverstone',
        'Belgian Grand Prix': 'spa-francorchamps',
        'Hungarian Grand Prix': 'hungaroring',
        'Dutch Grand Prix': 'zandvoort',
        'Italian Grand Prix': 'monza',
        'Azerbaijan Grand Prix': 'baku',
        'Singapore Grand Prix': 'marina_bay',
        'United States Grand Prix': 'cota',
        'Mexico City Grand Prix': 'rodriguez',
        'São Paulo Grand Prix': 'interlagos',
        'Las Vegas Grand Prix': 'las_vegas',
        'Qatar Grand Prix': 'losail',
        'Abu Dhabi Grand Prix': 'yas_marina'
    }
    df['circuitRef'] = df['race_name'].map(race_to_circuit).fillna('generic')

    # Driver & Constructor Mapping
    driver_mapping = {f"{parts[0][0].upper()}. {parts[-1].capitalize()}": driver_ref for driver_ref in historical_df['driverRef'].unique() for parts in [driver_ref.split('_')] if len(parts) >= 2}
    df['driverRef'] = df['driver'].map(driver_mapping).fillna('unknown')

    constructor_mapping = {
        'Red Bull Racing': 'red_bull', 'Ferrari': 'ferrari', 'Mercedes': 'mercedes',
        'McLaren': 'mclaren', 'Aston Martin': 'aston_martin', 'Alpine': 'alpine',
        'Williams': 'williams', 'RB': 'rb', 'Sauber': 'sauber', 'Haas F1 Team': 'haas'
    }
    df['constructorRef'] = df['team'].map(constructor_mapping).fillna('unknown')

    # Parallel feature engineering
    def calculate_recent_form(df):
        recent_races = historical_df[historical_df['raceYear'] >= historical_df['raceYear'].max() - 1]
        driver_form = recent_races.groupby('driverRef')['finish_position'].mean()
        return pd.DataFrame({
            'recent_driver_form': df['driverRef'].map(driver_form).fillna(recent_races['finish_position'].mean())
        })

    def calculate_constructor_stats(df):
        constructor_wins = historical_df[historical_df['finish_position'] == 1].groupby('constructorRef').size()
        constructor_total = historical_df.groupby('constructorRef').size()
        constructor_win_rate = (constructor_wins / constructor_total).fillna(0)
        return pd.DataFrame({
            'constructor_win_rate': df['constructorRef'].map(constructor_win_rate).fillna(0)
        })

    def calculate_experience_metrics(df):
        circuit_familiarity = historical_df.groupby(['driverRef', 'circuitRef']).size()
        driver_experience = historical_df.groupby('driverRef').size()
        constructor_experience = historical_df.groupby('constructorRef').size()
        return pd.DataFrame({
            'circuit_familiarity': df.set_index(['driverRef', 'circuitRef']).index.map(circuit_familiarity).fillna(1),
            'driver_experience': df['driverRef'].map(driver_experience).fillna(1),
            'constructor_experience': df['constructorRef'].map(constructor_experience).fillna(1)
        })

    feature_funcs = [calculate_recent_form, calculate_constructor_stats, calculate_experience_metrics]
    additional_features = parallel_feature_engineering(df, feature_funcs)
    df = pd.concat([df, additional_features], axis=1)

    qualifying_tracks = ['monaco', 'hungaroring', 'red_bull_ring', 'silverstone']
    df['grid_importance'] = df['circuitRef'].str.lower().isin(qualifying_tracks).astype(float) * 0.5 + 1.0
    df['weather_impact'] = 1.0

    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/upcoming_engineered.csv', index=False)
    print("✅ Upcoming races data engineered and saved!")

def main():
    historical_df = feature_engineering_historical()
    feature_engineering_upcoming(historical_df)

if __name__ == "__main__":
    main()
