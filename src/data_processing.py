# src/data_processing.py

import pandas as pd
import os

def load_data():
    print("🚦 Loading raw Kaggle data...")
    base_path = 'data/processed'

    circuits = pd.read_csv(f'{base_path}/circuits.csv')
    constructors = pd.read_csv(f'{base_path}/constructors.csv')
    drivers = pd.read_csv(f'{base_path}/drivers.csv')
    races = pd.read_csv(f'{base_path}/races.csv')
    results = pd.read_csv(f'{base_path}/results.csv')

    print("✅ Data loaded.")
    return circuits, constructors, drivers, races, results

def merge_data(circuits, constructors, drivers, races, results):
    print("🔗 Merging data...")

    # Merge results + races
    df = results.merge(races, on='raceId', how='left', suffixes=('', '_race'))

    # Merge drivers
    df = df.merge(drivers, on='driverId', how='left', suffixes=('', '_driver'))

    # Merge constructors
    df = df.merge(constructors, on='constructorId', how='left', suffixes=('', '_constructor'))

    # Merge circuits
    df = df.merge(circuits, on='circuitId', how='left', suffixes=('', '_circuit'))

    print(f"✅ Merged shape: {df.shape}")
    return df

def clean_data(df):
    print("🧹 Cleaning data...")

    # Drop columns not needed for now
    drop_cols = ['number', 'positionText', 'positionOrder', 'time', 'laps', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'statusId']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Clean data types
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    # Fill NaNs if needed (optional, flexible later)
    df.fillna({'position': 99}, inplace=True)

    print("✅ Cleaning complete.")
    return df

def save_data(df):
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/historical_clean.csv', index=False)
    print("💾 Saved: data/processed/historical_clean.csv")

def main():
    circuits, constructors, drivers, races, results = load_data()
    merged = merge_data(circuits, constructors, drivers, races, results)
    cleaned = clean_data(merged)
    save_data(cleaned)
    print("🎉 Data processing complete!")

if __name__ == '__main__':
    main()
