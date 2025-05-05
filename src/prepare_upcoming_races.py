# src/prepare_upcoming_races.py

import pandas as pd
import os
from schedule_parser import F1ScheduleParser

def create_upcoming_races():
    """Create upcoming races dataset with driver entries."""
    
    # 2025 Driver-Team mapping
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Oliver Bearman': 'Ferrari',
        'Lewis Hamilton': 'Ferrari',
        'George Russell': 'Mercedes',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine',
        'Alexander Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Daniel Ricciardo': 'RB',
        'Yuki Tsunoda': 'RB',
        'Valtteri Bottas': 'Sauber',
        'Zhou Guanyu': 'Sauber',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team'
    }
    
    # Get race calendar
    parser = F1ScheduleParser()
    calendar = parser.fetch_calendar()
    
    # If calendar fetch failed, use the fallback data
    if calendar is None:
        calendar = pd.DataFrame([
            {'round': 1, 'name': 'Qatar Grand Prix', 'circuit': 'Lusail International Circuit', 'date': '2025-04-11', 'circuit_difficulty': 3},
            {'round': 2, 'name': 'Chinese Grand Prix', 'circuit': 'Shanghai International Circuit', 'date': '2025-04-19', 'circuit_difficulty': 4},
            {'round': 3, 'name': 'Japanese Grand Prix', 'circuit': 'Suzuka Circuit', 'date': '2025-04-27', 'circuit_difficulty': 5},
            {'round': 4, 'name': 'Bahrain Grand Prix', 'circuit': 'Bahrain International Circuit', 'date': '2025-05-04', 'circuit_difficulty': 3},
            {'round': 5, 'name': 'Saudi Arabian Grand Prix', 'circuit': 'Jeddah Corniche Circuit', 'date': '2025-05-18', 'circuit_difficulty': 4},
            {'round': 6, 'name': 'Miami Grand Prix', 'circuit': 'Miami International Autodrome', 'date': '2025-05-25', 'circuit_difficulty': 3},
            {'round': 7, 'name': 'Emilia Romagna Grand Prix', 'circuit': 'Autodromo Enzo e Dino Ferrari', 'date': '2025-06-01', 'circuit_difficulty': 4},
            {'round': 8, 'name': 'Monaco Grand Prix', 'circuit': 'Circuit de Monaco', 'date': '2025-06-15', 'circuit_difficulty': 5},
            {'round': 9, 'name': 'Spanish Grand Prix', 'circuit': 'Circuit de Barcelona-Catalunya', 'date': '2025-06-29', 'circuit_difficulty': 3},
            {'round': 10, 'name': 'Canadian Grand Prix', 'circuit': 'Circuit Gilles Villeneuve', 'date': '2025-07-06', 'circuit_difficulty': 3},
            {'round': 11, 'name': 'Austrian Grand Prix', 'circuit': 'Red Bull Ring', 'date': '2025-07-20', 'circuit_difficulty': 3},
            {'round': 12, 'name': 'British Grand Prix', 'circuit': 'Silverstone Circuit', 'date': '2025-07-27', 'circuit_difficulty': 4},
            {'round': 13, 'name': 'Hungarian Grand Prix', 'circuit': 'Hungaroring', 'date': '2025-08-03', 'circuit_difficulty': 4},
            {'round': 14, 'name': 'Belgian Grand Prix', 'circuit': 'Circuit de Spa-Francorchamps', 'date': '2025-08-24', 'circuit_difficulty': 5},
            {'round': 15, 'name': 'Dutch Grand Prix', 'circuit': 'Circuit Zandvoort', 'date': '2025-08-31', 'circuit_difficulty': 4},
            {'round': 16, 'name': 'Italian Grand Prix', 'circuit': 'Monza Circuit', 'date': '2025-09-07', 'circuit_difficulty': 3},
            {'round': 17, 'name': 'Azerbaijan Grand Prix', 'circuit': 'Baku City Circuit', 'date': '2025-09-21', 'circuit_difficulty': 4},
            {'round': 18, 'name': 'Singapore Grand Prix', 'circuit': 'Marina Bay Street Circuit', 'date': '2025-10-05', 'circuit_difficulty': 5},
            {'round': 19, 'name': 'United States Grand Prix', 'circuit': 'Circuit of the Americas', 'date': '2025-10-19', 'circuit_difficulty': 4},
            {'round': 20, 'name': 'Mexico City Grand Prix', 'circuit': 'Autódromo Hermanos Rodríguez', 'date': '2025-10-26', 'circuit_difficulty': 3},
            {'round': 21, 'name': 'São Paulo Grand Prix', 'circuit': 'Interlagos Circuit', 'date': '2025-11-09', 'circuit_difficulty': 4},
            {'round': 22, 'name': 'Las Vegas Grand Prix', 'circuit': 'Las Vegas Strip Circuit', 'date': '2025-11-23', 'circuit_difficulty': 3},
            {'round': 23, 'name': 'Qatar Grand Prix', 'circuit': 'Lusail International Circuit', 'date': '2025-11-30', 'circuit_difficulty': 3},
            {'round': 24, 'name': 'Saudi Arabian Grand Prix', 'circuit': 'Jeddah Corniche Circuit', 'date': '2025-12-07', 'circuit_difficulty': 4}
        ])
    
    # Create entries for each driver for each race
    entries = []
    for _, race in calendar.iterrows():
        for driver, team in driver_teams.items():
            entry = {
                'round': race['round'],
                'race_name': race['name'],
                'circuit': race['circuit'],
                'date': race['date'],
                'circuit_difficulty': race['circuit_difficulty'],
                'driver': driver,
                'team': team,
                'grid_position': 10,  # Default grid position
                'circuit_performance': 0.5,  # Default circuit performance
                'time_weight': 0.5,  # Default time weight
                'team_consistency': 0.5,  # Default team consistency
                'driver_team_synergy': 0.5,  # Default driver-team synergy
                'form_experience': 0.5,  # Default form experience
                'team_circuit': 0.5,  # Default team circuit performance
                'grid_weather': 0.5  # Default grid weather impact
            }
            entries.append(entry)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(entries)
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/upcoming_races.csv', index=False)
    return df

if __name__ == "__main__":
    create_upcoming_races()
