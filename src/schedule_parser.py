import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1ScheduleParser:
    def __init__(self):
        self.base_url = "https://ergast.com/api/f1"
        self.calendar_url = f"{self.base_url}/2025.html"
        self.drivers = [
            'M. Verstappen', 'L. Hamilton', 'C. Leclerc', 'L. Norris', 'O. Piastri', 
            'G. Russell', 'S. Perez', 'F. Alonso', 'L. Stroll', 'Y. Tsunoda', 
            'D. Ricciardo', 'E. Ocon', 'P. Gasly', 'A. Albon', 'L. Sargeant',
            'V. Bottas', 'Z. Guanyu', 'N. Hülkenberg', 'K. Magnussen'
        ]
        self.teams = [
            'Red Bull Racing', 'Mercedes', 'Ferrari', 'McLaren', 'Aston Martin',
            'RB', 'Alpine', 'Williams', 'Sauber', 'Haas F1 Team'
        ]
        
    def fetch_calendar(self):
        """Fetch F1 calendar for the 2025 season."""
        calendar_data = [
            {"round": 1, "name": "Chinese Grand Prix", "circuit": "Shanghai International Circuit", "date": "2025-04-11"},
            {"round": 2, "name": "Miami Grand Prix", "circuit": "Miami International Autodrome", "date": "2025-05-04"},
            {"round": 3, "name": "Emilia Romagna Grand Prix", "circuit": "Autodromo Enzo e Dino Ferrari", "date": "2025-05-18"},
            {"round": 4, "name": "Monaco Grand Prix", "circuit": "Circuit de Monaco", "date": "2025-05-25"},
            {"round": 5, "name": "Spanish Grand Prix", "circuit": "Circuit de Barcelona-Catalunya", "date": "2025-06-01"},
            {"round": 6, "name": "Canadian Grand Prix", "circuit": "Circuit Gilles Villeneuve", "date": "2025-06-15"},
            {"round": 7, "name": "Austrian Grand Prix", "circuit": "Red Bull Ring", "date": "2025-06-29"},
            {"round": 8, "name": "British Grand Prix", "circuit": "Silverstone Circuit", "date": "2025-07-06"},
            {"round": 9, "name": "Hungarian Grand Prix", "circuit": "Hungaroring", "date": "2025-07-27"},
            {"round": 10, "name": "Belgian Grand Prix", "circuit": "Circuit de Spa-Francorchamps", "date": "2025-08-03"},
            {"round": 11, "name": "Dutch Grand Prix", "circuit": "Circuit Zandvoort", "date": "2025-08-31"},
            {"round": 12, "name": "Italian Grand Prix", "circuit": "Autodromo Nazionale Monza", "date": "2025-09-07"},
            {"round": 13, "name": "Azerbaijan Grand Prix", "circuit": "Baku City Circuit", "date": "2025-09-21"},
            {"round": 14, "name": "Singapore Grand Prix", "circuit": "Marina Bay Street Circuit", "date": "2025-09-28"},
            {"round": 15, "name": "United States Grand Prix", "circuit": "Circuit of the Americas", "date": "2025-10-19"},
            {"round": 16, "name": "Mexico City Grand Prix", "circuit": "Autódromo Hermanos Rodríguez", "date": "2025-10-26"},
            {"round": 17, "name": "São Paulo Grand Prix", "circuit": "Autódromo José Carlos Pace", "date": "2025-11-09"},
            {"round": 18, "name": "Las Vegas Grand Prix", "circuit": "Las Vegas Strip Circuit", "date": "2025-11-22"},
            {"round": 19, "name": "Qatar Grand Prix", "circuit": "Lusail International Circuit", "date": "2025-11-30"},
            {"round": 20, "name": "Abu Dhabi Grand Prix", "circuit": "Yas Marina Circuit", "date": "2025-12-07"}
        ]
        
        # Add circuit difficulty ratings (1-10 scale)
        circuit_difficulties = {
            "Shanghai International Circuit": 7,
            "Miami International Autodrome": 6,
            "Autodromo Enzo e Dino Ferrari": 8,
            "Circuit de Monaco": 10,
            "Circuit de Barcelona-Catalunya": 7,
            "Circuit Gilles Villeneuve": 8,
            "Red Bull Ring": 6,
            "Silverstone Circuit": 8,
            "Hungaroring": 7,
            "Circuit de Spa-Francorchamps": 9,
            "Circuit Zandvoort": 7,
            "Autodromo Nazionale Monza": 6,
            "Baku City Circuit": 8,
            "Marina Bay Street Circuit": 9,
            "Circuit of the Americas": 7,
            "Autódromo Hermanos Rodríguez": 7,
            "Autódromo José Carlos Pace": 8,
            "Las Vegas Strip Circuit": 7,
            "Lusail International Circuit": 7,
            "Yas Marina Circuit": 6
        }
        
        # Add difficulty ratings to calendar data
        for race in calendar_data:
            race["circuit_difficulty"] = circuit_difficulties[race["circuit"]]
            
        return pd.DataFrame(calendar_data)
    
    def fetch_calendar(self):
        """Fetch the F1 calendar from Formula1.com"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.calendar_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # For now, we'll use a fallback approach since the actual scraping would require
            # more complex parsing of the F1 website structure
            logger.info("Using fallback calendar data (actual scraping would require more complex parsing)")
            
            # Create a fallback calendar with the known 2025 races
            calendar_data = [
                {'race_name': 'Bahrain Grand Prix', 'date': '2025-04-13', 'round': 4, 'country': 'Bahrain', 'circuit_difficulty': 3},
                {'race_name': 'Saudi Arabian Grand Prix', 'date': '2025-04-20', 'round': 5, 'country': 'Saudi Arabia', 'circuit_difficulty': 4},
                {'race_name': 'Miami Grand Prix', 'date': '2025-05-04', 'round': 6, 'country': 'USA', 'circuit_difficulty': 4},
                {'race_name': 'Emilia Romagna Grand Prix', 'date': '2025-05-18', 'round': 7, 'country': 'Italy', 'circuit_difficulty': 4},
                {'race_name': 'Monaco Grand Prix', 'date': '2025-05-25', 'round': 8, 'country': 'Monaco', 'circuit_difficulty': 5},
                {'race_name': 'Spanish Grand Prix', 'date': '2025-06-01', 'round': 9, 'country': 'Spain', 'circuit_difficulty': 3},
                {'race_name': 'Canadian Grand Prix', 'date': '2025-06-15', 'round': 10, 'country': 'Canada', 'circuit_difficulty': 3},
                {'race_name': 'Austrian Grand Prix', 'date': '2025-06-29', 'round': 11, 'country': 'Austria', 'circuit_difficulty': 3},
                {'race_name': 'British Grand Prix', 'date': '2025-07-06', 'round': 12, 'country': 'United Kingdom', 'circuit_difficulty': 4},
                {'race_name': 'Belgian Grand Prix', 'date': '2025-07-27', 'round': 13, 'country': 'Belgium', 'circuit_difficulty': 4},
                {'race_name': 'Hungarian Grand Prix', 'date': '2025-08-03', 'round': 14, 'country': 'Hungary', 'circuit_difficulty': 3},
                {'race_name': 'Dutch Grand Prix', 'date': '2025-08-31', 'round': 15, 'country': 'Netherlands', 'circuit_difficulty': 4},
                {'race_name': 'Italian Grand Prix', 'date': '2025-09-07', 'round': 16, 'country': 'Italy', 'circuit_difficulty': 3},
                {'race_name': 'Azerbaijan Grand Prix', 'date': '2025-09-21', 'round': 17, 'country': 'Azerbaijan', 'circuit_difficulty': 4},
                {'race_name': 'Singapore Grand Prix', 'date': '2025-10-05', 'round': 18, 'country': 'Singapore', 'circuit_difficulty': 5},
                {'race_name': 'United States Grand Prix', 'date': '2025-10-19', 'round': 19, 'country': 'USA', 'circuit_difficulty': 3},
                {'race_name': 'Mexico City Grand Prix', 'date': '2025-10-26', 'round': 20, 'country': 'Mexico', 'circuit_difficulty': 3},
                {'race_name': 'São Paulo Grand Prix', 'date': '2025-11-09', 'round': 21, 'country': 'Brazil', 'circuit_difficulty': 4},
                {'race_name': 'Las Vegas Grand Prix', 'date': '2025-11-22', 'round': 22, 'country': 'USA', 'circuit_difficulty': 4},
                {'race_name': 'Qatar Grand Prix', 'date': '2025-11-30', 'round': 23, 'country': 'Qatar', 'circuit_difficulty': 4},
                {'race_name': 'Abu Dhabi Grand Prix', 'date': '2025-12-07', 'round': 24, 'country': 'Abu Dhabi', 'circuit_difficulty': 3}
            ]
            
            return pd.DataFrame(calendar_data)
            
        except Exception as e:
            logger.error(f"Error fetching calendar: {str(e)}")
            return None
    
    def generate_upcoming_races(self, calendar_df):
        """Generate upcoming races data with all drivers and teams"""
        if calendar_df is None or calendar_df.empty:
            logger.error("No calendar data available")
            return None
        
        # Create a list to store all race entries
        all_entries = []
        
        # For each race, create entries for all drivers
        for _, race in calendar_df.iterrows():
            race_name = race['race_name']
            date = race['date']
            round_num = race['round']
            country = race['country']
            circuit_difficulty = race['circuit_difficulty']
            
            # Create entries for each driver
            for driver in self.drivers:
                # Find the team for this driver
                team = self._get_team_for_driver(driver)
                
                # Add the entry
                all_entries.append({
                    'race_name': race_name,
                    'driver': driver,
                    'team': team,
                    'grid_position': 10,  # Default position, will be updated later
                    'circuit_difficulty': circuit_difficulty,
                    'date': date,
                    'round': round_num,
                    'country': country
                })
        
        # Convert to DataFrame
        upcoming_races_df = pd.DataFrame(all_entries)
        
        return upcoming_races_df
    
    def _get_team_for_driver(self, driver):
        """Get the team for a given driver"""
        # This mapping would ideally come from a database or API
        # For now, we'll use a simple mapping
        driver_team_mapping = {
            'M. Verstappen': 'Red Bull Racing',
            'S. Perez': 'Red Bull Racing',
            'L. Hamilton': 'Ferrari',
            'C. Leclerc': 'Ferrari',
            'G. Russell': 'Mercedes',
            'L. Norris': 'McLaren',
            'O. Piastri': 'McLaren',
            'F. Alonso': 'Aston Martin',
            'L. Stroll': 'Aston Martin',
            'Y. Tsunoda': 'RB',
            'D. Ricciardo': 'RB',
            'E. Ocon': 'Alpine',
            'P. Gasly': 'Alpine',
            'A. Albon': 'Williams',
            'L. Sargeant': 'Williams',
            'V. Bottas': 'Sauber',
            'Z. Guanyu': 'Sauber',
            'N. Hülkenberg': 'Haas F1 Team',
            'K. Magnussen': 'Haas F1 Team'
        }
        
        return driver_team_mapping.get(driver, 'Unknown')
    
    def save_upcoming_races(self, df, filename='data/processed/upcoming_races.csv'):
        """Save upcoming races data to CSV"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df.to_csv(filename, index=False)
        logger.info(f"Upcoming races saved to {filename}")

def fetch_2025_schedule():
    """Fetch the 2025 F1 race schedule"""
    try:
        # Sample schedule data for 2025 (since actual data might not be available yet)
        schedule_2025 = [
            {"race_name": "Bahrain Grand Prix", "date": "2025-04-13", "circuit": "Bahrain International Circuit"},
            {"race_name": "Saudi Arabian Grand Prix", "date": "2025-04-20", "circuit": "Jeddah Corniche Circuit"},
            {"race_name": "Miami Grand Prix", "date": "2025-05-04", "circuit": "Miami International Autodrome"},
            {"race_name": "Emilia Romagna Grand Prix", "date": "2025-05-18", "circuit": "Autodromo Enzo e Dino Ferrari"},
            {"race_name": "Monaco Grand Prix", "date": "2025-05-25", "circuit": "Circuit de Monaco"},
            {"race_name": "Spanish Grand Prix", "date": "2025-06-01", "circuit": "Circuit de Barcelona-Catalunya"},
            {"race_name": "Canadian Grand Prix", "date": "2025-06-15", "circuit": "Circuit Gilles Villeneuve"},
            {"race_name": "Austrian Grand Prix", "date": "2025-06-29", "circuit": "Red Bull Ring"},
            {"race_name": "British Grand Prix", "date": "2025-07-06", "circuit": "Silverstone Circuit"},
            {"race_name": "Belgian Grand Prix", "date": "2025-07-27", "circuit": "Circuit de Spa-Francorchamps"},
            {"race_name": "Hungarian Grand Prix", "date": "2025-08-03", "circuit": "Hungaroring"},
            {"race_name": "Dutch Grand Prix", "date": "2025-08-31", "circuit": "Circuit Zandvoort"},
            {"race_name": "Italian Grand Prix", "date": "2025-09-07", "circuit": "Autodromo Nazionale Monza"},
            {"race_name": "Azerbaijan Grand Prix", "date": "2025-09-21", "circuit": "Baku City Circuit"},
            {"race_name": "Singapore Grand Prix", "date": "2025-10-05", "circuit": "Marina Bay Street Circuit"},
            {"race_name": "United States Grand Prix", "date": "2025-10-19", "circuit": "Circuit of the Americas"},
            {"race_name": "Mexico City Grand Prix", "date": "2025-10-26", "circuit": "Autódromo Hermanos Rodríguez"},
            {"race_name": "São Paulo Grand Prix", "date": "2025-11-09", "circuit": "Autódromo José Carlos Pace"},
            {"race_name": "Las Vegas Grand Prix", "date": "2025-11-22", "circuit": "Las Vegas Street Circuit"},
            {"race_name": "Qatar Grand Prix", "date": "2025-11-30", "circuit": "Lusail International Circuit"},
            {"race_name": "Abu Dhabi Grand Prix", "date": "2025-12-07", "circuit": "Yas Marina Circuit"}
        ]
        
        # Convert to DataFrame
        df = pd.DataFrame(schedule_2025)
        
        # Add current drivers and teams
        drivers = [
            "M. Verstappen", "S. Perez",           # Red Bull
            "C. Leclerc", "C. Sainz",             # Ferrari
            "L. Hamilton", "G. Russell",           # Mercedes
            "L. Norris", "O. Piastri",            # McLaren
            "F. Alonso", "L. Stroll",             # Aston Martin
            "E. Ocon", "P. Gasly",                # Alpine
            "A. Albon", "L. Sargeant",            # Williams
            "D. Ricciardo", "Y. Tsunoda",         # RB
            "V. Bottas", "Z. Guanyu",             # Sauber
            "K. Magnussen", "N. Hulkenberg"        # Haas
        ]
        
        teams = [
            "Red Bull Racing", "Red Bull Racing",
            "Ferrari", "Ferrari",
            "Mercedes", "Mercedes",
            "McLaren", "McLaren",
            "Aston Martin", "Aston Martin",
            "Alpine", "Alpine",
            "Williams", "Williams",
            "RB", "RB",
            "Sauber", "Sauber",
            "Haas F1 Team", "Haas F1 Team"
        ]
        
        # Create a list to store all race entries
        all_entries = []
        
        # For each race, add entries for all drivers
        for _, race in df.iterrows():
            for driver, team in zip(drivers, teams):
                entry = {
                    'race_name': race['race_name'],
                    'date': race['date'],
                    'circuit': race['circuit'],
                    'driver': driver,
                    'team': team,
                    'grid_position': 0  # Will be updated later
                }
                all_entries.append(entry)
        
        # Convert to DataFrame
        final_df = pd.DataFrame(all_entries)
        
        # Save to CSV
        os.makedirs('data/processed', exist_ok=True)
        final_df.to_csv('data/processed/upcoming_races.csv', index=False)
        print("✅ Race schedule saved successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fetching race schedule: {str(e)}")
        return False

def main():
    fetch_2025_schedule()

if __name__ == "__main__":
    main() 