import pandas as pd

# Load CSV files
constructors = pd.read_csv("constructors.csv")
constructor_standings = pd.read_csv("constructor_standings.csv")
races = pd.read_csv("races.csv")
circuits = pd.read_csv("circuits.csv")

# Merge data
merged = constructor_standings.merge(races, on='raceId', how='left')
merged = merged.merge(constructors, on='constructorId', how='left')
merged = merged.merge(circuits, on='circuitId', how='left')

# Confirm column names
print("COLUMNS IN MERGED:\n", merged.columns.tolist())

# Final selection (use 'year' instead of 'season')
constructor_history = merged[[
    'year', 'round', 'name_x', 'name_y', 'location', 'nationality',
    'points', 'wins', 'position'
]]

# Rename for output
constructor_history.columns = [
    'Season', 'Round', 'RaceName', 'Constructor', 'Country', 'Nationality',
    'Points', 'Wins', 'Position'
]

# Save
constructor_history.to_csv("all_constructor_history.csv", index=False)
