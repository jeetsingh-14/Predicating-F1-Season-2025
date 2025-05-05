import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class F1Visualizer:
    def __init__(self):
        self.output_dir = 'results/visualizations'
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set the style for all visualizations
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Define team colors
        self.team_colors = {
            'Red Bull Racing': '#0600EF',
            'Mercedes': '#00D2BE',
            'Ferrari': '#DC0000',
            'McLaren': '#FF8700',
            'Aston Martin': '#006F62',
            'RB': '#1E3F9C',
            'Alpine': '#0090FF',
            'Williams': '#005AFF',
            'Sauber': '#52E252',
            'Haas F1 Team': '#FFFFFF',
            'Unknown': '#808080'
        }
    
    def load_prediction_data(self, filename='data/processed/predictions.csv'):
        """Load prediction data from CSV"""
        try:
            df = pd.read_csv(filename)
            logger.info(f"Loaded prediction data from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading prediction data: {str(e)}")
            return None
    
    def load_betting_data(self, filename='data/processed/betting_probabilities.csv'):
        """Load betting probability data from CSV"""
        try:
            df = pd.read_csv(filename)
            logger.info(f"Loaded betting data from {filename}")
            return df
        except Exception as e:
            logger.error(f"Error loading betting data: {str(e)}")
            return None
    
    def plot_podium_probabilities(self, df, race_name=None, top_n=10, filename=None):
        """Plot podium probabilities for a specific race or all races"""
        if df is None or df.empty:
            logger.error("No data available for visualization")
            return
        
        # Filter by race if specified
        if race_name:
            race_df = df[df['race_name'] == race_name]
            if race_df.empty:
                logger.error(f"No data found for race: {race_name}")
                return
        else:
            race_df = df
        
        # Sort by probability and get top N drivers
        top_drivers = race_df.sort_values('probability', ascending=False).head(top_n)
        
        # Create the figure
        plt.figure(figsize=(12, 8))
        
        # Create the bar plot
        bars = sns.barplot(
            x='probability', 
            y='driver', 
            data=top_drivers,
            palette=[self.team_colors.get(team, '#808080') for team in top_drivers['team']]
        )
        
        # Add probability labels to the bars
        for i, bar in enumerate(bars.patches):
            bars.text(
                bar.get_width() + 0.01, 
                bar.get_y() + bar.get_height()/2, 
                f'{top_drivers.iloc[i]["probability"]:.1%}', 
                va='center'
            )
        
        # Set the title and labels
        title = f"Top {top_n} Podium Probabilities"
        if race_name:
            title += f" - {race_name}"
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Probability', fontsize=12)
        plt.ylabel('Driver', fontsize=12)
        
        # Set x-axis limits to better display the probabilities
        plt.xlim(0, min(1.0, top_drivers['probability'].max() * 1.2))
        
        # Adjust the layout
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            race_suffix = f"_{race_name.replace(' ', '_')}" if race_name else ""
            filename = f"{self.output_dir}/podium_probabilities{race_suffix}_{timestamp}.png"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved podium probabilities plot to {filename}")
        
        # Close the figure to free memory
        plt.close()
    
    def plot_probability_comparison(self, pred_df, bet_df, race_name, top_n=10, filename=None):
        """Compare model predictions with betting odds for a specific race"""
        if pred_df is None or bet_df is None:
            logger.error("No data available for comparison")
            return
        
        # Filter by race
        pred_race = pred_df[pred_df['race_name'] == race_name]
        bet_race = bet_df[bet_df['race_name'] == race_name]
        
        if pred_race.empty or bet_race.empty:
            logger.error(f"No data found for race: {race_name}")
            return
        
        # Merge the dataframes
        comparison = pd.merge(
            pred_race, 
            bet_race[['driver', 'normalized_probability']], 
            on='driver', 
            how='inner'
        )
        
        # Sort by model probability and get top N drivers
        top_drivers = comparison.sort_values('probability', ascending=False).head(top_n)
        
        # Create the figure
        plt.figure(figsize=(14, 8))
        
        # Set up the bar positions
        x = np.arange(len(top_drivers))
        width = 0.35
        
        # Create the bars
        plt.bar(
            x - width/2, 
            top_drivers['probability'], 
            width, 
            label='Model Prediction',
            color=[self.team_colors.get(team, '#808080') for team in top_drivers['team']]
        )
        plt.bar(
            x + width/2, 
            top_drivers['normalized_probability'], 
            width, 
            label='Betting Odds',
            color=[self.team_colors.get(team, '#808080') for team in top_drivers['team']],
            alpha=0.5
        )
        
        # Add probability labels to the bars
        for i, (pred, bet) in enumerate(zip(top_drivers['probability'], top_drivers['normalized_probability'])):
            plt.text(i - width/2, pred + 0.01, f'{pred:.1%}', ha='center', va='bottom')
            plt.text(i + width/2, bet + 0.01, f'{bet:.1%}', ha='center', va='bottom')
        
        # Set the title and labels
        plt.title(f'Model Predictions vs Betting Odds - {race_name}', fontsize=16, pad=20)
        plt.xlabel('Driver', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        
        # Set x-axis labels
        plt.xticks(x, top_drivers['driver'], rotation=45, ha='right')
        
        # Set y-axis limits to better display the probabilities
        plt.ylim(0, min(1.0, max(top_drivers['probability'].max(), top_drivers['normalized_probability'].max()) * 1.2))
        
        # Add a legend
        plt.legend()
        
        # Adjust the layout
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/probability_comparison_{race_name.replace(' ', '_')}_{timestamp}.png"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved probability comparison plot to {filename}")
        
        # Close the figure to free memory
        plt.close()
    
    def plot_race_calendar(self, df, filename=None):
        """Plot the race calendar with podium probabilities"""
        if df is None or df.empty:
            logger.error("No data available for visualization")
            return
        
        # Group by race and get the top 3 drivers for each race
        top_drivers_by_race = []
        
        for race_name, race_df in df.groupby('race_name'):
            top_3 = race_df.nlargest(3, 'probability')
            top_drivers_by_race.append({
                'race_name': race_name,
                'date': race_df['date'].iloc[0],
                'top_drivers': top_3['driver'].tolist(),
                'probabilities': top_3['probability'].tolist()
            })
        
        # Convert to DataFrame and sort by date
        calendar_df = pd.DataFrame(top_drivers_by_race)
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        calendar_df = calendar_df.sort_values('date')
        
        # Create the figure
        plt.figure(figsize=(16, 10))
        
        # Create a grid of subplots
        n_races = len(calendar_df)
        n_cols = 3
        n_rows = (n_races + n_cols - 1) // n_cols
        
        for i, (_, race) in enumerate(calendar_df.iterrows()):
            # Create a subplot
            plt.subplot(n_rows, n_cols, i+1)
            
            # Create a horizontal bar chart
            drivers = race['top_drivers']
            probs = race['probabilities']
            
            bars = plt.barh(
                drivers, 
                probs, 
                color=[self.team_colors.get(team, '#808080') for team in [df[df['driver'] == d]['team'].iloc[0] for d in drivers]]
            )
            
            # Add probability labels
            for j, bar in enumerate(bars):
                plt.text(
                    bar.get_width() + 0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{probs[j]:.2%}', 
                    va='center'
                )
            
            # Set the title and labels
            plt.title(race['race_name'], fontsize=12)
            plt.xlabel('Probability', fontsize=10)
            
            # Remove y-axis labels for all but the first column
            if i % n_cols != 0:
                plt.ylabel('')
            
            # Set the x-axis limits
            plt.xlim(0, 1)
        
        # Add a main title
        plt.suptitle('F1 2025 Season - Top 3 Podium Probabilities by Race', fontsize=16, y=1.02)
        
        # Adjust the layout
        plt.tight_layout()
        
        # Save the figure
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/race_calendar_{timestamp}.png"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Saved race calendar plot to {filename}")
        
        # Close the figure to free memory
        plt.close()
    
    def create_dashboard(self, pred_df, bet_df=None, output_dir=None):
        """Create a comprehensive dashboard of all visualizations"""
        if pred_df is None or pred_df.empty:
            logger.error("No prediction data available for dashboard")
            return
        
        # Set the output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{self.output_dir}/dashboard_{timestamp}"
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get unique race names
        race_names = pred_df['race_name'].unique()
        
        # Create visualizations for each race
        for race_name in race_names:
            # Plot podium probabilities
            self.plot_podium_probabilities(
                pred_df, 
                race_name=race_name, 
                filename=f"{output_dir}/podium_probabilities_{race_name.replace(' ', '_')}.png"
            )
            
            # Plot probability comparison if betting data is available
            if bet_df is not None and not bet_df.empty:
                self.plot_probability_comparison(
                    pred_df, 
                    bet_df, 
                    race_name=race_name, 
                    filename=f"{output_dir}/probability_comparison_{race_name.replace(' ', '_')}.png"
                )
        
        # Plot the race calendar
        self.plot_race_calendar(
            pred_df, 
            filename=f"{output_dir}/race_calendar.png"
        )
        
        logger.info(f"Dashboard created successfully in {output_dir}")

def create_race_prediction_chart(predictions_df, race_name):
    """Create a bar chart for race predictions"""
    plt.figure(figsize=(12, 8))
    
    # Filter for the specific race
    race_data = predictions_df[predictions_df['race_name'] == race_name].copy()
    
    # Scale probabilities to be between 85% and 100%
    min_prob = 0.85
    max_prob = 1.00
    race_data['scaled_probability'] = min_prob + (max_prob - min_prob) * (race_data['probability'] - race_data['probability'].min()) / (race_data['probability'].max() - race_data['probability'].min())
    
    # Sort by probability
    race_data = race_data.sort_values('scaled_probability', ascending=True)
    
    # Create horizontal bar chart
    bars = plt.barh(race_data['driver'], race_data['scaled_probability'] * 100)
    
    # Customize the chart
    plt.title(f'Podium Predictions: {race_name}', pad=20, fontsize=14)
    plt.xlabel('Probability (%)', fontsize=12)
    plt.ylabel('Driver', fontsize=12)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Customize colors based on probability
    colors = ['#FF9999' if x < 90 else '#FFCC99' if x < 95 else '#99FF99' for x in race_data['scaled_probability'] * 100]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add grid lines
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gcf()

def create_driver_form_chart(historical_df):
    """Create a line chart showing driver form over time"""
    plt.figure(figsize=(15, 8))
    
    # Get the most recent races
    recent_races = historical_df.sort_values('date').tail(10)
    
    # Plot form for each driver
    for driver in recent_races['driver'].unique():
        driver_data = recent_races[recent_races['driver'] == driver]
        plt.plot(driver_data['date'], driver_data['recent_driver_form'], 
                marker='o', label=driver)
    
    plt.title('Driver Form Over Recent Races')
    plt.xlabel('Race Date')
    plt.ylabel('Form (Average Position)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt.gcf()

def create_team_performance_chart(historical_df):
    """Create a box plot showing team performance distribution"""
    plt.figure(figsize=(12, 6))
    
    # Create box plot
    sns.boxplot(x='team', y='position', data=historical_df)
    
    plt.title('Team Performance Distribution')
    plt.xlabel('Team')
    plt.ylabel('Race Position')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt.gcf()

def create_circuit_difficulty_chart(historical_df):
    """Create a heatmap showing circuit difficulty for each driver"""
    plt.figure(figsize=(15, 8))
    
    # Create pivot table
    circuit_perf = pd.pivot_table(
        historical_df,
        values='position',
        index='driver',
        columns='circuit',
        aggfunc='mean'
    )
    
    # Create heatmap
    sns.heatmap(circuit_perf, cmap='RdYlGn_r', center=10)
    
    plt.title('Circuit Performance by Driver')
    plt.xlabel('Circuit')
    plt.ylabel('Driver')
    
    plt.tight_layout()
    return plt.gcf()

def create_prediction_confidence_chart(predictions_df):
    """Create a scatter plot showing prediction confidence"""
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot
    plt.scatter(predictions_df['probability'] * 100, 
               predictions_df['grid_position'],
               alpha=0.6)
    
    plt.title('Prediction Confidence vs Grid Position')
    plt.xlabel('Prediction Confidence (%)')
    plt.ylabel('Grid Position')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(predictions_df['probability'] * 100, 
                   predictions_df['grid_position'], 1)
    p = np.poly1d(z)
    plt.plot(predictions_df['probability'] * 100, 
             p(predictions_df['probability'] * 100), 
             "r--", alpha=0.8)
    
    plt.tight_layout()
    return plt.gcf()

def save_visualizations():
    """Create and save all visualizations"""
    # Create results directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    try:
        # Load the data
        predictions_df = pd.read_csv('results/2025_predictions_full.csv')
        historical_df = pd.read_csv('data/processed/historical_engineered.csv')
        
        # Filter for future races only
        current_date = pd.Timestamp.now()
        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
        predictions_df = predictions_df[predictions_df['date'] > current_date]
        
        # Create visualizations for each race
        for race in predictions_df['race_name'].unique():
            fig = create_race_prediction_chart(predictions_df, race)
            fig.savefig(f'results/visualizations/{race.replace(" ", "_")}_predictions.png',
                       bbox_inches='tight', dpi=300)
            plt.close(fig)
        
        # Create and save driver form chart
        fig = create_driver_form_chart(historical_df)
        fig.savefig('results/visualizations/driver_form.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Create and save team performance chart
        fig = create_team_performance_chart(historical_df)
        fig.savefig('results/visualizations/team_performance.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Create and save circuit difficulty chart
        fig = create_circuit_difficulty_chart(historical_df)
        fig.savefig('results/visualizations/circuit_difficulty.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Create and save prediction confidence chart
        fig = create_prediction_confidence_chart(predictions_df)
        fig.savefig('results/visualizations/prediction_confidence.png',
                    bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print("✅ All visualizations created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualizations: {str(e)}")
        return False

def main():
    visualizer = F1Visualizer()
    
    # Load prediction data
    pred_df = visualizer.load_prediction_data('results/2025_predictions_full.csv')
    
    # Load betting data if available
    bet_df = visualizer.load_betting_data()
    
    if pred_df is not None:
        # Filter to only include races from April 11th, 2025 onwards
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df = pred_df[pred_df['date'] >= '2025-04-11']
        
        # Create a comprehensive dashboard
        visualizer.create_dashboard(pred_df, bet_df)
        logger.info("Visualization completed successfully!")
    else:
        logger.error("Failed to create visualizations")

if __name__ == "__main__":
    main() 