# src/visualize_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os

def load_prediction_data():
    """Load the prediction results."""
    full_predictions = pd.read_csv('results/2025_predictions_full.csv')
    
    # Add confidence levels based on podium probability
    full_predictions['confidence'] = pd.cut(
        full_predictions['podium_probability'],
        bins=[0, 0.4, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return full_predictions

def create_podium_probability_heatmap(predictions):
    """Create a heatmap of podium probabilities by driver and race."""
    plt.figure(figsize=(15, 8))
    
    # Pivot the data for the heatmap
    heatmap_data = predictions.pivot(
        index='driver', 
        columns='race_name', 
        values='podium_probability'
    )
    
    # Create heatmap
    sns.heatmap(
        heatmap_data, 
        cmap='RdYlGn', 
        center=0.5,
        annot=True, 
        fmt='.2f', 
        cbar_kws={'label': 'Podium Probability'}
    )
    
    plt.title('Podium Probability Heatmap by Driver and Race')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/visualizations/podium_probability_heatmap.png')
    plt.close()

def create_driver_performance_bar_plot(predictions):
    """Create a bar plot of average podium probabilities by driver."""
    avg_driver_prob = predictions.groupby('driver')['podium_probability'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    avg_driver_prob.plot(kind='bar')
    plt.title('Average Podium Probability by Driver')
    plt.xlabel('Driver')
    plt.ylabel('Average Podium Probability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/visualizations/driver_performance_bar.png')
    plt.close()

def create_team_performance_box_plot(predictions):
    """Create a box plot of podium probabilities by team."""
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=predictions, x='team', y='podium_probability')
    plt.title('Team Performance Distribution')
    plt.xlabel('Team')
    plt.ylabel('Podium Probability')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/visualizations/team_performance_box.png')
    plt.close()

def create_interactive_race_predictions(predictions):
    """Create an interactive plot for race predictions."""
    fig = px.scatter(
        predictions,
        x='race_name',
        y='podium_probability',
        color='team',
        size='podium_probability',
        hover_data=['driver', 'confidence', 'grid_position'],
        title='Race Predictions Overview',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        xaxis_title='Race',
        yaxis_title='Podium Probability',
        xaxis_tickangle=45,
        height=600,
        showlegend=True,
        legend_title='Team',
        hovermode='closest'
    )
    
    # Save as HTML for interactivity
    fig.write_html('results/visualizations/interactive_race_predictions.html')

def create_confidence_distribution(predictions):
    """Create a stacked bar chart of confidence levels by race."""
    confidence_counts = predictions.groupby(['race_name', 'confidence']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    confidence_counts.plot(kind='bar', stacked=True)
    plt.title('Confidence Distribution by Race')
    plt.xlabel('Race')
    plt.ylabel('Number of Predictions')
    plt.legend(title='Confidence Level')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/visualizations/confidence_distribution.png')
    plt.close()

def create_grid_vs_podium_plot(predictions):
    """Create a scatter plot of grid position vs podium probability."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=predictions,
        x='grid_position',
        y='podium_probability',
        hue='team',
        style='race_name',
        s=100
    )
    
    plt.title('Grid Position vs Podium Probability')
    plt.xlabel('Grid Position')
    plt.ylabel('Podium Probability')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/visualizations/grid_vs_podium.png')
    plt.close()

def main():
    # Create visualizations directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Load prediction data
    predictions = load_prediction_data()
    
    print("🎨 Creating visualizations...")
    
    # Create various visualizations
    create_podium_probability_heatmap(predictions)
    create_driver_performance_bar_plot(predictions)
    create_team_performance_box_plot(predictions)
    create_interactive_race_predictions(predictions)
    create_confidence_distribution(predictions)
    create_grid_vs_podium_plot(predictions)
    
    print("✅ Visualizations created successfully!")
    print("📊 Visualization files saved in results/visualizations/")

if __name__ == "__main__":
    main()
