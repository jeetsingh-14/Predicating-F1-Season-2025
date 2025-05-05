import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

def analyze_news():
    """Analyze news and generate sentiment scores for drivers"""
    try:
        # Sample sentiment data (since we can't actually fetch news in real-time)
        driver_sentiments = {
            'M. Verstappen': 0.8,    # Very positive sentiment
            'S. Perez': 0.2,         # Slightly positive
            'C. Leclerc': 0.6,       # Moderately positive
            'C. Sainz': 0.7,         # Positive
            'L. Hamilton': 0.5,      # Neutral to positive
            'G. Russell': 0.4,       # Neutral
            'L. Norris': 0.7,        # Positive
            'O. Piastri': 0.6,       # Moderately positive
            'F. Alonso': 0.5,        # Neutral to positive
            'L. Stroll': 0.3,        # Slightly positive
            'E. Ocon': 0.4,          # Neutral
            'P. Gasly': 0.4,         # Neutral
            'A. Albon': 0.6,         # Moderately positive
            'L. Sargeant': 0.3,      # Slightly positive
            'D. Ricciardo': 0.5,     # Neutral to positive
            'Y. Tsunoda': 0.5,       # Neutral to positive
            'V. Bottas': 0.4,        # Neutral
            'Z. Guanyu': 0.4,        # Neutral
            'K. Magnussen': 0.3,     # Slightly positive
            'N. Hulkenberg': 0.4     # Neutral
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {'driver': driver, 'sentiment': sentiment}
            for driver, sentiment in driver_sentiments.items()
        ])
        
        # Add timestamp
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/news_analysis.csv', index=False)
        print("✅ News analysis completed and saved!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in news analysis: {str(e)}")
        return False

def main():
    analyze_news()

if __name__ == "__main__":
    main() 