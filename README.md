# F1 Predictor 2025

A personal Formula 1 race prediction system by **Ashwinth Reddy**, designed for fun and learning purposes!

This project combines machine learning, news sentiment analysis, and betting odds to predict podium probabilities for upcoming F1 races.  
It is an open learning tool for anyone to explore, improve, and play with. 

> **Disclaimer:** This is a personal academic project. Predictions are made for educational purposes only and are **not** financial or betting advice.

## Features

-  **Machine Learning Model**  
  Uses stacked ensemble of XGBoost, LightGBM, and CatBoost models to predict podium probabilities.

-  **News Sentiment Analysis**  
  Incorporates sentiment from F1 news articles to adjust predictions dynamically.

-  **Betting Odds Integration**  
  Compares model predictions with market odds from major bookmakers. *(Manually sourced from Google Sports snippets and bookmaker websites.)*

-  **Dynamic Schedule Parser**  
  Automatically fetches and updates the F1 race calendar.

-  **Visualization Dashboard**  
  Creates clear visualizations of predictions and comparisons.

-  **Automated Pipeline**  
  End-to-end automation of data collection, feature engineering, model training, and visualizations.


## Installation

1. Clone the repository:
```bash
git clone https://github.com/AshPlayer-1415/f1-predictor-2025.git
cd f1-predictor-2025
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

To run the complete pipeline with all features:

```bash
python src/master_pipeline.py --update-schedule --include-news --include-betting --create-visualizations
```

### Command Line Arguments

- `--update-schedule`: Update the race schedule before running the pipeline
- `--include-news`: Include news sentiment analysis in the pipeline
- `--include-betting`: Include betting odds analysis in the pipeline
- `--create-visualizations`: Create visualizations after running the pipeline

### Running Individual Components

You can also run individual components of the pipeline:

```bash
# Update race schedule
python src/schedule_parser.py

# Analyze news sentiment
python src/news_analyzer.py

# Fetch betting odds
python src/betting_odds_analyzer.py

# Engineer features
python src/feature_engineering.py

# Train model and make predictions
python src/model.py

# Create visualizations
python src/visualization.py
```

## Streamlit Dashboard

A comprehensive Streamlit dashboard for the F1 Race Outcome Predictor 2025 project, designed to deliver real-time and historical insights into Formula 1 race predictions using machine learning and deep learning models.

### Running the Dashboard

To run the Streamlit dashboard:

```bash
streamlit run app.py
```

This will start the dashboard on your local machine, typically at http://localhost:8501.

### Dashboard Features

#### Core Functionalities
- **Dynamic Race Data Loading**
  - Select any upcoming race from the dropdown
  - Upload custom race data in CSV format
  - Automatic validation and synthetic data generation for missing features

- **Model Selection & Explanation**
  - Choose between multiple models (Stacking Ensemble, Random Forest, XGBoost, Logistic Regression, Deep Neural Net)
  - SHAP-based feature importance visualization
  - Detailed model explanation

- **Real-Time Podium Prediction**
  - Display top-3 predicted podium finishers with driver photos and team logos
  - Probability visualization with confidence indicators
  - Toggle between classification and probabilistic views

#### Data Visualizations & Analytics
- **Driver & Constructor Performance Comparison**
  - Average qualifying position trends
  - Driver vs Constructor influence analysis
  - ELO rating progression over time
  - Radar charts for driver skill breakdown

- **Track Insights Panel**
  - Track details and interactive maps
  - Weather forecast
  - Overtaking difficulty and safety car probability
  - Historical team performance at each track

- **Model Metrics Panel**
  - Accuracy, Precision, Recall, Log Loss, F1 Score
  - Confusion Matrix for Podium vs Non-Podium
  - ROC Curve and PR Curve
  - Feature importance visualization

#### Scenario Simulation Engine
- **What-If Engine**
  - Manipulate input features for selected drivers
  - Real-time probability updates
  - Visual feedback on prediction changes
  - AI-powered race preview based on parameters

- **Race Outcome Simulator**
  - Monte Carlo simulation based on feature variances
  - Distribution of outcomes per driver
  - Team performance probability estimates

#### Persistence & Export
- **Prediction History**
  - Save predictions to database
  - Filter by race, driver, date
  - Visualize prediction patterns
  - Export to CSV with one click

## Project Structure

```
f1-predictor-2025/
├── app.py                  # Main dashboard entry point
├── pages/                  # Dashboard pages
│   ├── driver_analysis.py  # Driver & constructor analysis
│   ├── track_insights.py   # Track information and insights
│   ├── model_metrics.py    # Model performance metrics
│   ├── scenario_simulator.py # What-if engine and race simulator
│   └── prediction_history.py # Prediction history and analytics
├── utils/                  # Utility functions
│   ├── data_loader.py      # Data loading functions
│   └── ui_helper.py        # UI component functions
├── data/
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed data files
│   └── db/                 # Database files
├── logs/                   # Pipeline logs
├── models/                 # Model files
├── results/
│   └── visualizations/     # Generated visualizations
├── src/
│   ├── master_pipeline.py        # Main pipeline script
│   ├── schedule_parser.py        # Race schedule parser
│   ├── news_analyzer.py          # News sentiment analyzer
│   ├── betting_odds_analyzer.py  # Betting odds analyzer
│   ├── feature_engineering.py    # Feature engineering
│   ├── model.py                  # Model training and prediction
│   └── visualization.py          # Visualization generator
└── dashboad_content/       # Dashboard assets
    ├── F1 2025 Season Cars/    # Car images
    ├── F1 2025 Season Drivers/ # Driver images
    └── F1 Race Tracks/         # Track images
```

## Model Details

The prediction model uses a stacked ensemble approach:

1. **Base Models**:
   - XGBoost
   - LightGBM
   - CatBoost

2. **Features**:
   - Historical performance
   - Recent form
   - Grid position
   - Circuit characteristics
   - News sentiment (optional)
   - Betting odds (optional)

3. **Output**:
   - Podium probability for each driver, per race

4. **Challenges Faced**:
   - Live data scraping restrictions (manual fallback for betting odds)
   - Complex feature engineering (driver consistency, team synergy, circuit difficulty)
   - Dynamic schedule updates and real-time predictions
   - Integrating multiple models cleanly into one pipeline

5. **Solutions**:
   - Robust error handling
   - Modular design
   - Manual data entry fallback
   - And of course, support from our friendly neighbourhood ChatGPT!

## Technical Details

- **Caching**: Efficient data loading with `st.cache_data` and `st.cache_resource`
- **Database**: SQLite for prediction history storage
- **Visualization**: Interactive charts with Plotly and Altair
- **Maps**: Interactive track maps with Folium and PyDeck
- **UI**: Clean, responsive design with dark/light mode toggle
- **Models**: Integration with scikit-learn, XGBoost, and other ML libraries

## Contributing

Contributions are very welcome!
If you’d like to improve the model, add new features, or just experiment — feel free to fork, clone, and play around.

## License

No license restrictions.

This is a personal project, open for public use, learning, and improvement.
No warranty is provided. Predictions are for fun and educational purposes only!

## Acknowledgments

- 🏁 Formula 1 for historical race data
- 📰 F1 news websites for sentiment data
- 🎰 Google Sports snippets and bookmaker sources for odds data
- 📊 Kaggle datasets for historical performance data
- 🧩 The open-source community for ML libraries
- ☕️ And all the coffee that fueled this project!
