# F1 Race Outcome Predictor 2025 - Streamlit Dashboard

A comprehensive Streamlit dashboard for the F1 Race Outcome Predictor 2025 project, designed to deliver real-time and historical insights into Formula 1 race predictions using machine learning and deep learning models.

## Features

### Core Functionalities
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

### Data Visualizations & Analytics
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

### Scenario Simulation Engine
- **What-If Engine**
  - Manipulate input features for selected drivers
  - Real-time probability updates
  - Visual feedback on prediction changes
  - AI-powered race preview based on parameters

- **Race Outcome Simulator**
  - Monte Carlo simulation based on feature variances
  - Distribution of outcomes per driver
  - Team performance probability estimates

### Persistence & Export
- **Prediction History**
  - Save predictions to database
  - Filter by race, driver, date
  - Visualize prediction patterns
  - Export to CSV with one click

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

## Running the Dashboard

To run the Streamlit dashboard:

```bash
streamlit run app.py
```

This will start the dashboard on your local machine, typically at http://localhost:8501.

## Dashboard Structure

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
├── data/                   # Data files
│   ├── processed/          # Processed data files
│   └── db/                 # Database files
├── models/                 # Model files
├── results/                # Prediction results
└── dashboad_content/       # Dashboard assets
    ├── F1 2025 Season Cars/    # Car images
    ├── F1 2025 Season Drivers/ # Driver images
    └── F1 Race Tracks/         # Track images
```

## Technical Details

- **Caching**: Efficient data loading with `st.cache_data` and `st.cache_resource`
- **Database**: SQLite for prediction history storage
- **Visualization**: Interactive charts with Plotly and Altair
- **Maps**: Interactive track maps with Folium and PyDeck
- **UI**: Clean, responsive design with dark/light mode toggle
- **Models**: Integration with scikit-learn, XGBoost, and other ML libraries

## Contributing

Contributions are welcome! If you'd like to improve the dashboard, add new features, or fix bugs, please feel free to submit a pull request.

## License

This project is open for public use, learning, and improvement. No warranty is provided. Predictions are for fun and educational purposes only!

## Acknowledgments

- Formula 1 for historical race data
- The open-source community for ML libraries and visualization tools
- Streamlit for the amazing dashboard framework