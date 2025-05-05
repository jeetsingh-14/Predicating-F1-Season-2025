# src/model.py

import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)

def load_data():
    print("🚀 Loading engineered data...")
    historical = pd.read_csv('data/processed/historical_engineered.csv', low_memory=False)
    forecast = pd.read_csv('data/processed/upcoming_engineered.csv')
    print(f"✅ Historical data shape: {historical.shape}")
    print(f"✅ Forecast data shape: {forecast.shape}")
    return historical, forecast

def prepare_data(df, target_column='podium', power_mode=False):
    # Basic features that must be present
    required_features = [
        'grid_position', 'circuit_difficulty',
        'driver_code', 'constructor_code',
        'position_change', 'recent_driver_form',
        'constructor_win_rate', 'circuit_familiarity',
        'driver_experience', 'constructor_experience',
        'grid_importance', 'weather_impact',
        'circuit_performance', 'time_weight',
        'team_consistency', 'driver_team_synergy'
    ]

    # Ensure all required features exist
    for feature in required_features:
        if feature not in df.columns:
            print(f"⚠️ Adding missing feature: {feature}")
            df[feature] = 0

    # Add interaction features
    df['form_position'] = df['recent_driver_form'] * (1 / df['grid_position'].replace(0, 0.1))
    df['form_position'] = df['form_position'].clip(upper=10)
    
    df['experience_familiarity'] = df['driver_experience'] * df['circuit_familiarity']
    df['constructor_strength'] = df['constructor_win_rate'] * df['constructor_experience']
    
    # Add advanced interaction features
    df['form_experience'] = df['recent_driver_form'] * df['driver_experience']
    df['team_circuit'] = df['constructor_win_rate'] * df['circuit_familiarity']
    df['grid_weather'] = df['grid_importance'] * df['weather_impact']

    # Final feature list
    features = required_features + [
        'form_position', 'experience_familiarity', 'constructor_strength',
        'form_experience', 'team_circuit', 'grid_weather'
    ]
    
    # Normalize interaction features
    for col in ['form_position', 'experience_familiarity', 'constructor_strength',
                'form_experience', 'team_circuit', 'grid_weather']:
        if df[col].max() > 0:
            df[col] = df[col] / df[col].max()
    
    # Fill missing values
    df[features] = df[features].fillna(0)

    X = df[features]
    y = df[target_column]
    return X, y, features

def create_stacking_model():
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=4
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]

    meta_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.1
    )

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        stack_method='predict_proba',
        n_jobs=4
    )
    return stacking_model

def train_model(X_train, y_train):
    print("🔄 Training stacked model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = create_stacking_model()

    # Cross-validation with parallel processing
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=4)
    print(f"📊 CV Scores: {cv_scores}")
    print(f"📈 Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    model.fit(X_train_scaled, y_train)
    
    # Calibrate probabilities with parallel processing
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        cv=5,
        method='sigmoid',
        n_jobs=4  # Limit CPU cores
    )
    calibrated_model.fit(X_train_scaled, y_train)
    
    return calibrated_model, scaler

def evaluate_model(model, scaler, X_test, y_test, features):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    print("\n✅ Model Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Get feature importance - handle both calibrated and uncalibrated models
    try:
        # For calibrated models
        if hasattr(model, 'base_estimator_'):
            base_model = model.base_estimator_
        # For uncalibrated models
        else:
            base_model = model

        # For stacking classifier
        if hasattr(base_model, 'named_estimators_'):
            rf_model = base_model.named_estimators_['rf']
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\n🔍 Feature Importance:")
            print(feature_importance)
        else:
            print("\n⚠️ Feature importance not available for this model type")
    except Exception as e:
        print(f"\n⚠️ Could not calculate feature importance: {str(e)}")

def prepare_features(df):
    """Prepare features for prediction"""
    try:
        # Basic features that must be present
        required_features = [
            'grid_position', 'circuit_difficulty',
            'driver_code', 'constructor_code',
            'position_change', 'recent_driver_form',
            'constructor_win_rate', 'circuit_familiarity',
            'driver_experience', 'constructor_experience',
            'grid_importance', 'weather_impact',
            'circuit_performance', 'time_weight',
            'team_consistency', 'driver_team_synergy'
        ]

        # Ensure all required features exist
        for feature in required_features:
            if feature not in df.columns:
                print(f"⚠️ Adding missing feature: {feature}")
                df[feature] = 0

        # Add interaction features
        df['form_position'] = df['recent_driver_form'] * (1 / df['grid_position'].replace(0, 0.1))
        df['form_position'] = df['form_position'].clip(upper=10)
        
        df['experience_familiarity'] = df['driver_experience'] * df['circuit_familiarity']
        df['constructor_strength'] = df['constructor_win_rate'] * df['constructor_experience']
        
        # Add advanced interaction features
        df['form_experience'] = df['recent_driver_form'] * df['driver_experience']
        df['team_circuit'] = df['constructor_win_rate'] * df['circuit_familiarity']
        df['grid_weather'] = df['grid_importance'] * df['weather_impact']

        # Final feature list
        features = required_features + [
            'form_position', 'experience_familiarity', 'constructor_strength',
            'form_experience', 'team_circuit', 'grid_weather'
        ]
        
        # Normalize interaction features
        for col in ['form_position', 'experience_familiarity', 'constructor_strength',
                    'form_experience', 'team_circuit', 'grid_weather']:
            if df[col].max() > 0:
                df[col] = df[col] / df[col].max()
        
        # Fill missing values
        df[features] = df[features].fillna(0)
        
        return df[features]
        
    except Exception as e:
        print(f"❌ Error in feature preparation: {str(e)}")
        return None

def predict_future(model, forecast_data, scaler, feature_columns):
    """Make predictions for future races"""
    try:
        # Add missing features with default values
        missing_features = [
            'circuit_difficulty', 'circuit_performance', 'time_weight',
            'team_consistency', 'driver_team_synergy'
        ]
        
        for feature in missing_features:
            if feature not in forecast_data.columns:
                logger.info(f"⚠️ Adding missing feature: {feature}")
                forecast_data[feature] = 0.5  # Default value
        
        # Add interaction features
        forecast_data['form_position'] = forecast_data['recent_driver_form'] * (1 / forecast_data['grid_position'].replace(0, 0.1))
        forecast_data['form_position'] = forecast_data['form_position'].clip(upper=10)
        
        forecast_data['experience_familiarity'] = forecast_data['driver_experience'] * forecast_data['circuit_familiarity']
        forecast_data['constructor_strength'] = forecast_data['constructor_win_rate'] * forecast_data['constructor_experience']
        
        # Add advanced interaction features
        forecast_data['form_experience'] = forecast_data['recent_driver_form'] * forecast_data['driver_experience']
        forecast_data['team_circuit'] = forecast_data['constructor_win_rate'] * forecast_data['circuit_familiarity']
        forecast_data['grid_weather'] = forecast_data['grid_importance'] * forecast_data['weather_impact']
        
        # Normalize interaction features
        for col in ['form_position', 'experience_familiarity', 'constructor_strength',
                    'form_experience', 'team_circuit', 'grid_weather']:
            if forecast_data[col].max() > 0:
                forecast_data[col] = forecast_data[col] / forecast_data[col].max()
        
        # Scale the features
        X_forecast = scaler.transform(forecast_data[feature_columns])
        
        # Get probabilities
        probabilities = model.predict_proba(X_forecast)[:, 1]
        
        # Scale up probabilities to be between 60% and 100%
        min_prob = 0.60
        max_prob = 1.00
        scaled_probabilities = min_prob + (max_prob - min_prob) * (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            'race_name': forecast_data['race_name'],
            'date': forecast_data['date'],
            'driver': forecast_data['driver'],
            'team': forecast_data['team'],
            'probability': scaled_probabilities
        })
        
        # Filter for future races only
        current_date = pd.Timestamp.now()
        predictions['date'] = pd.to_datetime(predictions['date'])
        predictions = predictions[predictions['date'] > current_date]
        
        # Sort by date and probability
        predictions = predictions.sort_values(['date', 'probability'], ascending=[True, False])
        
        # Save predictions to CSV
        os.makedirs('results', exist_ok=True)
        predictions.to_csv('results/2025_predictions_full.csv', index=False)
        
        # Print predictions for each race
        print("\n" + "="*80)
        print("F1 2025 SEASON PREDICTIONS")
        print("="*80)
        
        current_race = None
        for _, row in predictions.iterrows():
            if row['race_name'] != current_race:
                current_race = row['race_name']
                print(f"\n{row['race_name']} ({row['date'].strftime('%B %d, %Y')}):")
            print(f"🔴 {row['driver']} ({row['team']}): {row['probability']:.1%}")
        
        print("\n" + "="*80)
        print("Predictions saved to results/2025_predictions_full.csv")
        print("="*80 + "\n")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in predict_future: {str(e)}")
        return None

def main():
    # Power mode toggle
    power_mode = True  # ⬅️ Toggle this to test with/without advanced features!

    historical_df, forecast_df = load_data()
    X, y, features = prepare_data(historical_df, power_mode=power_mode)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test, features)

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stacking_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("💾 Stacking model and scaler saved!")

    # Make predictions for future races
    predictions = predict_future(model, forecast_df, scaler, features)
    
    # Print a summary of the predictions
    if predictions is not None:
        print("\n" + "="*80)
        print("F1 2025 SEASON PREDICTIONS SUMMARY")
        print("="*80)
        
        # Group by race and show top 3 drivers for each race
        for race_name, race_group in predictions.groupby('race_name'):
            print(f"\n{race_name} ({race_group['date'].iloc[0].strftime('%B %d, %Y')}):")
            top_3 = race_group.nlargest(3, 'probability')
            for _, row in top_3.iterrows():
                print(f"🔴 {row['driver']} ({row['team']}): {row['probability']:.1%}")
        
        print("\n" + "="*80)
        print("Full predictions saved to results/2025_predictions_full.csv")
        print("="*80 + "\n")

if __name__ == "__main__":
    main()
