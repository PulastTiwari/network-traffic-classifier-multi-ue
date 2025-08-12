#!/usr/bin/env python3
"""
Flask Web Application for Network Traffic Classifier
Provides web interface and API endpoints for traffic classification
"""

import sys
import os
from pathlib import Path
import json
import random
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib

from synthetic_generator import SyntheticDataGenerator

app = Flask(__name__)
CORS(app)

# Global variables
models_dir = Path('models')
generator = SyntheticDataGenerator(data_dir='data')
current_predictions = []
model_metrics = {}

# Load trained model and components
trained_model = None
feature_engineer = None
categories = ['Video Streaming', 'Audio Calls', 'Video Calls', 'Gaming', 'Video Uploads', 'Browsing', 'Texting']

def load_model():
    """Load the trained model and feature engineer"""
    global trained_model, feature_engineer
    
    try:
        # Load model
        model_path = models_dir / 'traffic_classifier_random_forest.joblib'
        if not model_path.exists():
            # Try alternative naming
            alt_paths = list(models_dir.glob('traffic_classifier_*.joblib'))
            if alt_paths:
                model_path = alt_paths[0]
        
        if model_path.exists():
            trained_model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("No trained model found")
            
        # Load feature engineer
        fe_path = models_dir / 'feature_engineer.joblib'
        if fe_path.exists():
            feature_engineer = joblib.load(fe_path)
            print(f"Feature engineer loaded from {fe_path}")
        else:
            print("No feature engineer found")
            
    except Exception as e:
        print(f"Error loading model components: {e}")

# Load model on startup
load_model()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """Get API status and model information"""
    return jsonify({
        'status': 'active',
        'model_loaded': trained_model is not None,
        'feature_engineer_loaded': feature_engineer is not None,
        'categories': categories,
        'total_predictions': len(current_predictions)
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    """Train a new model (fallback if main model not available)"""
    try:
        print("Training fallback model...")
        
        # Generate synthetic data for quick training
        dataset = generator.generate_dataset(samples_per_category=200)
        
        # Simple training pipeline
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, classification_report
        
        # Prepare features
        feature_cols = ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
                       'serror_rate', 'srv_serror_rate', 'same_srv_rate', 'diff_srv_rate']
        
        X = dataset[feature_cols].fillna(0)
        y = dataset['category']
        
        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save as fallback
        global trained_model
        trained_model = {'model': model, 'scaler': scaler, 'features': feature_cols}
        
        global model_metrics
        model_metrics = {
            'accuracy': float(accuracy),
            'features_used': len(feature_cols),
            'training_samples': len(X_train),
            'categories': list(y.unique())
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Fallback model trained successfully',
            'accuracy': float(accuracy),
            'features': len(feature_cols)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Training failed: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Make traffic classification predictions"""
    try:
        # Check if model is available
        if trained_model is None:
            return jsonify({
                'status': 'error', 
                'message': 'No model available. Please train a model first.'
            })
        
        # Get input data
        if request.json and 'traffic_data' in request.json:
            traffic_data = request.json['traffic_data']
            input_df = pd.DataFrame([traffic_data])
        else:
            # Generate random sample for demo
            category = random.choice(categories)
            sample = generator.generate_sample(category, 1)
            input_df = sample
            traffic_data = input_df.iloc[0].to_dict()
        
        # Make prediction
        if isinstance(trained_model, dict):
            # Fallback model format
            features = trained_model['features']
            X = input_df[features].fillna(0)
            X_scaled = trained_model['scaler'].transform(X)
            
            prediction = trained_model['model'].predict(X_scaled)[0]
            probabilities = trained_model['model'].predict_proba(X_scaled)[0]
            classes = trained_model['model'].classes_
            
        else:
            # Full model format
            if feature_engineer is not None:
                X_processed = feature_engineer.transform_new_data(input_df)
                prediction = trained_model.predict(X_processed)[0]
                probabilities = trained_model.predict_proba(X_processed)[0]
                classes = feature_engineer.get_category_names()
            else:
                return jsonify({'status': 'error', 'message': 'Feature engineer not available'})
        
        # Prepare result
        result = {
            'predicted_category': str(prediction) if hasattr(prediction, '__iter__') else prediction,
            'confidence': float(max(probabilities)),
            'probabilities': {
                str(classes[i]): float(prob) for i, prob in enumerate(probabilities)
            },
            'timestamp': datetime.now().isoformat(),
            'input_features': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                             for k, v in traffic_data.items()}
        }
        
        # Store prediction
        global current_predictions
        current_predictions.append(result)
        if len(current_predictions) > 50:
            current_predictions = current_predictions[-50:]
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        })

@app.route('/api/simulate')
def api_simulate():
    """Simulate real-time traffic classification"""
    try:
        if trained_model is None:
            return jsonify({'status': 'error', 'message': 'No model available'})
        
        # Generate random traffic
        true_category = random.choice(categories)
        sample = generator.generate_sample(true_category, 1)
        
        # Make prediction (reuse predict logic)
        input_df = sample
        traffic_data = input_df.iloc[0].to_dict()
        
        if isinstance(trained_model, dict):
            features = trained_model['features']
            X = input_df[features].fillna(0)
            X_scaled = trained_model['scaler'].transform(X)
            prediction = trained_model['model'].predict(X_scaled)[0]
            probabilities = trained_model['model'].predict_proba(X_scaled)[0]
        else:
            X_processed = feature_engineer.transform_new_data(input_df)
            prediction = trained_model.predict(X_processed)[0]
            probabilities = trained_model.predict_proba(X_processed)[0]
        
        confidence = float(max(probabilities))
        is_correct = str(prediction) == true_category
        
        result = {
            'true_category': true_category,
            'predicted_category': str(prediction),
            'confidence': confidence,
            'correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Simulation failed: {str(e)}'
        })

@app.route('/api/data-stream')
def api_data_stream():
    """Get recent predictions and statistics"""
    global current_predictions, model_metrics
    
    # Calculate basic statistics
    if current_predictions:
        recent = current_predictions[-10:]
        avg_confidence = sum(p.get('confidence', 0) for p in recent) / len(recent)
        
        category_counts = {}
        for pred in current_predictions:
            cat = pred.get('predicted_category', 'Unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
    else:
        avg_confidence = 0
        category_counts = {}
    
    return jsonify({
        'recent_predictions': current_predictions[-20:],
        'statistics': {
            'total_predictions': len(current_predictions),
            'average_confidence': avg_confidence,
            'category_distribution': category_counts
        },
        'model_info': model_metrics
    })

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

def main():
    """Main function to start the Flask app"""
    print("Network Traffic Classifier - Web Interface")
    print("=" * 50)
    
    # Check if model exists
    if trained_model is None:
        print("\nWARNING: No trained model found!")
        print("Please run 'python traffic_classifier.py' first to train a model,")
        print("or use the /api/train endpoint to train a fallback model.")
    else:
        print(f"\nModel loaded successfully!")
        print(f"Categories: {categories}")
    
    print(f"\nStarting web server...")
    print(f"Dashboard: http://localhost:9000")
    print(f"API Status: http://localhost:9000/api/status")
    
    print(f"\nAPI Endpoints:")
    print(f"  GET  /api/status      - Check API status")
    print(f"  POST /api/train       - Train fallback model")
    print(f"  POST /api/predict     - Classify traffic")
    print(f"  GET  /api/simulate    - Simulate classification")
    print(f"  GET  /api/data-stream - Get prediction data")
    
    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0', port=9000)

if __name__ == '__main__':
    main()
