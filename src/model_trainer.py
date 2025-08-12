#!/usr/bin/env python3
"""
Model Training for Network Traffic Classification
Trains and evaluates machine learning models with fallback strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, models_dir='../models'):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.model_type = None
        self.feature_names = None
        self.label_encoder = None
        self.scaler = None
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                ),
                'param_grid': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [8, 10, 12],
                    'learning_rate': [0.05, 0.1, 0.2]
                }
            }
        }
    
    def train_model(self, X_train, y_train, model_type='random_forest', tune_hyperparameters=True):
        """Train the specified model type"""
        print(f"Training {model_type} model...")
        
        if model_type not in self.model_configs:
            raise ValueError(f"Model type {model_type} not supported")
        
        config = self.model_configs[model_type]
        
        if tune_hyperparameters and len(X_train) > 1000:
            # Hyperparameter tuning for larger datasets
            print("Performing hyperparameter tuning...")
            
            grid_search = GridSearchCV(
                config['model'],
                config['param_grid'],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters
            self.model = config['model']
            self.model.fit(X_train, y_train)
        
        self.model_type = model_type
        print(f"Model training completed!")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test, category_names=None):
        """Evaluate the trained model"""
        print("Evaluating model performance...")
        
        if self.model is None:
            raise ValueError("No model trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        if category_names is not None:
            target_names = [category_names[i] for i in sorted(set(y_test))]
        else:
            target_names = None
            
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    
    def plot_confusion_matrix(self, cm, category_names=None, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        if category_names is not None:
            labels = category_names
        else:
            labels = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names=None, top_n=15, save_path=None):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature importance")
            return None
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create feature importance dataframe
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_imp)), feature_imp['importance'])
        plt.yticks(range(len(feature_imp)), feature_imp['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances ({self.model_type.title()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        return plt.gcf()
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"Performing {cv}-fold cross-validation...")
        
        if self.model is None:
            raise ValueError("No model trained yet")
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, filename=None, feature_engineer=None):
        """Save the trained model and associated objects"""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        if filename is None:
            filename = f'traffic_classifier_{self.model_type}.joblib'
        
        model_path = self.models_dir / filename
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save feature engineer if provided
        if feature_engineer is not None:
            fe_path = self.models_dir / 'feature_engineer.joblib'
            joblib.dump(feature_engineer, fe_path)
            print(f"Feature engineer saved to {fe_path}")
        
        # Save model metadata
        metadata = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'model_path': str(model_path)
        }
        
        metadata_path = self.models_dir / 'model_metadata.joblib'
        joblib.dump(metadata, metadata_path)
        print(f"Model metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, filename=None):
        """Load a saved model"""
        if filename is None:
            # Try to load the latest model
            metadata_path = self.models_dir / 'model_metadata.joblib'
            if metadata_path.exists():
                metadata = joblib.load(metadata_path)
                filename = Path(metadata['model_path']).name
                self.model_type = metadata['model_type']
                self.feature_names = metadata['feature_names']
            else:
                filename = 'traffic_classifier.joblib'
        
        model_path = self.models_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        return self.model
    
    def predict_new_data(self, X_new):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        predictions = self.model.predict(X_new)
        probabilities = self.model.predict_proba(X_new)
        
        return predictions, probabilities
    
    def train_with_fallback(self, X_train, y_train, X_test, y_test, 
                           accuracy_threshold=0.80, category_names=None):
        """Train model with fallback strategy if accuracy is too low"""
        print("Training model with fallback strategy...")
        
        # Try RandomForest first
        print("\n--- Trying RandomForest ---")
        self.train_model(X_train, y_train, model_type='random_forest')
        rf_results = self.evaluate_model(X_test, y_test, category_names)
        
        if rf_results['accuracy'] >= accuracy_threshold:
            print(f"RandomForest achieved {rf_results['accuracy']:.4f} accuracy (>= {accuracy_threshold})")
            return rf_results
        
        # Fallback to GradientBoosting
        print(f"\nRandomForest accuracy ({rf_results['accuracy']:.4f}) below threshold ({accuracy_threshold})")
        print("--- Falling back to GradientBoosting ---")
        
        self.train_model(X_train, y_train, model_type='gradient_boosting')
        gb_results = self.evaluate_model(X_test, y_test, category_names)
        
        if gb_results['accuracy'] >= rf_results['accuracy']:
            print(f"GradientBoosting achieved better accuracy: {gb_results['accuracy']:.4f}")
            return gb_results
        else:
            print(f"RandomForest performed better, reverting...")
            self.train_model(X_train, y_train, model_type='random_forest')
            return rf_results


def main():
    """Main function for testing model training"""
    from data_loader import DataLoader
    from synthetic_generator import SyntheticDataGenerator
    from feature_engineer import FeatureEngineer
    
    # Try to load data
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test, features = data_loader.run_pipeline()
    
    # If real data not available, use synthetic data
    if X_train is None:
        print("Using synthetic data for model training test...")
        generator = SyntheticDataGenerator()
        X_train, y_train, X_test, y_test, features = generator.generate_and_save(n_samples_per_category=200)
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    (X_train_processed, y_train_processed, X_test_processed, y_test_processed, 
     final_features, profiles) = engineer.process_pipeline(X_train, y_train, X_test, y_test)
    
    # Train model
    trainer = ModelTrainer()
    trainer.feature_names = final_features
    
    # Train with fallback strategy
    results = trainer.train_with_fallback(
        X_train_processed, y_train_processed, 
        X_test_processed, y_test_processed,
        category_names=engineer.get_category_names()
    )
    
    # Save model
    trainer.save_model(feature_engineer=engineer)
    
    # Plot results
    if results['confusion_matrix'] is not None:
        trainer.plot_confusion_matrix(
            results['confusion_matrix'], 
            category_names=engineer.get_category_names()
        )
    
    trainer.plot_feature_importance(final_features)
    
    print(f"\nModel training completed!")
    print(f"Final accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
