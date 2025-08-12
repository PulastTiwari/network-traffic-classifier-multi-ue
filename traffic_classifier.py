#!/usr/bin/env python3
"""
Network Traffic Classifier - Main Execution Script
Runs the complete pipeline from data loading to model training
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_loader import DataLoader
from synthetic_generator import SyntheticDataGenerator
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def main():
    """Main execution pipeline"""
    print("="*60)
    print("NETWORK TRAFFIC CLASSIFIER - COMPLETE PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Data Loading
        print("\n[STEP 1] Loading and preprocessing data...")
        data_loader = DataLoader(data_dir='data')
        X_train, y_train, X_test, y_test, features = data_loader.run_pipeline()
        
        # Step 2: Fallback to synthetic data if needed
        if X_train is None:
            print("\n[FALLBACK] Real dataset not available, generating synthetic data...")
            generator = SyntheticDataGenerator(data_dir='data')
            X_train, y_train, X_test, y_test, features = generator.generate_and_save(n_samples_per_category=500)
        
        print(f"Data loaded successfully!")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(features)}")
        print(f"Categories: {sorted(y_train.unique())}")
        
        # Step 3: Feature Engineering
        print("\n[STEP 2] Feature engineering and preprocessing...")
        engineer = FeatureEngineer()
        (X_train_processed, y_train_processed, X_test_processed, y_test_processed, 
         final_features, profiles) = engineer.process_pipeline(X_train, y_train, X_test, y_test)
        
        print(f"Feature engineering completed!")
        print(f"Final feature count: {len(final_features)}")
        
        # Step 4: Model Training
        print("\n[STEP 3] Training machine learning model...")
        trainer = ModelTrainer(models_dir='models')
        trainer.feature_names = final_features
        
        # Train with fallback strategy
        results = trainer.train_with_fallback(
            X_train_processed, y_train_processed, 
            X_test_processed, y_test_processed,
            accuracy_threshold=0.75,  # Lower threshold for demo
            category_names=engineer.get_category_names()
        )
        
        print(f"Model training completed!")
        print(f"Final model type: {trainer.model_type}")
        print(f"Final accuracy: {results['accuracy']:.4f}")
        
        # Step 5: Model Persistence
        print("\n[STEP 4] Saving model and components...")
        model_path = trainer.save_model(feature_engineer=engineer)
        
        # Step 6: Generate visualizations
        print("\n[STEP 5] Generating visualizations...")
        try:
            # Save confusion matrix
            cm_path = Path('models') / 'confusion_matrix.png'
            trainer.plot_confusion_matrix(
                results['confusion_matrix'], 
                category_names=engineer.get_category_names(),
                save_path=str(cm_path)
            )
            
            # Save feature importance plot
            fi_path = Path('models') / 'feature_importance.png'
            trainer.plot_feature_importance(
                final_features, 
                save_path=str(fi_path)
            )
            
            print(f"Visualizations saved to models/ directory")
            
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")
        
        # Step 7: Model Validation
        print("\n[STEP 6] Final model validation...")
        
        # Cross-validation
        cv_scores = trainer.cross_validate_model(X_train_processed, y_train_processed)
        
        # Summary
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model Type: {trainer.model_type.upper()}")
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Cross-Validation Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Categories Classified: {len(engineer.get_category_names())}")
        print(f"Feature Count: {len(final_features)}")
        print(f"Training Samples: {len(X_train_processed)}")
        print(f"Test Samples: {len(X_test_processed)}")
        
        print(f"\nModel saved to: {model_path}")
        print(f"Feature engineer saved to: models/feature_engineer.joblib")
        
        print(f"\nNext steps:")
        print(f"1. Run 'python app.py' to start the web interface")
        print(f"2. Navigate to http://localhost:5000 for the demo")
        
        # Performance summary by category
        print(f"\n[PERFORMANCE BY CATEGORY]")
        report = results['classification_report']
        categories = engineer.get_category_names()
        
        if categories:
            for i, category in enumerate(categories):
                if str(i) in report:
                    precision = report[str(i)]['precision']
                    recall = report[str(i)]['recall']
                    f1 = report[str(i)]['f1-score']
                    print(f"{category:15}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error: {e}")
        print(f"Check the error details above and ensure all requirements are installed.")
        return False



if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
