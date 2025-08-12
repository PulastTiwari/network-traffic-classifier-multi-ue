#!/usr/bin/env python3
"""
Feature Engineering for Network Traffic Classification
Creates meaningful features from raw network traffic data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def create_traffic_features(self, X):
        """Create domain-specific features for network traffic classification"""
        print("Creating traffic-specific features...")
        
        X_enhanced = X.copy()
        
        # Bandwidth features
        X_enhanced['total_bytes'] = X_enhanced['src_bytes'] + X_enhanced['dst_bytes']
        X_enhanced['byte_ratio'] = X_enhanced['src_bytes'] / (X_enhanced['dst_bytes'] + 1)
        X_enhanced['bytes_per_second'] = X_enhanced['total_bytes'] / (X_enhanced['duration'] + 1)
        
        # Connection patterns
        X_enhanced['connections_per_service'] = X_enhanced['count'] / (X_enhanced['srv_count'] + 1)
        X_enhanced['error_rate_total'] = X_enhanced['serror_rate'] + X_enhanced['rerror_rate']
        X_enhanced['srv_error_rate_total'] = X_enhanced['srv_serror_rate'] + X_enhanced['srv_rerror_rate']
        
        # Traffic intensity features
        X_enhanced['high_volume_flag'] = (X_enhanced['total_bytes'] > X_enhanced['total_bytes'].quantile(0.9)).astype(int)
        X_enhanced['long_duration_flag'] = (X_enhanced['duration'] > X_enhanced['duration'].quantile(0.9)).astype(int)
        X_enhanced['high_frequency_flag'] = (X_enhanced['count'] > X_enhanced['count'].quantile(0.9)).astype(int)
        
        # Protocol-specific features
        X_enhanced['is_udp'] = (X_enhanced['protocol_type'] == 0).astype(int)
        X_enhanced['is_tcp'] = (X_enhanced['protocol_type'] == 1).astype(int)
        
        # Service patterns
        X_enhanced['service_diversity'] = X_enhanced['diff_srv_rate'] * X_enhanced['srv_count']
        X_enhanced['connection_stability'] = X_enhanced['same_srv_rate'] * (1 - X_enhanced['error_rate_total'])
        
        # Advanced ratios
        X_enhanced['upload_download_ratio'] = np.log1p(X_enhanced['src_bytes']) / (np.log1p(X_enhanced['dst_bytes']) + 1)
        X_enhanced['packets_per_connection'] = (X_enhanced['src_bytes'] + X_enhanced['dst_bytes']) / (X_enhanced['count'] + 1)
        
        print(f"Enhanced features shape: {X_enhanced.shape}")
        return X_enhanced
    
    def select_features(self, X, y=None, method='correlation'):
        """Select the most relevant features"""
        print(f"Selecting features using {method} method...")
        
        if method == 'correlation' and y is not None:
            # Calculate correlation with target (for synthetic data)
            if hasattr(y, 'cat'):
                y_numeric = y.cat.codes
            else:
                y_numeric = self.label_encoder.fit_transform(y)
            
            correlations = []
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    corr = np.corrcoef(X[col], y_numeric)[0, 1]
                    correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
            
            # Sort by correlation and select top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in correlations[:20]]
            
        else:
            # Use variance-based selection
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            variances = X[numeric_cols].var()
            high_variance_features = variances[variances > variances.quantile(0.1)].index.tolist()
            selected_features = high_variance_features[:20]
        
        X_selected = X[selected_features]
        print(f"Selected {len(selected_features)} features: {selected_features}")
        
        self.feature_names = selected_features
        return X_selected
    
    def scale_features(self, X_train, X_test=None):
        """Scale features to standard range"""
        print("Scaling features...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def encode_labels(self, y_train, y_test=None):
        """Encode categorical labels to numeric"""
        print("Encoding labels...")
        
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if y_test is not None:
            y_test_encoded = self.label_encoder.transform(y_test)
            return y_train_encoded, y_test_encoded
        
        return y_train_encoded
    
    def get_feature_importance_stats(self, X, y):
        """Calculate basic feature importance statistics"""
        print("Calculating feature importance statistics...")
        
        # Encode labels for correlation calculation
        y_numeric = self.label_encoder.fit_transform(y) if hasattr(y, 'dtype') and y.dtype == 'object' else y
        
        stats = {}
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Basic statistics
                stats[col] = {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'min': X[col].min(),
                    'max': X[col].max(),
                    'correlation_with_target': np.corrcoef(X[col], y_numeric)[0, 1] if not X[col].isna().all() else 0
                }
        
        return stats
    
    def create_traffic_profiles(self, X, y):
        """Create traffic profiles for each category"""
        print("Creating traffic profiles by category...")
        
        profiles = {}
        categories = y.unique()
        
        for category in categories:
            mask = y == category
            category_data = X[mask]
            
            profile = {}
            for col in X.columns:
                if X[col].dtype in ['int64', 'float64']:
                    profile[col] = {
                        'mean': category_data[col].mean(),
                        'median': category_data[col].median(),
                        'std': category_data[col].std(),
                        'percentile_25': category_data[col].quantile(0.25),
                        'percentile_75': category_data[col].quantile(0.75)
                    }
            
            profiles[category] = profile
        
        return profiles
    
    def process_pipeline(self, X_train, y_train, X_test, y_test):
        """Run the complete feature engineering pipeline"""
        print("Starting feature engineering pipeline...")
        
        # Create enhanced features
        X_train_enhanced = self.create_traffic_features(X_train)
        X_test_enhanced = self.create_traffic_features(X_test)
        
        # Select best features
        X_train_selected = self.select_features(X_train_enhanced, y_train)
        X_test_selected = X_test_enhanced[self.feature_names]
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train_selected, X_test_selected)
        
        # Encode labels
        y_train_encoded, y_test_encoded = self.encode_labels(y_train, y_test)
        
        # Create profiles for analysis
        profiles = self.create_traffic_profiles(X_train_scaled, y_train)
        
        print("Feature engineering pipeline completed!")
        print(f"Final feature set: {self.feature_names}")
        
        return (X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 
                self.feature_names, profiles)
    
    def transform_new_data(self, X_new):
        """Transform new data using fitted transformations"""
        # Create enhanced features
        X_new_enhanced = self.create_traffic_features(X_new)
        
        # Select features
        X_new_selected = X_new_enhanced[self.feature_names]
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new_selected)
        X_new_scaled = pd.DataFrame(X_new_scaled, columns=X_new_selected.columns, index=X_new_selected.index)
        
        return X_new_scaled
    
    def get_category_names(self):
        """Get the category names in order"""
        return self.label_encoder.classes_ if hasattr(self.label_encoder, 'classes_') else None


def main():
    """Main function for testing feature engineering"""
    from data_loader import DataLoader
    from synthetic_generator import SyntheticDataGenerator
    
    # Try to load data
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test, features = data_loader.run_pipeline()
    
    # If real data not available, use synthetic data
    if X_train is None:
        print("Using synthetic data for feature engineering test...")
        generator = SyntheticDataGenerator()
        X_train, y_train, X_test, y_test, features = generator.generate_and_save(n_samples_per_category=100)
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    (X_train_processed, y_train_processed, X_test_processed, y_test_processed, 
     final_features, profiles) = engineer.process_pipeline(X_train, y_train, X_test, y_test)
    
    print(f"\nFeature engineering results:")
    print(f"Training data shape: {X_train_processed.shape}")
    print(f"Test data shape: {X_test_processed.shape}")
    print(f"Selected features: {len(final_features)}")
    print(f"Categories: {engineer.get_category_names()}")


if __name__ == "__main__":
    main()
