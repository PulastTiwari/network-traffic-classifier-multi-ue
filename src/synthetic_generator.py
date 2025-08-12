#!/usr/bin/env python3
"""
Synthetic Network Traffic Data Generator
Generates realistic network traffic patterns for the 7 target categories
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

class SyntheticDataGenerator:
    def __init__(self, data_dir='../data'):
        self.data_dir = Path(data_dir)
        self.synthetic_dir = self.data_dir / 'synthetic'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the 7 target categories
        self.categories = [
            'Video Streaming', 'Audio Calls', 'Video Calls', 
            'Gaming', 'Video Uploads', 'Browsing', 'Texting'
        ]
        
        # Define feature names
        self.features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'count', 
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate'
        ]
        
        # Traffic patterns for each category
        self.traffic_patterns = {
            'Video Streaming': {
                'duration': (10, 7200),  # 10 sec to 2 hours
                'src_bytes': (1000, 50000),  # High bandwidth
                'dst_bytes': (5000000, 500000000),  # Very high download
                'protocol_type': 1,  # TCP
                'service': 3,  # HTTP
                'count': (1, 100),
                'srv_count': (1, 50),
                'same_srv_rate': (0.8, 1.0)
            },
            'Audio Calls': {
                'duration': (30, 3600),  # 30 sec to 1 hour
                'src_bytes': (8000, 64000),  # Voice data
                'dst_bytes': (8000, 64000),  # Bidirectional
                'protocol_type': 0,  # UDP
                'service': 15,  # RTP/Audio
                'count': (10, 1000),
                'srv_count': (1, 10),
                'same_srv_rate': (0.9, 1.0)
            },
            'Video Calls': {
                'duration': (60, 3600),  # 1 min to 1 hour
                'src_bytes': (50000, 200000),  # Video + audio
                'dst_bytes': (50000, 200000),  # Bidirectional
                'protocol_type': 0,  # UDP
                'service': 16,  # Video/RTC
                'count': (20, 2000),
                'srv_count': (1, 5),
                'same_srv_rate': (0.95, 1.0)
            },
            'Gaming': {
                'duration': (300, 7200),  # 5 min to 2 hours
                'src_bytes': (100, 5000),  # Low latency, small packets
                'dst_bytes': (100, 10000),
                'protocol_type': 0,  # UDP mostly
                'service': 20,  # Gaming
                'count': (50, 5000),
                'srv_count': (1, 3),
                'same_srv_rate': (0.7, 1.0)
            },
            'Video Uploads': {
                'duration': (60, 1800),  # 1 min to 30 min
                'src_bytes': (1000000, 1000000000),  # Very high upload
                'dst_bytes': (1000, 10000),  # Small responses
                'protocol_type': 1,  # TCP
                'service': 3,  # HTTP/HTTPS
                'count': (1, 50),
                'srv_count': (1, 10),
                'same_srv_rate': (0.6, 0.9)
            },
            'Browsing': {
                'duration': (1, 300),  # 1 sec to 5 min
                'src_bytes': (500, 5000),  # Small requests
                'dst_bytes': (1000, 100000),  # Various page sizes
                'protocol_type': 1,  # TCP
                'service': 3,  # HTTP/HTTPS
                'count': (1, 20),
                'srv_count': (1, 20),
                'same_srv_rate': (0.3, 0.8)
            },
            'Texting': {
                'duration': (1, 10),  # Very short
                'src_bytes': (10, 1000),  # Very small
                'dst_bytes': (10, 1000),  # Very small
                'protocol_type': 1,  # TCP mostly
                'service': 25,  # Messaging
                'count': (1, 10),
                'srv_count': (1, 5),
                'same_srv_rate': (0.5, 1.0)
            }
        }

    def generate_sample(self, category):
        """Generate a single sample for the given category"""
        pattern = self.traffic_patterns[category]
        sample = {}
        
        # Generate features based on category patterns
        sample['duration'] = np.random.uniform(*pattern['duration'])
        sample['src_bytes'] = np.random.uniform(*pattern['src_bytes'])
        sample['dst_bytes'] = np.random.uniform(*pattern['dst_bytes'])
        sample['protocol_type'] = pattern['protocol_type']
        sample['service'] = pattern['service']
        sample['count'] = np.random.randint(*pattern['count'])
        sample['srv_count'] = np.random.randint(*pattern['srv_count'])
        sample['same_srv_rate'] = np.random.uniform(*pattern['same_srv_rate'])
        
        # Generate other features with some correlation
        sample['flag'] = np.random.randint(0, 11)  # Various TCP flags
        sample['wrong_fragment'] = np.random.randint(0, 2)
        sample['urgent'] = np.random.randint(0, 2)
        sample['hot'] = np.random.randint(0, 5)
        
        # Error rates (generally low)
        sample['serror_rate'] = np.random.uniform(0, 0.1)
        sample['srv_serror_rate'] = np.random.uniform(0, 0.1)
        sample['rerror_rate'] = np.random.uniform(0, 0.05)
        sample['srv_rerror_rate'] = np.random.uniform(0, 0.05)
        sample['diff_srv_rate'] = np.random.uniform(0, 1.0)
        
        # Add some noise and correlations
        if category == 'Gaming':
            sample['urgent'] = np.random.randint(0, 3)  # More urgent packets
        elif category in ['Video Streaming', 'Video Uploads']:
            sample['hot'] = np.random.randint(1, 10)  # More "hot" indicators
        elif category == 'Texting':
            sample['count'] = min(sample['count'], 5)  # Fewer connections
            
        sample['category'] = category
        return sample

    def generate_dataset(self, n_samples_per_category=1000):
        """Generate a complete synthetic dataset"""
        print(f"Generating synthetic dataset with {n_samples_per_category} samples per category...")
        
        all_samples = []
        
        for category in self.categories:
            print(f"Generating {category} samples...")
            for _ in range(n_samples_per_category):
                sample = self.generate_sample(category)
                all_samples.append(sample)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_samples)
        
        # Shuffle the data
        df = df.sample(frac=1).reset_index(drop=True)
        
        print(f"Generated {len(df)} total samples")
        print(f"Category distribution:\n{df['category'].value_counts()}")
        
        return df

    def split_data(self, df, train_ratio=0.8):
        """Split data into training and test sets"""
        n_train = int(len(df) * train_ratio)
        
        train_df = df[:n_train].copy()
        test_df = df[n_train:].copy()
        
        print(f"Training set: {len(train_df)} samples")
        print(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df

    def save_synthetic_data(self, train_df, test_df):
        """Save synthetic data in the same format as real data"""
        # Separate features and labels
        feature_cols = [col for col in train_df.columns if col != 'category']
        
        X_train = train_df[feature_cols]
        y_train = train_df['category']
        X_test = test_df[feature_cols]
        y_test = test_df['category']
        
        # Save to processed directory
        train_data = X_train.copy()
        train_data['category'] = y_train
        train_data.to_csv(self.processed_dir / 'train_processed.csv', index=False)
        
        test_data = X_test.copy()
        test_data['category'] = y_test
        test_data.to_csv(self.processed_dir / 'test_processed.csv', index=False)
        
        # Save feature names
        with open(self.processed_dir / 'features.txt', 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        # Also save raw synthetic data
        train_df.to_csv(self.synthetic_dir / 'synthetic_train.csv', index=False)
        test_df.to_csv(self.synthetic_dir / 'synthetic_test.csv', index=False)
        
        print("Synthetic data saved successfully")
        
        return X_train, y_train, X_test, y_test, feature_cols

    def generate_and_save(self, n_samples_per_category=1000):
        """Generate and save complete synthetic dataset"""
        print("Starting synthetic data generation...")
        
        # Generate dataset
        df = self.generate_dataset(n_samples_per_category)
        
        # Split into train/test
        train_df, test_df = self.split_data(df)
        
        # Save data
        X_train, y_train, X_test, y_test, features = self.save_synthetic_data(train_df, test_df)
        
        print("Synthetic data generation completed!")
        return X_train, y_train, X_test, y_test, features

    def add_realistic_noise(self, df, noise_level=0.1):
        """Add realistic noise to make synthetic data more believable"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'category']
        
        for col in numeric_cols:
            if col not in ['protocol_type', 'service', 'flag']:  # Don't add noise to categorical
                std = df[col].std()
                noise = np.random.normal(0, std * noise_level, len(df))
                df[col] += noise
                df[col] = np.maximum(df[col], 0)  # Ensure no negative values
        
        return df


def main():
    """Main function for testing synthetic data generation"""
    generator = SyntheticDataGenerator()
    X_train, y_train, X_test, y_test, features = generator.generate_and_save(n_samples_per_category=500)
    
    print(f"\nSynthetic data generation completed!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {features}")
    print(f"Categories: {sorted(y_train.unique())}")


if __name__ == "__main__":
    main()
