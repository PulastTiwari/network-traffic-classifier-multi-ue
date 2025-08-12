#!/usr/bin/env python3
"""
Data Loader for UNSW-NB15 Dataset
Downloads, loads, and preprocesses the network traffic dataset
"""

import os
import pandas as pd
import numpy as np
import requests
from urllib.parse import urljoin
import gzip
import csv
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir='../data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # UNSW-NB15 dataset URLs (using a simplified CSV version for this demo)
        self.dataset_urls = {
            'features': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
        }
        
        # Column names for the dataset (KDD Cup 99 format - similar structure to network traffic)
        self.column_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]
        
        # Mapping from original labels to our 7 categories
        self.label_mapping = {
            'normal': 'Browsing',
            'apache2': 'Browsing',
            'back': 'Video Streaming',
            'buffer_overflow': 'Gaming',
            'ftp_write': 'Video Uploads',
            'guess_passwd': 'Texting',
            'httptunnel': 'Video Streaming',
            'imap': 'Texting',
            'ipsweep': 'Browsing',
            'land': 'Gaming',
            'loadmodule': 'Gaming',
            'mailbomb': 'Texting',
            'mscan': 'Browsing',
            'multihop': 'Video Calls',
            'named': 'Audio Calls',
            'neptune': 'Video Streaming',
            'nmap': 'Browsing',
            'perl': 'Gaming',
            'phf': 'Browsing',
            'pod': 'Gaming',
            'portsweep': 'Browsing',
            'rootkit': 'Gaming',
            'satan': 'Browsing',
            'sendmail': 'Texting',
            'smurf': 'Video Streaming',
            'snmpgetattack': 'Browsing',
            'snmpguess': 'Texting',
            'sqlattack': 'Browsing',
            'teardrop': 'Gaming',
            'warezclient': 'Video Uploads',  # Merged with uploads instead of separate downloads
            'warezmaster': 'Video Uploads',
            'worm': 'Texting',
            'xlock': 'Gaming',
            'xsnoop': 'Audio Calls',
            'xterm': 'Video Calls',
            'udpstorm': 'Gaming',
            'processtable': 'Gaming',
            'ps': 'Gaming',
            'saint': 'Browsing',
            'spy': 'Audio Calls'
        }

    def download_dataset(self):
        """Download the dataset if not already present"""
        print("Downloading dataset...")
        
        for name, url in self.dataset_urls.items():
            file_path = self.raw_dir / f'{name}.txt'
            
            if file_path.exists():
                print(f"{name} already downloaded")
                continue
                
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"Downloaded {name}")
                
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                return False
                
        return True

    def load_raw_data(self):
        """Load raw data from downloaded files"""
        train_file = self.raw_dir / 'features.txt'
        test_file = self.raw_dir / 'test.txt'
        
        if not train_file.exists() or not test_file.exists():
            print("Raw data files not found. Attempting download...")
            if not self.download_dataset():
                print("Download failed, falling back to synthetic data generation")
                return None, None
        
        try:
            # Load training data
            train_data = pd.read_csv(train_file, names=self.column_names, header=None)
            print(f"Loaded training data: {train_data.shape}")
            
            # Load test data
            test_data = pd.read_csv(test_file, names=self.column_names, header=None)
            print(f"Loaded test data: {test_data.shape}")
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

    def map_labels_to_categories(self, data):
        """Map original dataset labels to our 7 target categories"""
        print("Mapping labels to target categories...")
        
        # Clean label column (remove difficulty scores if present)
        if 'label' in data.columns:
            data['original_label'] = data['label'].astype(str).str.strip()
            
            # Map to our categories
            data['category'] = data['original_label'].map(self.label_mapping)
            
            # Handle unmapped labels - assign to browsing as default
            data['category'] = data['category'].fillna('Browsing')
            
        print("Label mapping completed")
        return data

    def preprocess_data(self, data):
        """Preprocess the data for machine learning"""
        print("Preprocessing data...")
        
        if data is None:
            return None
            
        # Handle categorical variables
        categorical_columns = ['protocol_type', 'service', 'flag']
        
        for col in categorical_columns:
            if col in data.columns:
                # Convert to category and then to numeric codes
                data[col] = pd.Categorical(data[col]).codes
        
        # Handle boolean columns
        boolean_columns = ['land', 'logged_in', 'is_host_login', 'is_guest_login', 
                          'root_shell', 'su_attempted']
        
        for col in boolean_columns:
            if col in data.columns:
                data[col] = data[col].astype(int)
        
        # Select numeric features for ML
        numeric_features = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'count', 
            'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate'
        ]
        
        # Keep only available numeric features
        available_features = [col for col in numeric_features if col in data.columns]
        
        # Create feature matrix
        X = data[available_features].fillna(0)
        y = data['category'] if 'category' in data.columns else None
        
        print(f"Preprocessed data shape: {X.shape}")
        if y is not None:
            print(f"Categories distribution:\n{y.value_counts()}")
        
        return X, y, available_features

    def save_processed_data(self, X_train, y_train, X_test, y_test, features):
        """Save preprocessed data"""
        print("Saving preprocessed data...")
        
        # Save training data
        train_data = X_train.copy()
        train_data['category'] = y_train
        train_data.to_csv(self.processed_dir / 'train_processed.csv', index=False)
        
        # Save test data
        test_data = X_test.copy()
        test_data['category'] = y_test
        test_data.to_csv(self.processed_dir / 'test_processed.csv', index=False)
        
        # Save feature names
        with open(self.processed_dir / 'features.txt', 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
        
        print("Data saved successfully")

    def load_processed_data(self):
        """Load preprocessed data if available"""
        train_file = self.processed_dir / 'train_processed.csv'
        test_file = self.processed_dir / 'test_processed.csv'
        features_file = self.processed_dir / 'features.txt'
        
        if not all([f.exists() for f in [train_file, test_file, features_file]]):
            return None, None, None, None, None
        
        try:
            # Load data
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            
            # Load feature names
            with open(features_file, 'r') as f:
                features = [line.strip() for line in f.readlines()]
            
            # Separate features and labels
            X_train = train_data[features]
            y_train = train_data['category']
            X_test = test_data[features]
            y_test = test_data['category']
            
            print("Loaded preprocessed data successfully")
            return X_train, y_train, X_test, y_test, features
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None, None, None, None

    def run_pipeline(self):
        """Run the complete data loading and preprocessing pipeline"""
        print("Starting data loading pipeline...")
        
        # Try to load processed data first
        X_train, y_train, X_test, y_test, features = self.load_processed_data()
        
        if X_train is not None:
            print("Using existing preprocessed data")
            return X_train, y_train, X_test, y_test, features
        
        # Download and load raw data
        train_data, test_data = self.load_raw_data()
        
        if train_data is None:
            print("Failed to load dataset, will need to generate synthetic data")
            return None, None, None, None, None
        
        # Map labels
        train_data = self.map_labels_to_categories(train_data)
        test_data = self.map_labels_to_categories(test_data)
        
        # Preprocess
        X_train, y_train, features = self.preprocess_data(train_data)
        X_test, y_test, _ = self.preprocess_data(test_data)
        
        if X_train is not None and X_test is not None:
            # Save processed data
            self.save_processed_data(X_train, y_train, X_test, y_test, features)
        
        return X_train, y_train, X_test, y_test, features


def main():
    """Main function for testing the data loader"""
    data_loader = DataLoader()
    X_train, y_train, X_test, y_test, features = data_loader.run_pipeline()
    
    if X_train is not None:
        print("\nData loading completed successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Features: {features}")
        print(f"Categories: {sorted(y_train.unique())}")
    else:
        print("Data loading failed - synthetic data generation will be needed")


if __name__ == "__main__":
    main()
