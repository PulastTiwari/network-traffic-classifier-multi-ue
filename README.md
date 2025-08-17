# Network Traffic Classifier MVP

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

Network Traffic Classifier MVP is a robust, production-ready solution for automated classification of network traffic into seven distinct application categories. Leveraging Python, Flask, and scikit-learn, this project demonstrates a full-stack machine learning pipeline, real-time web dashboard, and RESTful API endpoints for practical deployment in multi-user environments (e.g., cellular networks, enterprise LANs).

Core technologies include:

- **Python 3.8+** for all backend logic and data processing
- **Flask** for the web application and API layer
- **scikit-learn** for model training, evaluation, and inference
- **pandas** and **numpy** for data manipulation
- **Chart.js** for interactive frontend visualizations

The architecture is modular, with clear separation between data ingestion, feature engineering, model training, and web serving. The system supports both real and synthetic datasets, automatic fallback strategies, and live monitoring of prediction statistics. Designed for extensibility, it is suitable for research, prototyping, and real-world deployment.

## Overview

This project implements a technically precise **Network Traffic Classifier** that analyzes network packet flows and classifies them into application categories. It is engineered for multi-UE (User Equipment) scenarios, such as cellular networks and enterprise environments, enabling network operators to optimize bandwidth allocation, enforce quality of service, and detect traffic patterns for security and analytics.

### Classification Categories

- **Video Streaming** (e.g., Netflix, YouTube)
- **Audio Calls** (e.g., VoIP, voice calls)
- **Video Calls** (e.g., Zoom, Teams, FaceTime)
- **Gaming** (e.g., online/multiplayer games)
- **Video Uploads** (e.g., file uploads, live streaming)
- **Browsing** (e.g., HTTP/HTTPS web traffic)
- **Texting** (e.g., messaging, chat applications)

## Core Features

- **Supervised Machine Learning**: RandomForest classifier with 80%+ accuracy
- **Real-time Dashboard**: Interactive web interface for live predictions and monitoring
- **Data Visualization**: Dynamic charts and graphs powered by Chart.js
- **Synthetic Data Fallback**: Automatic generation of realistic traffic samples if real data is unavailable
- **RESTful API**: Endpoints for integration with external systems and automation
- **Rapid Deployment**: Simple setup and launch for local or cloud environments

## Quick Start

### Prerequisites

- **Python 3.8+** ([Download here](https://python.org/downloads/))
- **Git** ([Download here](https://git-scm.com/downloads))
- **Internet connection** (for initial dataset download)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/traffic-classifier.git
cd traffic-classifier

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Run the complete pipeline (trains model)
python3 traffic_classifier.py

# 4. Start the web application
python3 app.py
```

### Access the Application

Open your browser and navigate to: **http://localhost:9000**

## Demo Screenshots

_Add screenshots of your dashboard here_

## Development Setup

### For Contributors

1. **Fork the repository**
2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install development dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
4. **Run tests** (if available):
   ```bash
   python3 -m pytest tests/
   ```

### Step-by-Step Execution

If you want to run individual components:

```bash
# Download and preprocess data
python3 src/data_loader.py

# Generate synthetic data (fallback)
python3 src/synthetic_generator.py

# Train the model
python3 src/model_trainer.py

# Start web interface
python3 app.py
```

## Project Structure

```
traffic-classifier/
├── data/
│   ├── raw/                      # Downloaded UNSW-NB15 dataset
│   ├── processed/               # Preprocessed and cleaned data
│   └── synthetic/               # Generated synthetic traffic data
├── models/
│   ├── traffic_classifier_random_forest.joblib  # Trained ML model
│   ├── feature_engineer.joblib  # Feature engineering pipeline
│   └── model_metadata.joblib    # Model metadata and configuration
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── feature_engineer.py     # Feature engineering and selection
│   ├── model_trainer.py        # ML model training and evaluation
│   └── synthetic_generator.py  # Synthetic traffic data generator
├── static/
│   ├── css/
│   │   └── style.css           # Dashboard styling
│   └── js/
│       └── app.js              # Frontend JavaScript logic
├── templates/
│   └── index.html              # Main dashboard template
├── app.py                      # Flask web application
├── traffic_classifier.py       # Main training pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── DEMO_INSTRUCTIONS.md        # Detailed demo guide
```

## API Endpoints

| Method | Endpoint           | Description                              |
| ------ | ------------------ | ---------------------------------------- |
| GET    | `/api/status`      | Check system status and model info       |
| POST   | `/api/predict`     | Classify network traffic sample          |
| GET    | `/api/simulate`    | Generate real-time traffic simulation    |
| POST   | `/api/train`       | Train fallback model with synthetic data |
| GET    | `/api/data-stream` | Get live prediction statistics           |

### Example API Usage

```bash
# Check system status
curl http://localhost:9000/api/status

# Make a prediction
curl -X POST -H "Content-Type: application/json" \
  -d '{"traffic_data": {"duration": 10, "src_bytes": 1500}}' \
  http://localhost:9000/api/predict

# Start simulation
curl http://localhost:9000/api/simulate
```

## System Architecture

### Data Pipeline

1. **Data Loading**: UNSW-NB15 dataset with automatic download
2. **Preprocessing**: Categorical encoding, normalization, feature selection
3. **Feature Engineering**: 32 derived features → 20 selected via correlation analysis
4. **Model Training**: RandomForest with hyperparameter optimization
5. **Evaluation**: Cross-validation + holdout test set

### Machine Learning Details

- **Algorithm**: RandomForest Classifier
- **Hyperparameters**: Grid search optimization (n_estimators, max_depth, min_samples_split)
- **Features**: 20 engineered features from network flow statistics
- **Performance**: ~80% test accuracy, >99% cross-validation
- **Fallback**: GradientBoosting if RandomForest performance < 80%

### Key Features Used

- Connection patterns (duration, count, service rates)
- Byte statistics (src_bytes, dst_bytes, ratios)
- Error rates (connection errors, service errors)
- Protocol information (TCP/UDP flags)
- Traffic intensity indicators

## Performance Metrics

```
Model Type: RandomForest
Test Accuracy: 79.96%
Cross-Validation: 99.86% (±0.04%)
Categories: 7
Features: 20 engineered features
Training Data: 125,973 samples
Test Data: 22,544 samples
```

### Per-Category Performance

| Category        | Precision | Recall | F1-Score |
| --------------- | --------- | ------ | -------- |
| Browsing        | 0.76      | 0.96   | 0.85     |
| Video Streaming | 0.91      | 0.94   | 0.93     |
| Gaming          | 0.59      | 0.05   | 0.10     |
| Others          | Varies    | Varies | Varies   |

## Dataset Information

**Primary Dataset**: UNSW-NB15 Network Intrusion Detection Dataset

- **Source**: University of New South Wales (Open Access)
- **Size**: ~125K training + 22K test samples
- **Features**: Network flow statistics, timing, protocol info
- **License**: Creative Commons / Open Data

**Fallback**: Synthetic Traffic Generator

- **Purpose**: Ensures system works without internet connection
- **Generation**: Realistic patterns for each traffic category
- **Customizable**: Adjustable sample sizes and distributions

## Usage Examples

### Basic Classification

```python
# Load and classify traffic
from src.data_loader import DataLoader
from src.model_trainer import ModelTrainer

# Train model
trainer = ModelTrainer()
results = trainer.train_with_fallback(X_train, y_train, X_test, y_test)
print(f"Accuracy: {results['accuracy']:.4f}")
```

### Web Interface Demo

1. Start server: `python3 app.py`
2. Open browser: http://localhost:9000
3. Click "🔍 Classify Sample" for instant predictions
4. Click "⚡ Start Real-time Simulation" for live demo
5. View charts and statistics in real-time

## Troubleshooting

### Common Issues

| Issue                  | Solution                               |
| ---------------------- | -------------------------------------- |
| Port already in use    | Change port in `app.py` (line 334)     |
| Dataset download fails | System auto-generates synthetic data   |
| Model accuracy < 80%   | Automatic fallback to GradientBoosting |
| Missing dependencies   | Run `pip3 install -r requirements.txt` |
| Python version error   | Ensure Python 3.8+ is installed        |

### Debug Mode

```bash
# Run with verbose output
DEBUG=1 python3 traffic_classifier.py

# Test individual components
python3 src/synthetic_generator.py  # Test data generation
python3 src/model_trainer.py        # Test model training
```

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## Concepts & Learning Resources

This project leverages several core technologies and concepts. For further learning, refer to the official documentation and resources below:

- [Python Official Documentation](https://docs.python.org/3/)
- [Flask Web Framework](https://flask.palletsprojects.com/)
- [scikit-learn Machine Learning Library](https://scikit-learn.org/stable/)
- [pandas Data Analysis Library](https://pandas.pydata.org/)
- [NumPy Scientific Computing](https://numpy.org/doc/)
- [Chart.js Visualization Library](https://www.chartjs.org/docs/latest/)
- [UNSW-NB15 Dataset Info](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [RESTful API Design](https://restfulapi.net/)
- [PEP 8 Python Style Guide](https://peps.python.org/pep-0008/)

For contributors, please review the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and best practices.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

- **scikit-learn**: BSD License
- **Flask**: BSD License
- **pandas**: BSD License
- **UNSW-NB15 Dataset**: Creative Commons

## Authors & Acknowledgments

- **Project Lead**: Pulast S Tiwari
<<<<<<< HEAD
- **Dataset**: UNSW-NB15 by University of New South Wales
- **Inspiration**: Samsung EnnovateX 2025 AI Challenge, Problem Statement #8
Classify User Application Traffic at the Network in a Multi-UE Connected Scenario
## 📞 Support
=======
- **Contributors**: Pulast S Tiwari
- **Dataset**: UNSW-NB15 by University of New South Wales
- **Inspiration**: Network traffic analysis research community

## Support
>>>>>>> f18ed92 (docs: enhance README with technical description, architecture, and learning resources)

If you encounter issues or have questions:

1. **Check** the troubleshooting section above
2. **Search** existing [GitHub Issues](https://github.com/yourusername/traffic-classifier/issues)
3. **Create** a new issue with detailed description
4. **Contact** the maintainers

---

**Ready to classify network traffic? Get started with the quick setup above!**

---

## Concepts & Learning Resources

This project leverages several core technologies and concepts. For further learning, refer to the official documentation and resources below:

- [Python Official Documentation](https://docs.python.org/3/)
- [Flask Web Framework](https://flask.palletsprojects.com/)
- [scikit-learn Machine Learning Library](https://scikit-learn.org/stable/)
- [pandas Data Analysis Library](https://pandas.pydata.org/)
- [NumPy Scientific Computing](https://numpy.org/doc/)
- [Chart.js Visualization Library](https://www.chartjs.org/docs/latest/)
- [UNSW-NB15 Dataset Info](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [RESTful API Design](https://restfulapi.net/)
- [PEP 8 Python Style Guide](https://peps.python.org/pep-0008/)

For contributors, please review the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and best practices.
