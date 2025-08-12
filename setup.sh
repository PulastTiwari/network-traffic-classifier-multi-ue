#!/bin/bash

# Network Traffic Classifier - Setup Script
# This script sets up the project environment and runs the complete pipeline

set -e  # Exit on any error

echo "🌐 Network Traffic Classifier - Setup & Installation"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
    
    # Check if Python version is 3.8+
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3,8) else 1)'; then
        echo -e "${GREEN}✓ Python version is compatible${NC}"
    else
        echo -e "${RED}✗ Python 3.8+ required. Please upgrade Python.${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

# Check if virtual environment exists
echo -e "${BLUE}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate || {
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
}
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed successfully${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p data/{raw,processed,synthetic}
mkdir -p models
mkdir -p logs
echo -e "${GREEN}✓ Project directories created${NC}"

# Check if model exists
echo -e "${BLUE}Checking for existing model...${NC}"
if [ -f "models/traffic_classifier_random_forest.joblib" ]; then
    echo -e "${GREEN}✓ Trained model found${NC}"
    TRAIN_MODEL=false
else
    echo -e "${YELLOW}! No trained model found - will train new model${NC}"
    TRAIN_MODEL=true
fi

# Ask user if they want to train the model
if [ "$TRAIN_MODEL" = true ]; then
    echo -e "${YELLOW}Do you want to train the model now? This may take 5-10 minutes. (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${BLUE}Training machine learning model...${NC}"
        python3 traffic_classifier.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Model training completed successfully${NC}"
        else
            echo -e "${RED}✗ Model training failed${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Skipping model training. You can train later with: python3 traffic_classifier.py${NC}"
    fi
fi

# Test the installation
echo -e "${BLUE}Testing installation...${NC}"
if python3 -c "from src.synthetic_generator import SyntheticDataGenerator; print('Import test passed')" 2>/dev/null; then
    echo -e "${GREEN}✓ Installation test passed${NC}"
else
    echo -e "${RED}✗ Installation test failed${NC}"
    exit 1
fi

# Final instructions
echo -e "\n${GREEN}🎉 Setup completed successfully!${NC}"
echo -e "\n${BLUE}Next steps:${NC}"
echo -e "1. Start the web application: ${YELLOW}python3 app.py${NC}"
echo -e "2. Open your browser and navigate to: ${YELLOW}http://localhost:9000${NC}"
echo -e "3. Try the interactive demo and classify network traffic!"

echo -e "\n${BLUE}Alternative commands:${NC}"
echo -e "• Train model: ${YELLOW}python3 traffic_classifier.py${NC}"
echo -e "• Generate synthetic data: ${YELLOW}python3 src/synthetic_generator.py${NC}"
echo -e "• Test individual components: ${YELLOW}python3 src/model_trainer.py${NC}"

echo -e "\n${BLUE}For development:${NC}"
echo -e "• Activate virtual environment: ${YELLOW}source venv/bin/activate${NC}"
echo -e "• Run tests: ${YELLOW}python3 -m pytest tests/${NC}"
echo -e "• View API endpoints: ${YELLOW}curl http://localhost:9000/api/status${NC}"

echo -e "\n${BLUE}Need help?${NC}"
echo -e "• Check the README.md for detailed documentation"
echo -e "• Visit the project repository for issues and discussions"
echo -e "• Review CONTRIBUTING.md for development guidelines"

echo -e "\n${GREEN}Happy traffic classifying! 🚀${NC}"
