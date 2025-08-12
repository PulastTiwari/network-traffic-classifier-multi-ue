// Network Traffic Classifier - Dashboard JavaScript

// Global variables
let categoryChart = null;
let simulationInterval = null;
let isSimulating = false;
let latestPredictions = [];

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initializing...');
    initializeDashboard();
    checkAPIStatus();
    startDataStream();
});

// Initialize dashboard components
function initializeDashboard() {
    // Initialize category chart
    initializeCategoryChart();
    
    // Set up event listeners
    setupEventListeners();
    
    console.log('Dashboard initialized successfully');
}

// Check API status on load
async function checkAPIStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        updateStatusIndicator(data.status === 'active', 'API Connected');
        updateModelStatus(data);
        
    } catch (error) {
        console.error('Failed to check API status:', error);
        updateStatusIndicator(false, 'API Disconnected');
    }
}

// Update status indicator
function updateStatusIndicator(isActive, message) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.getElementById('status-text');
    
    if (statusDot && statusText) {
        statusDot.classList.toggle('error', !isActive);
        statusText.textContent = message;
    }
}

// Update model status information
function updateModelStatus(statusData) {
    const modelStatus = document.getElementById('model-status');
    const totalPredictions = document.getElementById('total-predictions');
    
    if (modelStatus) {
        modelStatus.textContent = statusData.model_loaded ? 'Ready' : 'Not Loaded';
        modelStatus.style.color = statusData.model_loaded ? 'var(--success)' : 'var(--danger)';
    }
    
    if (totalPredictions) {
        totalPredictions.textContent = statusData.total_predictions || 0;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Button click handlers are defined inline in HTML
    // Additional event listeners can be added here
}

// Initialize category distribution chart
function initializeCategoryChart() {
    const ctx = document.getElementById('categoryChart');
    if (!ctx) return;
    
    const categories = ['Video Streaming', 'Audio Calls', 'Video Calls', 'Gaming', 'Video Uploads', 'Browsing', 'Texting'];
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#84cc16'];
    
    categoryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: [0, 0, 0, 0, 0, 0, 0],
                backgroundColor: colors,
                borderWidth: 2,
                borderColor: '#1e293b'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#f1f5f9',
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 11
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#f1f5f9',
                    bodyColor: '#f1f5f9',
                    borderColor: '#334155',
                    borderWidth: 1
                }
            }
        }
    });
}

// Train model function
async function trainModel() {
    const trainBtn = document.getElementById('train-btn');
    const trainStatus = document.getElementById('train-status');
    
    showLoading(true);
    setButtonState(trainBtn, false, 'Training...');
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            showStatusMessage(trainStatus, 'Model trained successfully!', 'success');
            updateModelAccuracy(data.accuracy);
            await checkAPIStatus(); // Refresh status
        } else {
            showStatusMessage(trainStatus, `Training failed: ${data.message}`, 'error');
        }
        
    } catch (error) {
        console.error('Training error:', error);
        showStatusMessage(trainStatus, `Training error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
        setButtonState(trainBtn, true, '🔧 Train Model');
    }
}

// Make a single prediction
async function makePrediction() {
    const predictBtn = document.getElementById('predict-btn');
    const predictStatus = document.getElementById('predict-status');
    
    setButtonState(predictBtn, false, 'Classifying...');
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.predicted_category) {
            showStatusMessage(predictStatus, `Classified as: ${data.predicted_category}`, 'success');
            updateLatestPrediction(data);
            addPredictionToLog(data);
        } else {
            showStatusMessage(predictStatus, `Prediction failed: ${data.message}`, 'error');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        showStatusMessage(predictStatus, `Prediction error: ${error.message}`, 'error');
    } finally {
        setButtonState(predictBtn, true, '🔍 Classify Sample');
    }
}

// Toggle simulation
function toggleSimulation() {
    if (isSimulating) {
        stopSimulation();
    } else {
        startSimulation();
    }
}

// Start real-time simulation
function startSimulation() {
    const simulateBtn = document.getElementById('simulate-btn');
    const simulationStatus = document.getElementById('simulation-status');
    
    isSimulating = true;
    setButtonState(simulateBtn, true, '⏹️ Stop Simulation');
    showStatusMessage(simulationStatus, 'Real-time simulation started', 'info');
    
    // Run simulation every 3 seconds
    simulationInterval = setInterval(async () => {
        try {
            const response = await fetch('/api/simulate');
            const data = await response.json();
            
            if (data.predicted_category) {
                updateLatestPrediction(data);
                addPredictionToLog(data);
                
                // Show accuracy in status
                const accuracy = data.correct ? 'Correct' : 'Incorrect';
                showStatusMessage(simulationStatus, `Simulation running - Last: ${accuracy}`, 'info');
            }
            
        } catch (error) {
            console.error('Simulation error:', error);
        }
    }, 3000);
}

// Stop simulation
function stopSimulation() {
    const simulateBtn = document.getElementById('simulate-btn');
    const simulationStatus = document.getElementById('simulation-status');
    
    isSimulating = false;
    setButtonState(simulateBtn, true, '⚡ Start Real-time Simulation');
    showStatusMessage(simulationStatus, 'Simulation stopped', 'info');
    
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
}

// Update latest prediction display
function updateLatestPrediction(prediction) {
    const categoryName = document.querySelector('.category-name');
    const confidenceBadge = document.querySelector('.confidence-badge');
    const predictionTime = document.getElementById('prediction-time');
    const predictionCard = document.querySelector('.prediction-card');
    
    if (categoryName) {
        categoryName.textContent = prediction.predicted_category || prediction.true_category;
    }
    
    if (confidenceBadge) {
        const confidence = Math.round((prediction.confidence || 0) * 100);
        confidenceBadge.textContent = `${confidence}%`;
        
        // Color based on confidence
        if (confidence >= 80) {
            confidenceBadge.style.background = 'var(--success)';
        } else if (confidence >= 60) {
            confidenceBadge.style.background = 'var(--warning)';
        } else {
            confidenceBadge.style.background = 'var(--danger)';
        }
    }
    
    if (predictionTime) {
        const time = new Date(prediction.timestamp).toLocaleTimeString();
        predictionTime.textContent = time;
    }
    
    // Add active animation
    if (predictionCard) {
        predictionCard.classList.add('active');
        setTimeout(() => predictionCard.classList.remove('active'), 2000);
    }
    
    // Update probability bars if available
    if (prediction.probabilities) {
        updateProbabilityBars(prediction.probabilities);
    }
}

// Update probability bars
function updateProbabilityBars(probabilities) {
    const probabilitiesContainer = document.getElementById('prediction-probabilities');
    if (!probabilitiesContainer) return;
    
    probabilitiesContainer.innerHTML = '';
    
    // Sort probabilities by value
    const sortedProbs = Object.entries(probabilities)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5); // Show top 5
    
    sortedProbs.forEach(([category, probability]) => {
        const bar = document.createElement('div');
        bar.className = 'probability-bar';
        
        const percentage = Math.round(probability * 100);
        
        bar.innerHTML = `
            <span class=\"probability-label\">${category}</span>
            <div class=\"probability-track\">
                <div class=\"probability-fill\" style=\"width: ${percentage}%\"></div>
            </div>
            <span class=\"probability-value\">${percentage}%</span>
        `;
        
        probabilitiesContainer.appendChild(bar);
    });
}

// Add prediction to log
function addPredictionToLog(prediction) {
    const predictionsList = document.getElementById('predictions-list');
    if (!predictionsList) return;
    
    // Remove no-predictions message
    const noPredsMsg = predictionsList.querySelector('.no-predictions');
    if (noPredsMsg) {
        noPredsMsg.remove();
    }
    
    // Create prediction item
    const item = document.createElement('div');
    item.className = 'prediction-item slide-in';
    
    const confidence = Math.round((prediction.confidence || 0) * 100);
    const time = new Date(prediction.timestamp).toLocaleTimeString();
    
    item.innerHTML = `
        <div class=\"prediction-text\">
            <div class=\"prediction-category-name\">${prediction.predicted_category || prediction.true_category}</div>
            <div class=\"prediction-timestamp\">${time}</div>
        </div>
        <div class=\"prediction-confidence\">${confidence}%</div>
    `;
    
    // Add to beginning of list
    predictionsList.insertBefore(item, predictionsList.firstChild);
    
    // Keep only last 20 predictions
    const items = predictionsList.querySelectorAll('.prediction-item');
    if (items.length > 20) {
        items[items.length - 1].remove();
    }
    
    // Store prediction for chart update
    latestPredictions.push(prediction);
    if (latestPredictions.length > 50) {
        latestPredictions = latestPredictions.slice(-50);
    }
    
    updateCategoryChart();
}

// Update category distribution chart
function updateCategoryChart() {
    if (!categoryChart || latestPredictions.length === 0) return;
    
    const categories = ['Video Streaming', 'Audio Calls', 'Video Calls', 'Gaming', 'Video Uploads', 'Browsing', 'Texting'];
    const counts = new Array(categories.length).fill(0);
    
    // Count predictions by category
    latestPredictions.forEach(pred => {
        const category = pred.predicted_category || pred.true_category;
        const index = categories.indexOf(category);
        if (index !== -1) {
            counts[index]++;
        }
    });
    
    // Update chart data
    categoryChart.data.datasets[0].data = counts;
    categoryChart.update('none'); // No animation for smoother updates
}

// Start data stream for live updates
function startDataStream() {
    setInterval(async () => {
        try {
            const response = await fetch('/api/data-stream');
            const data = await response.json();
            
            if (data.statistics) {
                updateModelAccuracy(null, data.statistics.average_confidence);
                document.getElementById('total-predictions').textContent = data.statistics.total_predictions || 0;
            }
            
        } catch (error) {
            // Silently handle stream errors
        }
    }, 5000); // Update every 5 seconds
}

// Update model accuracy display
function updateModelAccuracy(accuracy, avgConfidence) {
    const modelAccuracy = document.getElementById('model-accuracy');
    if (!modelAccuracy) return;
    
    if (accuracy !== null && accuracy !== undefined) {
        modelAccuracy.textContent = `${Math.round(accuracy * 100)}%`;
    } else if (avgConfidence !== null && avgConfidence !== undefined) {
        modelAccuracy.textContent = `${Math.round(avgConfidence * 100)}% avg`;
    }
}

// Utility functions
function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.display = show ? 'flex' : 'none';
    }
}

function setButtonState(button, enabled, text) {
    if (!button) return;
    
    button.disabled = !enabled;
    if (text) button.textContent = text;
}

function showStatusMessage(element, message, type = 'info') {
    if (!element) return;
    
    element.textContent = message;
    element.className = `status-message ${type}`;
    
    // Auto-clear after 5 seconds
    setTimeout(() => {
        element.textContent = '';
        element.className = 'status-message';
    }, 5000);
}

// Error handling for fetch requests
window.addEventListener('unhandledrejection', event => {
    console.error('Unhandled promise rejection:', event.reason);
    // Could show user-friendly error message here
});

// Log initialization
console.log('Traffic Classifier Dashboard - JavaScript loaded');
console.log('Dashboard ready for interactions');
