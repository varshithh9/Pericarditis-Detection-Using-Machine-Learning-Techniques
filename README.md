# Pericarditis Detection Using Machine Learning

This project implements a machine learning system to detect Pericarditis using ECG features and clinical data. The system uses Support Vector Machine (SVM) and XGBoost models to predict pericarditis based on key ECG markers and clinical indicators.

## Project Overview

Pericarditis is an inflammation of the pericardium and early detection is crucial. This project automates the detection process using ECG features, clinical markers, and machine learning.

## Dataset Description

The dataset contains 2,500 synthetic ECG-based medical records.

### ECG Features
- ST Elevation (mV)
- PR Depression (mV)
- Heart Rate (bpm)
- QRS Duration (sec)
- T-Wave Amplitude (mV)
- Heart Axis (degrees)

### Clinical Features
- CRP Level (mg/L)
- Age (years)
- Sex (0 = Female, 1 = Male)

### Label
- Diagnosis (0 = Normal, 1 = Pericarditis)

## Preprocessing Steps

- StandardScaler used for normalization  
- SMOTE applied to balance normal and pericarditis cases  
- Train-test split at 75% training, 25% testing  

## Machine Learning Models

### Support Vector Machine (SVM - RBF Kernel)
- Handles non-linear decision boundaries  
- Suitable for medical datasets  

### XGBoost Classifier
- High accuracy and stability  
- Handles imbalanced data well  
- Provides feature importance  

## Model Evaluation

Models were evaluated using:
- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-Score
- AUC (Area Under ROC Curve)
- Confusion Matrix
- ROC Curve

XGBoost achieved higher accuracy, better AUC, and fewer false negatives than SVM.

## Interactive Clinical Prediction System

The project includes a command-line prediction tool where users enter ECG and clinical parameters.  
The system outputs:
- Diagnosis (Normal / Pericarditis)
- Confidence score
- Risk level (Low / Moderate / High)
- Basic clinical recommendations

## Installation

### Clone the repository
```bash
git clone https://github.com/your-username/pericarditis-detection.git
cd pericarditis-detection
```

### Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python pericarditis_ml_model.py
```

This will:
- Generate synthetic data
- Train the models
- Show performance graphs
- Save the trained model
- Start the clinical prediction tool

## Saved Output Files

- pericarditis_detection_model.pkl  
- feature_scaler.pkl  
- model_configuration.json  
- All generated graphs (PNG files)

## Key Features Learned by the Model

- ST Elevation (increases)
- PR Depression (increases)
- CRP Level (increases)
- Heart Rate (increases)
- T-Wave Amplitude (decreases)

## Model Output Visualizations

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve
![ROC Curve](images/roc_curve.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

### Feature Distribution (Histograms)
![Feature Distribution](images/feature_distribution.png)

### Correlation Heatmap
![Correlation Matrix](images/correlation_matrix.png)

## Future Improvements

- Use deep learning for raw ECG signals  
- Add web or mobile interface  
- Train with real hospital ECG datasets  
- Deploy as an API for clinical systems  

## Contributing

Contributions are welcome through issues and pull requests.

## License

This project is available under the MIT License.
