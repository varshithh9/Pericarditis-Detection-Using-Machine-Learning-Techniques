# Pericarditis-Detection-Using-Machine-Learning-Techniques
A complete machine learning system designed to detect Pericarditis using ECG features and clinical data, built with SVM and XGBoost.
The project includes data generation, preprocessing, model training, evaluation, and a fully interactive prediction system.
<hr>
#🚀 Project Overview
Pericarditis is an inflammatory condition of the heart's pericardium that presents with specific ECG and clinical symptoms.
This project uses machine learning to analyse key ECG markers and clinical parameters to automatically predict the presence of pericarditis.<br>

The system performs:
 * Data generation (if no real dataset found)
 * ECG + clinical feature extraction
 * Preprocessing & balancing (SMOTE)
 * Training (SVM & XGBoost)
 * Evaluation (AUC, Confusion Matrix, ROC Curve)
 * Model saving & interactive clinical prediction
<hr>
📊 Dataset Description

The dataset contains 2,500 synthetic medical records with the following features:

<h2> ECG Features</h2>
 * ST Elevation (mV)
 * PR Depression (mV)
 * Heart Rate (bpm)
 * QRS Duration (sec)
 * T-Wave Amplitude (mV)
 * Heart Axis (degrees)

<h2>Clinical Features</h2>
 * CRP Level (mg/L)
 * Age (years)
 * Sex (0 = Female, 1 = Male)

<h2>Label</h2>
 * Diagnosis (0 = Normal, 1 = Pericarditis)


