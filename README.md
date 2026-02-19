This project presents a deep learning-based approach for predicting traffic speeds in urban road networks. The system is designed to forecast short-term traffic conditions (up to 30 minutes ahead) using historical sensor data and a spatial-temporal neural network architecture.

The objective is to build a practical prototype that demonstrates how AI can support smart traffic management and congestion reduction.

Project Overview
------
Urban traffic congestion is influenced by both spatial and temporal factors. Traffic at one road segment often affects nearby segments, and traffic patterns vary based on time (peak hours, weekdays, etc.).

To model this behavior, we implemented a Long Short-Term Temporal Network (LSTTN) that captures:

Spatial dependencies between traffic sensors

Temporal patterns in time-series traffic data

Important signal contributions using attention mechanisms

The system predicts traffic speeds for the next:

5 minutes

10 minutes

15 minutes

20 minutes

25 minutes

30 minutes

Dataset
---
We used the METR-LA dataset, which contains:

Data from 207 traffic loop detectors

Speed measurements recorded every 5 minutes

Stored in .npz format

The dataset provides real-world traffic patterns and allows the model to learn daily and weekly congestion trends.

Model Architecture
---
The proposed model is based on a multi-stream feature fusion approach:

1. CNN Layers

Used to extract spatial relationships between neighboring road sensors.

2. LSTM Layers

Used to capture temporal dependencies in sequential traffic data.

3. Attention Mechanism

Helps the model focus on the most influential sensors and time steps.

4. Feature Fusion

Combines spatial and temporal features before final prediction.

This architecture enables multi-step traffic forecasting instead of only predicting the next time step.

System Workflow
---
Load and preprocess historical traffic data

Normalize and create time-sequence windows

Train LSTTN model on spatial-temporal features

Generate multi-step predictions

Display predictions via Flask web interface

Web Interface
---
A Flask-based web application was developed to:

Upload .npz dataset files

Trigger prediction using the trained model

Display predicted vs actual traffic speeds

The interface demonstrates how a research model can be connected to a user-facing system.

Technologies Used
---
Python

TensorFlow / Keras

Flask

NumPy, Pandas

Matplotlib, Seaborn

Model Evaluation
--
The model was evaluated using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

These metrics measure prediction accuracy and robustness for multi-step forecasting.

Limitations
--
Does not currently incorporate external factors such as weather, accidents, or road construction

Requires manual dataset upload (no live API integration)

Trained specifically on METR-LA dataset

Future Scope
---
Integration with real-time traffic APIs

Inclusion of weather and incident data as additional features

Deployment for Indian urban traffic systems

Extension to real-time adaptive signal control
