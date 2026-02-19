This project presents a deep learning-based approach for predicting traffic speeds in urban road networks. The system is designed to forecast short-term traffic conditions (up to 30 minutes ahead) using historical sensor data and a spatial-temporal neural network architecture.

The objective is to build a practical prototype that demonstrates how AI can support smart traffic management and congestion reduction.

Project Overview
------
Urban traffic congestion is influenced by both spatial and temporal factors. Traffic at one road segment often affects nearby segments, and traffic patterns vary based on time (peak hours, weekdays, etc.).

To model this behavior, we implemented a Long Short-Term Temporal Network (LSTTN) that captures:

1.Spatial dependencies between traffic sensors

2.Temporal patterns in time-series traffic data

3.Important signal contributions using attention mechanisms

4.The system predicts traffic speeds for the next:

5 minutes

10 minutes

15 minutes

20 minutes

25 minutes

30 minutes

Dataset
---
1.We used the METR-LA dataset, which contains:

2.Data from 207 traffic loop detectors

3.Speed measurements recorded every 5 minutes

4.Stored in .npz format

The dataset provides real-world traffic patterns and allows the model to learn daily and weekly congestion trends.

Model Architecture
---
The proposed model is based on a multi-stream feature fusion approach:

1. CNN Layers:

Used to extract spatial relationships between neighboring road sensors.

2. LSTM Layers:

Used to capture temporal dependencies in sequential traffic data.

3. Attention Mechanism:

Helps the model focus on the most influential sensors and time steps.

4. Feature Fusion:

Combines spatial and temporal features before final prediction.

This architecture enables multi-step traffic forecasting instead of only predicting the next time step.

System Workflow
---
1.Load and preprocess historical traffic data

2.Normalize and create time-sequence windows

3.Train LSTTN model on spatial-temporal features

4.Generate multi-step predictions

5.Display predictions via Flask web interface

Web Interface
---
1.A Flask-based web application was developed to:

2.Upload .npz dataset files

3.Trigger prediction using the trained model

4.Display predicted vs actual traffic speeds

5.The interface demonstrates how a research model can be connected to a user-facing system.

Technologies Used
---
1.Python

2.TensorFlow 

3.Flask

4.NumPy, Pandas

5.Matplotlib, Seaborn

Model Evaluation
--
1.The model was evaluated using:

2.MAE (Mean Absolute Error)

3.RMSE (Root Mean Squared Error)

These metrics measure prediction accuracy and robustness for multi-step forecasting.

Limitations
--
1.Does not currently incorporate external factors such as weather, accidents, or road construction

2.Requires manual dataset upload (no live API integration)

3.Trained specifically on METR-LA dataset

Future Scope
---
1.Integration with real-time traffic APIs

2.Inclusion of weather and incident data as additional features

3.Deployment for Indian urban traffic systems

4.Extension to real-time adaptive signal control
