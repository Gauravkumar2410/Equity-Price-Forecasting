# Equity-Price-Forecasting
## Overview
This repository contains the solution for the **Stock Price Predictor** challenge in **HackRush 2023** organized by IIT Gandhinagar. The goal is to predict stock prices for a fictional company named **TGD**, which has three subsidiaries: **TGD Consultancy**, **TGD Automobiles**, and **TGD Power**. The model aims to forecast the values of these stocks, leveraging machine learning algorithms to maximize prediction accuracy and assess uncertainty in the volatile stock market environment.

## Challenge Description
In this hypothetical scenario, you are an experienced stock market dealer using AI to forecast stock prices and make informed buy/sell decisions for profit maximization. The dataset provided includes information about the internal workings of TGD and its subsidiaries, enabling participants to explore correlations and dependencies among the stock prices of these three subsidiaries.

### Key Tasks
1. **Predict Stock Prices**: Forecast future values of the three TGD subsidiary stocks.
2. **Uncertainty Estimation**: Calculate an uncertainty score alongside each predicted value to measure prediction confidence.
3. **Data Visualization**: Create advanced visualizations to provide deeper insights into model predictions, including attention distributions, model workings, and other informative metrics.

### Bonus Points
- **Uncertainty Estimation (30 points)**: Generate an uncertainty score for each subsidiary’s stock price prediction. This score helps assess confidence in the predictions.
- **Advanced Visualizations (20 points)**: Additional points will be awarded for informative and visually appealing charts (e.g., feature attention distribution, model interpretability using LIME, etc.) that improve the understanding of the model’s performance and data characteristics.

## Dataset
The dataset contains historical daily stock prices of **TGD Consultancy**, **TGD Automobiles**, and **TGD Power** subsidiaries, along with relevant features that may influence their values. The timeline in this dataset is hypothetical and does not match real-world time.

## Requirements
The challenge allows the use of any freely available resources, open-source libraries, pre-trained models, or other features to improve the accuracy of the predictions.

Useful resources:
- **Libraries information**: [GitHub - Resources of ML](https://github.com/dwipddalal/Resources_of_ml/blob/main/ML%20Material.pdf)
- **Hugging Face Transformers for Time Series**: [Hugging Face Documentation](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
- **Flow-Forecast (Time-Series Forecasting)**: [Flow-Forecast GitHub](https://github.com/AIStream-Peelout/flow-forecast)

**Hints**: Key terms to explore include *Multi-task*, *Multi-variate time-series forecasting*, *Transformers*, *Uncertainty Prediction* (e.g., Dropout, Bayesian Inference).

## Approach
1. **Data Preprocessing**: Clean and prepare the dataset, handling missing values, scaling, and encoding where necessary.
2. **Exploratory Data Analysis (EDA)**: Perform analysis to identify trends, correlations, and feature importance.
3. **Feature Engineering**: Generate additional features that may help improve model predictions.
4. **Model Selection and Training**:
   - Utilize multi-task, multi-variate time-series forecasting models.
   - Consider models like Transformers for capturing dependencies and attention across time series data.
5. **Uncertainty Estimation**:
   - Techniques like Dropout and Bayesian Inference are applied to estimate uncertainty.
   - An uncertainty score is calculated for each subsidiary’s stock price prediction.
6. **Evaluation Metrics**:
   - **Root Mean Square Error (RMSE)**: Used to evaluate model accuracy.
   - The **final score** is a combination of Leaderboard score, Uncertainty score, and Visualization score.

## Evaluation and Judging Criteria
- **Prediction Accuracy (Leaderboard Score)**: 
  - Position 1 on the leaderboard = 100 points
  - Position 2 on the leaderboard = 90 points
  - Position 3 on the leaderboard = 80 points, and so forth.
- **Uncertainty Estimation**: 30 points if correctly calculated for each of the three stock prices individually (Binary Grading).
- **Visualization Quality**: Up to 20 points for advanced visualizations showcasing insights into the model’s behavior and data patterns.

## Submission Requirements
- **Deliverable**: A Jupyter notebook (or `.py` files) containing the final model submitted to the leaderboard. The notebook should be reproducible, with code for each prediction task, including the uncertainty evaluations and visualizations.
- **Metric for Leaderboard**: Predictions are evaluated based on RMSE, calculated as the sum of the values for all three stocks (i.e., `TGD Automobiles + TGD Consultancy + TGD Power`).

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Stock-Price-Predictor.git
   cd Stock-Price-Predictor
