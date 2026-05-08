# 🏏 RCB Mid-Innings Win Predictor (IPL)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)]()
[![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikit-learn)]()
[![Pandas](https://img.shields.io/badge/Data%20Analysis-Pandas-green?logo=pandas)]()

## 📖 Project Overview
This project is an end-to-end Machine Learning pipeline that analyzes historical Indian Premier League (IPL) data to predict the probability of Royal Challengers Bengaluru (RCB) successfully chasing a target. 

Instead of looking at the entire match, this model isolates the **exact 10-over mark (mid-innings)** to evaluate the game state. By calculating mid-game pressure metrics, it functions as a dynamic, real-time win probability engine similar to those used by live sports broadcasters.

## ✨ Key Features
* **Automated Feature Engineering:** Processes raw ball-by-ball data to instantly calculate Current Run Rate (CRR), Required Run Rate (RRR), and Wickets in Hand at exactly ball 10.0.
* **Logistic Regression Model:** Utilizes an 80/20 train-test split to train a highly interpretable probability model that resists overfitting.
* **Interactive Live Predictor:** A built-in Command Line Interface (CLI) allows users to input live match scenarios (Target, Current Score, Wickets Lost) to get instant win probability forecasts.
* **Visual Analytics Dashboard:** Automatically generates a 4-panel `matplotlib` and `seaborn` dashboard illustrating historical trends, RRR distributions, and feature correlations.

## 🧮 The Methodology
Why the 10-over mark? The halfway point of a T20 chase is a critical pressure gauge. The model extracts three core features at this exact moment:
1. **CRR (Current Run Rate):** Pace of scoring in the first 10 overs.
2. **RRR (Required Run Rate):** The pace required for the remaining 10 overs.
3. **Wickets in Hand:** The most critical resource for the death overs.

The algorithm uses **Logistic Regression** because it perfectly translates these interconnected variables into a smooth, mathematically sound probability curve (unlike Random Forests which may overfit small datasets, or Naive Bayes which assumes feature independence).

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn`
* **Data Visualization:** `matplotlib`, `seaborn`

## 🚀 Installation & Setup

1. **Clone the repository** (or download the project folder).
2. **Ensure datasets are present:** Place `matches.csv` and `deliveries.csv` in the root directory.
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

💻 Usage
Run the main script from your terminal:

Bash
python rcb_predictor.py
Execution Flow:

Data Prep: Cleans and merges the datasets.

Training: Trains the ML model and outputs the unseen test accuracy.

Report Generation: Saves a year-by-year historical analysis to rcb_chase_predictor_output.csv.

Live Prediction: Prompts you to enter a custom match scenario in the terminal.

Dashboard: Launches the visual analytics dashboard.

📊 Visual Outputs
When the script finishes execution, it generates a visual dashboard containing:

Win Probability Trend: A line chart showing how RCB's mid-innings dominance has shifted across different IPL seasons.

RRR & Wickets Box Plots: Visual proof of how higher required rates and early wicket losses statistically degrade winning chances.

Correlation Heatmap: The mathematical "brain" of the model, showing the exact Pearson correlation coefficients between match-state variables and the final outcome.

👨‍💻 Author
Utkarsh Pathak 
