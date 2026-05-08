import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_filter_data(matches_path, deliveries_path, team_names):
    """Loads the CSV files and filters for the target team's 10-over chase snapshot."""
    print("Loading data...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    print(f"Filtering chases up to the 10th over for {team_names[0]}...")
    team_chases = deliveries[
        (deliveries['inning'] == 2) & 
        (deliveries['batting_team'].isin(team_names)) & 
        (deliveries['over'] <= 9)
    ]
    
    print(f"Total deliveries captured: {len(team_chases)}")
    print(f"Unique matches captured: {team_chases['match_id'].nunique()}\n")
    
    return matches, deliveries, team_chases


def engineer_features(matches, deliveries, team_chases, team_names):    
    """Calculates Target Score, CRR, RRR, Wickets in Hand, and final Win/Loss."""
    print("Starting Feature Engineering...")
    
    # 1. Calculate Target Score
    first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    first_innings.rename(columns={'total_runs': 'target_score'}, inplace=True)
    first_innings['target_score'] = first_innings['target_score'] + 1
    
    # 2. Summarize 10-over snapshot
    chase_stats = team_chases.groupby('match_id').agg(
        runs_at_10=('total_runs', 'sum'),
        wickets_lost_at_10=('is_wicket', 'sum')
    ).reset_index()
    
    # 3. Merge data
    # chase stats[match_id, runs_at_10, wickets_lost_at_10]
    # match_outcomes[match_id, season, winner]
    # first_innings[match_id, target_score]
    match_outcomes = matches[['id', 'season', 'winner']].rename(columns={'id': 'match_id'})
    df = chase_stats.merge(match_outcomes, on='match_id', how='inner')
    df = df.merge(first_innings, on='match_id', how='inner')
    # final df[match_id, runs, wickets, season, winner, target_score]
    
    # 4. Calculate Features
    df['CRR'] = df['runs_at_10'] / 10.0
    df['runs_required'] = df['target_score'] - df['runs_at_10']
    df['RRR'] = df['runs_required'] / 10.0
    df['wickets_in_hand'] = 10 - df['wickets_lost_at_10']
    
    # 5. Target Variable (1 for Win, 0 for Loss)
    df['chase_successful'] = df['winner'].isin(team_names).astype(int)
    
    # Clean missing values
    df.dropna(subset=['CRR', 'RRR', 'wickets_in_hand', 'chase_successful'], inplace=True)
    
    print("Features engineered successfully.\n")
    return df #[match_id, CRR, RRR, wickets_in_hand, chase_success]


def train_model(df):
    """Trains a Logistic Regression model and tests its accuracy."""
    print("Training and Testing Logistic Regression Model...")
    
    X = df[['CRR', 'RRR', 'wickets_in_hand']]
    y = df['chase_successful']
    
    # 1. SPLIT THE DATA (80% Train, 20% Test)
    # random_state=42 ensures we get the same random split every time we run the script
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train the model ONLY on the 80% training data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # 3. Test the model on the 20% unseen data
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"--> Model Accuracy on Unseen Test Data: {accuracy * 100:.2f}%")
    
    # 4. Predict probabilities for the entire dataset so our final charts still work
    df['win_probability'] = model.predict_proba(X)[:, 1]
    
    print("Model trained and probabilities calculated.\n")
    return model, df

def generate_report(df, output_filename):
    """Groups data year-by-year, formats the output, and saves to CSV."""
    print("Generating Final Analysis...")
    
    # Group by season
    year_wise = df.groupby('season').agg(
        matches_chased=('match_id', 'count'),
        avg_crr=('CRR', 'mean'),
        avg_rrr=('RRR', 'mean'),
        avg_wickets_in_hand=('wickets_in_hand', 'mean'),
        avg_win_probability=('win_probability', 'mean')
    ).reset_index()
    
    # Format percentages
    year_wise['avg_win_probability'] = (year_wise['avg_win_probability'] * 100).round(2).astype(str) + '%'
    
    # Cumulative stats
    cumulative_stats = pd.DataFrame([{
        'season': 'Cumulative (All Time)',
        'matches_chased': df['match_id'].count(),
        'avg_crr': df['CRR'].mean(),
        'avg_rrr': df['RRR'].mean(),
        'avg_wickets_in_hand': df['wickets_in_hand'].mean(),
        'avg_win_probability': str(round(df['win_probability'].mean() * 100, 2)) + '%'
    }])
    
    # Combine and round numbers
    final_df = pd.concat([year_wise, cumulative_stats], ignore_index=True)
    final_df['avg_crr'] = final_df['avg_crr'].round(2)
    final_df['avg_rrr'] = final_df['avg_rrr'].round(2)
    final_df['avg_wickets_in_hand'] = final_df['avg_wickets_in_hand'].round(2)
    
    print("\n=======================================================")
    print("  FINAL OUTPUT: MID-INNINGS CHASE WIN PROBABILITY")
    print("=======================================================\n")
    print(final_df.to_string(index=False))
    
    # Save to CSV
    final_df.to_csv(output_filename, index=False)
    print(f"\nSuccess! Saved final report to '{output_filename}'")
    return final_df

def generate_visualizations(df, final_df):
    """Creates visualizations based on the training dataset and final predictions."""
    print("Generating Visualizations...")
    
    # Create a figure with subplots (2 rows, 2 columns)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('RCB Mid-Innings (Over 10) Chase Analysis', fontsize=16, fontweight='bold')

    # --- Plot 1: Line Plot (Win Probability Trend over Seasons) ---
    # Filter out the 'Cumulative' row for the line plot
    trend_data = final_df[final_df['season'] != 'Cumulative (All Time)'].copy()
    # Convert string percentage back to float for plotting
    trend_data['win_prob_float'] = trend_data['avg_win_probability'].str.rstrip('%').astype(float)
    
    axes[0, 0].plot(trend_data['season'], trend_data['win_prob_float'], marker='o', color='blue', linewidth=2)
    axes[0, 0].set_title('Average Predicted Win Probability by Season')
    axes[0, 0].set_xlabel('Season')
    axes[0, 0].set_ylabel('Win Probability (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # --- Plot 2: Box Plot (RRR Distribution: Wins vs Losses) ---
    sns.boxplot(ax=axes[0, 1], x='chase_successful', y='RRR', hue='chase_successful', data=df, palette='pastel', legend=False)
    axes[0, 1].set_title('Required Run Rate (RRR) at Over 10')
    axes[0, 1].set_xticks([0, 1]) 
    axes[0, 1].set_xticklabels(['Loss (0)', 'Win (1)'])
    axes[0, 1].set_xlabel('Match Outcome')
    axes[0, 1].set_ylabel('Required Run Rate')

    # --- Plot 3: Box Plot (Wickets in Hand: Wins vs Losses) ---
    sns.boxplot(ax=axes[1, 0], x='chase_successful', y='wickets_in_hand', hue='chase_successful', data=df, palette='Set2', legend=False)
    axes[1, 0].set_title('Wickets in Hand at Over 10')
    axes[1, 0].set_xticks([0, 1]) 
    axes[1, 0].set_xticklabels(['Loss (0)', 'Win (1)'])
    axes[1, 0].set_xlabel('Match Outcome')
    axes[1, 0].set_ylabel('Wickets in Hand')

    # --- Plot 4: Correlation Heatmap ---
    # Select only the numeric columns for correlation
    numeric_cols = df[['target_score', 'CRR', 'RRR', 'wickets_in_hand', 'chase_successful', 'win_probability']]
    correlation = numeric_cols.corr()
    
    sns.heatmap(correlation, ax=axes[1, 1], annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    # Rotate the x-axis labels 45 degrees and align them to the right
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45, ha='right')

    # Adjust layout so labels don't overlap
    plt.tight_layout(h_pad=4.0, w_pad=3.0)
    
    # Force an extra 15% margin at the bottom of the window specifically for those long labels
    plt.subplots_adjust(bottom=0.15) 
    
    plt.show()
    print("Visualizations complete.\n")

def predict_scenario(model, target, runs_at_10, wickets_lost):
    """Takes a live match scenario and uses the trained model to predict the outcome."""
    print("\n=======================================================")
    print(f"  LIVE PREDICTION: Chasing {target} | Score: {runs_at_10}/{wickets_lost} at 10 Overs")
    print("=======================================================")
    
    # 1. Calculate the features exactly as we did in training
    crr = runs_at_10 / 10.0
    runs_required = target - runs_at_10
    rrr = runs_required / 10.0
    wickets_in_hand = 10 - wickets_lost
    
    print(f"Metrics -> CRR: {crr:.2f} | RRR: {rrr:.2f} | Wickets in Hand: {wickets_in_hand}")
    
    # 2. Format it into a DataFrame so the model recognizes the column names
    scenario_df = pd.DataFrame({
        'CRR': [crr],
        'RRR': [rrr],
        'wickets_in_hand': [wickets_in_hand]
    })
    
    # 3. Predict!
    win_prob = model.predict_proba(scenario_df)[0][1] #[[P(loss), P(win)]]
    
    print(f"--> PREDICTED WIN PROBABILITY: {win_prob * 100:.2f}%\n")

def main():
    """Main execution function that ties all steps together."""
    
    # Configuration
    matches_file = 'matches.csv'
    deliveries_file = 'deliveries.csv'
    team_names = ['Royal Challengers Bangalore', 'Royal Challengers Bengaluru']
    output_file = 'rcb_chase_predictor_output.csv'
    
    # Step 1: Load and Filter
    matches, deliveries, team_chases = load_and_filter_data(matches_file, deliveries_file, team_names)
    
    # Step 2 & 3: Engineer Features
    df = engineer_features(matches, deliveries, team_chases, team_names)
    
    # Step 4: Train Model
    model, df_with_probs = train_model(df)
    
    # Step 5: Generate Report
    final_report = generate_report(df_with_probs, output_file)

    # Step 6: The Grand Finale (Live Predictions!)
    print("\n=======================================================")
    print(f"  Please Enter the required data for prediction: ")
    print("=======================================================")
    target = int(input("Enter target: "))
    score_at_10 = int(input("Enter team score at the end of 10 overs: "))
    wickets_lost = int(input("Enter number of wickets lost: "))
    predict_scenario(model, target, score_at_10, wickets_lost)

    # Step 7: Generate Visualizations 
    generate_visualizations(df_with_probs, final_report)


# Standard Python boilerplate to call the main() function
if __name__ == "__main__":
    main()