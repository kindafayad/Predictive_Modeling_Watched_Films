import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load the updated datasets
historical_data_df = pd.read_csv('Netflix Assignment Data - Updated-1.csv')
predictions_data_df = pd.read_csv('YourPredictionW24.csv')

# Preprocess genres in both historical and predictions data
for df in [historical_data_df, predictions_data_df]:
    df.fillna('', inplace=True)  # Fill NaNs with empty strings for concatenation
    df['Combined_Genres'] = df[['Genre 1', 'Genre 2', 'Genre 3']].values.tolist()
    df['Combined_Genres'] = df['Combined_Genres'].apply(lambda x: list(set([genre.strip() for genre in x if genre.strip()])))

# Assuming 'Duration' in historical_data_df is in seconds, convert it to minutes
historical_data_df['Duration_minutes'] = historical_data_df['Duration'] / 60

# Aggregate and calculate average viewing duration by all combined genres
average_duration_by_genre = historical_data_df.explode('Combined_Genres').groupby('Combined_Genres')['Duration_minutes'].mean().reset_index()

# Calculate the genre match score for predictions data
unique_genres = average_duration_by_genre['Combined_Genres'].unique().tolist()

def calculate_genre_match_score(row):
    show_genres = row['Combined_Genres']
    match_score = sum(genre in unique_genres for genre in show_genres)
    return match_score

predictions_data_df['Genre_Match_Score'] = predictions_data_df.apply(calculate_genre_match_score, axis=1)

# Prepare the dataset for modeling
predictions_data_df.dropna(subset=['Watched40'], inplace=True)  # Ensure no NaN values in target variable

X = predictions_data_df[['Genre_Match_Score']]  # Features
y = predictions_data_df['Watched40'].astype('int')  # Target variable, ensuring it's integer

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Adjust max_iter to ensure convergence
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)

print(f"Model Evaluation Results:\n- Accuracy: {accuracy}\n- Precision: {precision}\n- Recall: {recall}")
