import pandas as pd
from collections import Counter

# Load the datasets
netflix_data_path = 'data/Netflix_Data.csv'
your_prediction_path = 'data/YourPredictionW24.csv'
netflix_data = pd.read_csv(netflix_data_path)
your_prediction = pd.read_csv(your_prediction_path)

# Analyze Historical Viewing Data
# Prepare the genres in the Netflix data for analysis
netflix_data['Genres'] = netflix_data['Genres'].apply(lambda x: x.split(', ') if pd.notnull(x) else [])
# Count the occurrences of each genre
genre_counts = Counter([genre for sublist in netflix_data['Genres'].tolist() for genre in sublist])
# Identify the top 5 most popular genres
most_popular_genres = [genre for genre, count in genre_counts.most_common(5)]

# Define a function to check if any of a title's genres match the most popular genres
def matches_popular_genres(genres, popular_genres):
    title_genres = genres.split(', ')
    return any(genre in popular_genres for genre in title_genres)

# Apply the function to fill the "Watched40" column in the YourPrediction dataset
your_prediction['Watched40'] = your_prediction['Genres'].apply(lambda x: matches_popular_genres(x, most_popular_genres)).astype(int)

# Display the updated YourPrediction dataset
print(your_prediction)

# Optionally, save the updated dataset to a new CSV file
output_path = 'YourPredictionW24.csv'
your_prediction.to_csv(output_path, index=False)