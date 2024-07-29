from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the data
books = pd.read_csv('data/Books.csv', sep=';', encoding='ISO-8859-1', on_bad_lines='warn')
ratings = pd.read_csv('data/Ratings.csv', sep=';', encoding='ISO-8859-1', on_bad_lines='warn')

# Add dummy genre data for the example
if 'genre' not in books.columns:
    books['genre'] = 'comedy'  # Replace with actual genre data

# Create a similarity matrix based on genres
count_vectorizer = CountVectorizer()
genre_matrix = count_vectorizer.fit_transform(books['genre'])
similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)

def recommend_books(book_title, top_n=5):
    if book_title not in books['title'].values:
        return pd.DataFrame(columns=['title', 'genre'])

    idx = books[books['title'] == book_title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the first one as it is the book itself
    book_indices = [i[0] for i in sim_scores]
    return books.iloc[book_indices][['title', 'genre']]

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    search_type = None
    search_term = None

    if request.method == 'POST':
        search_type = request.form.get('search_type')
        search_term = request.form.get('search_term')

        if search_type == 'title':
            recommendations = recommend_books(search_term)
        elif search_type == 'genre':
            recommendations = books[books['genre'].str.contains(search_term, case=False)]

    return render_template('index.html', recommendations=recommendations, search_type=search_type, search_term=search_term)

if __name__ == '__main__':
    app.run(debug=True)
