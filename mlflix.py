from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask,jsonify,request,render_template
import flask_cors

path = "NetFlix.csv"
data = pd.read_csv(path)
data = data.fillna('')
data['combined'] = (data['genres']+' '+data['description']+' '+data['director']+' '+data['cast'])

vector = TfidfVectorizer(stop_words="english")
matrix = vector.fit_transform(data['combined'])

similarity =cosine_similarity(matrix)

app = Flask(__name__)
flask_cors.CORS(app)

def recommend(name, num=5):
    name = name.strip()
    matches = data[data['title'].str.lower() == name.lower()]
    if matches.empty:
        return ["Movie not found in database"]
    idx = matches.index[0]
    score = list(enumerate(similarity[idx]))
    score = sorted(score, key=lambda x: x[1], reverse=True)
    top_movie = score[1:num+1]
    movie_idx = [i[0] for i in top_movie]
    print("Received movie:", name)

    return data['title'].iloc[movie_idx].tolist()
@app.route('/recommend', methods=["POST"])
def recommend_api():
    try:
        req_data = request.get_json()
        print("RAW REQUEST:", req_data)

        if not req_data:
            return jsonify({"error": "No JSON received"}), 400

        movie_name = req_data.get('search', '').strip()

        if not movie_name:
            return jsonify({"error": "No movie name provided"}), 400

        results = recommend(movie_name)

        return jsonify({"recommendations": results})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
