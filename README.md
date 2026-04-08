# Movie-recommendation-project
A content-based movie recommendation system that suggests similar movies based on a user’s selection. It uses cosine similarity on movie metadata and is built with Python and Streamlit, displaying recommended titles along with posters fetched from the TMDB API.
<br>
My project is deployed in server, here's the link to that: https://mrs-hrithik-58185457f593.herokuapp.com/


# 🎬 Movie Recommendation System

> *A content-based movie recommendation engine that suggests five similar films for any selected title — powered by NLP, cosine similarity, and live TMDB poster fetching — deployed as an interactive Streamlit web app.*

---

## 📖 Project Overview

Ever finished a great film and had no idea what to watch next? This project solves that. Using the **TMDB 5000 Movie Dataset**, we build a content-based recommender that understands what a movie is actually *about* — its plot, genres, keywords, cast, and director — and finds the five most similar films from a catalogue of thousands.

The system is entirely **content-based**: no user ratings, no collaborative filtering, just the DNA of the movie itself. The final app fetches real poster artwork from The Movie Database API and presents recommendations in a clean five-column visual layout.

---

## 🗂️ Project Structure

```
movie-recommendation-system/
│
├── tmdb_5000_movies.csv           # Raw movie metadata (4803 entries)
├── tmdb_5000_credits.csv          # Cast & crew data
├── movie_recommender_code.ipynb   # Full NLP + ML notebook
├── movies.pkl                     # Serialised movie DataFrame
├── similarity.pkl                 # Serialised cosine similarity matrix
└── app.py                         # Streamlit web application
```

---

## 🧭 Project Journey

### Stage 1 — Loading & Merging the Dataset

Two separate CSV files form the foundation of this project:

| File | Contents |
|---|---|
| `tmdb_5000_movies.csv` | Metadata: title, overview, genres, keywords |
| `tmdb_5000_credits.csv` | Cast and crew information per film |

The two datasets were merged on the `title` column, producing a unified dataframe. From the full set of columns, only seven were retained for the recommendation engine:

```
movie_id · title · overview · genres · keywords · cast · crew
```

This deliberate feature selection keeps the model focused on content-relevant signals only — runtime, budget, and revenue tell us nothing about what a film *feels like*.

---

### Stage 2 — Data Cleaning

*Raw data from real-world scraped sources is never clean.* Before any NLP work could begin, two structural issues were resolved:

- **Null rows** — `dropna()` removed any entries missing critical fields
- **Duplicate rows** — checked and confirmed zero duplicates in the merged dataset

---

### Stage 3 — Feature Extraction & Parsing

*This was the most intricate stage.* The `genres`, `keywords`, `cast`, and `crew` columns were not clean string lists — they were **stringified Python objects** (JSON-like dicts stored as text). Four custom parsing functions were written to unpack them:

**`convert(obj)`** — Parses genres and keywords, extracting just the `name` field from each dict:
```python
ast.literal_eval(obj)  # "string that looks like a list" → actual list
```

**`convert3(obj)`** — Parses cast but keeps only the **top 3 actors** to prevent noise from large ensemble casts diluting the similarity signal.

**`fetch_director(obj)`** — Scans the crew list and extracts only the person with `"job": "Director"`, ignoring producers, editors, and everyone else.

**Overview split** — The plain text plot summary was tokenised into a list of words via `.split()` to match the list format of the other features.

---

### Stage 4 — Tag Engineering

With all features in clean list form, spaces within multi-word names were removed — a crucial step to prevent *"Sam Worthington"* from being split into unrelated tokens *"sam"* and *"worthington"*:

```python
[i.replace(" ", "") for i in x]  # "Sam Worthington" → "SamWorthington"
```

All five features were then **concatenated into a single `tags` column** — one long string per movie representing its full content identity:

```
tags = overview + genres + keywords + cast + crew
```

The final `new_df` kept just three columns: `movie_id`, `title`, and `tags`.

The tags were then:
1. Joined into a single string with spaces
2. Lowercased for consistency

---

### Stage 5 — Stemming

Before vectorisation, **Porter Stemming** (via NLTK) was applied to every word in the tags. This collapses morphological variants to their root form:

```
"loved" → "love"    "fighting" → "fight"    "action" → "action"
```

This prevents the model from treating *"fight"* and *"fighting"* as separate, unrelated concepts — increasing the accuracy of similarity matching.

---

### Stage 6 — Vectorisation (Bag of Words)

*Tags are human-readable text. Cosine similarity requires numbers.* The `CountVectorizer` from scikit-learn converts each movie's tag string into a high-dimensional numerical vector:

```python
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
# Shape: (4806, 5000)
```

- `max_features=5000` — caps the vocabulary at the 5,000 most frequent meaningful words
- `stop_words='english'` — removes filler words like *"the"*, *"a"*, *"is"* that carry no content signal
- The result is a **4806 × 5000 matrix** — each movie is now a point in 5,000-dimensional space

---

### Stage 7 — Cosine Similarity

With every movie as a vector, **cosine similarity** measures how closely aligned two movies are in that vector space:

```python
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)
# Shape: (4806, 4806)
```

The output is a **4806 × 4806 matrix** where `similarity[i][j]` is a score from 0 to 1 indicating how similar movie `i` and movie `j` are in content. A score of 1 means identical content; 0 means no overlap.

*Why cosine and not Euclidean distance?* Cosine similarity ignores the magnitude of vectors — a short film with a brief overview won't be penalised against a franchise film with a dense tag string.

---

### Stage 8 — Recommendation Logic

The core recommendation function:

```python
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
```

1. Look up the row index of the selected movie
2. Retrieve its similarity scores against all other movies
3. Sort descending by score
4. Skip index 0 (the movie itself) and return positions 1–5

**Test result on *Avatar*:**
```
Titan A.E. · Small Soldiers · Independence Day · Ender's Game · Aliens vs Predator: Requiem
```
All five share the sci-fi, alien, and action DNA of Avatar — the system works as intended.

---

### Stage 9 — Serialisation

Both the processed DataFrame and the similarity matrix were serialised with `pickle`:

```python
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
```

Precomputing and saving the full similarity matrix means the Streamlit app loads it once at startup and serves instant recommendations — no recomputing at query time.

---

### Stage 10 — Streamlit Web Application

`app.py` wraps the recommender in a clean, interactive UI:

- **Dropdown** — populated from all movie titles in `movies.pkl`
- **Recommend button** — triggers the similarity lookup for the selected title
- **5-column poster grid** — for each recommended film, the TMDB API is called in real-time to fetch the official poster image

```python
def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=...")
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
```

A fallback placeholder image is shown if a poster is unavailable, preventing broken UI states.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| NLP | NLTK (PorterStemmer) |
| Vectorisation | scikit-learn (CountVectorizer) |
| Similarity | scikit-learn (cosine_similarity) |
| Serialisation | pickle |
| Web framework | Streamlit |
| Poster API | TMDB REST API |

---

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

**2. Install dependencies**

```bash
pip install streamlit pandas numpy scikit-learn nltk requests
```

**3. Download NLTK data** *(first run only)*

```python
import nltk
nltk.download('punkt')
```

**4. Run the notebook** to generate `movies.pkl` and `similarity.pkl`

```bash
jupyter notebook movie_recommender_code.ipynb
```

**5. Launch the app**

```bash
streamlit run app.py
```

---

## 🔮 How to Use

1. Select any movie from the dropdown (4,800+ titles available)
2. Click **Recommend**
3. Five similar movies appear with their official poster artwork

---

## 📊 System Specs

| Parameter | Value |
|---|---|
| Dataset size | 4,806 movies |
| Vocabulary size | 5,000 top terms |
| Vector dimensions | 5,000 |
| Similarity matrix size | 4,806 × 4,806 |
| Recommendations returned | 5 per query |
| Approach | Content-based filtering |

---

## 🔭 Future Improvements

- Swap `CountVectorizer` for **TF-IDF** to downweight overly common terms
- Use **Word2Vec** or sentence embeddings for semantic (not just keyword) similarity
- Add a **hybrid layer** combining content similarity with collaborative filtering from user ratings
- Cache TMDB API calls to reduce latency on repeat recommendations
- Add genre and year filters to narrow recommendations

---

## 👨‍💻 Author

Built as an end-to-end NLP and recommendation systems project, demonstrating the full pipeline from raw text data to a live, poster-rich web application.
