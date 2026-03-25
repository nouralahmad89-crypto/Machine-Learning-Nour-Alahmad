import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(movies_path="Data/movies.csv", tags_path="Data/tags.csv"):
    """
    Load movies and tags datasets.
    """
    movies = pd.read_csv(movies_path)
    tags = pd.read_csv(tags_path)
    return movies, tags


def prepare_data(movies, tags, remove_no_genres=True):
    """
    Clean data and create a 'content' column from genres + tags.
    """
    movies = movies.copy()
    tags = tags.copy()

    if remove_no_genres:
        movies = movies[movies["genres"] != "(no genres listed)"].copy()

    # Aggregate tags per movie
    tag_data = tags.groupby("movieId")["tag"].apply(
        lambda x: " ".join(
            sorted(set(str(t).lower().strip() for t in x if pd.notna(t)))
        )
    )

    # Merge tags into movies
    movies = movies.merge(tag_data, on="movieId", how="left")

    # Clean text columns
    movies["genres"] = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    movies["tag"] = movies["tag"].fillna("")
    movies["title"] = movies["title"].fillna("").str.strip()

    # Build content column
    movies["content"] = (movies["genres"] + " " + movies["tag"]).str.strip()
    movies["content"] = movies["content"].fillna("")

    # Reset index so TF-IDF row indices align cleanly
    movies = movies.reset_index(drop=True)

    return movies


def build_tfidf_matrix(movies, stop_words="english", max_features=None):
    """
    Build TF-IDF matrix from the movies['content'] column.
    """
    tfidf = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(movies["content"])
    return tfidf, tfidf_matrix


def build_title_index(movies):
    """
    Create a case-insensitive title-to-index mapping.
    """
    normalized_titles = movies["title"].str.lower().str.strip()
    return pd.Series(movies.index, index=normalized_titles).drop_duplicates()


def recommend_movies(title, movies, tfidf_matrix, indices=None, top_n=5):
    """
    Recommend movies similar to the given title.
    """
    if indices is None:
        indices = build_title_index(movies)

    normalized_title = title.lower().strip()

    if normalized_title not in indices:
        matches = movies[
            movies["title"].str.lower().str.contains(normalized_title, na=False)
        ]["title"].head(10).tolist()

        if matches:
            raise ValueError(
                f"Movie '{title}' not found. Did you mean one of these? {matches}"
            )
        raise ValueError(f"Movie '{title}' not found in the dataset.")

    idx = indices[normalized_title]

    # Compare only the selected movie to all others
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Sort by similarity descending
    similar_indices = sim_scores.argsort()[::-1]

    # Remove the movie itself
    similar_indices = [i for i in similar_indices if i != idx]

    top_indices = similar_indices[:top_n]

    return movies.iloc[top_indices]["title"].tolist()


def main():
    
    if len(sys.argv) < 2:
        print('Usage: python recommendation.py "Movie Title" [top_n]')
        return

    movie_title = sys.argv[1]

    if len(sys.argv) >= 3:
        try:
            top_n = int(sys.argv[2])
        except ValueError:
            print("Error: top_n must be an integer.")
            return
    else:
        top_n = 5

    try:
        movies, tags = load_data()
        movies = prepare_data(movies, tags)
        _, tfidf_matrix = build_tfidf_matrix(movies)
        indices = build_title_index(movies)

        recommendations = recommend_movies(
            movie_title,
            movies,
            tfidf_matrix,
            indices=indices,
            top_n=top_n,
        )

        print(f"\nTop {top_n} recommendations for '{movie_title}':")
        for i, movie in enumerate(recommendations, start=1):
            print(f"{i}. {movie}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()