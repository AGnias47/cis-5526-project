import pandas as pd
from pathlib import Path
from uuid import uuid4

from limiting_retrying_session import limiting_retrying_session
from sentiment_analysis_model import SentimentAnalysisModel
from scrape_imdb_data import scrape_imdb_data

DF_DIR = "df"
RAW_DATA_DIR = "raw_data"


import isodate

GENRES = {
    "Action",
    "Adult",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Film-Noir",
    "Game-Show",
    "History",
    "Horror",
    "Music",
    "Musical",
    "Mystery",
    "News",
    "Reality-TV",
    "Romance",
    "Sci-Fi",
    "Science Fiction",
    "Short",
    "Sport",
    "TV Movie",
    "Talk-Show",
    "Thriller",
    "War",
    "Western",
}


def combine_raw_data():
    amazon = pd.read_csv(f"{RAW_DATA_DIR}/amazon.csv")
    apple = pd.read_csv(f"{RAW_DATA_DIR}/apple.csv")
    hulu = pd.read_csv(f"{RAW_DATA_DIR}/hulu.csv")
    netflix = pd.read_csv(f"{RAW_DATA_DIR}/netflix.csv")
    hbo_max = pd.read_csv(f"{RAW_DATA_DIR}/max.csv")
    df = pd.concat([amazon, apple, hulu, netflix, hbo_max])
    df = df.fillna(0)
    df = df.astype({"releaseYear": "int32"})
    df = df[df.type == "movie"]
    for attr in ["title", "releaseYear", "imdbAverageRating"]:
        df = df[df[attr] != 0]
    df = df.drop(["imdbNumVotes"], axis=1)
    return df


def duration_to_minute(x):
    try:
        return round(isodate.parse_duration(x).seconds / 60)
    except:
        return 0


def encode_genres(df):
    df = df[df.genres != "0"]
    for genre in GENRES:
        df[f"is_{genre.lower()}"] = 0
    for i, row in df.iterrows():
        for genre in row.genres.split(","):
            df.loc[i, f"is_{genre.strip().lower()}"] = 1
    df = df.drop(["genres"], axis=1, errors="ignore")
    return df


def normalize_numerical_fields(df):
    for field in ["releaseYear", "runtime", "sa_desc"]:
        df[field] = df[field].apply(lambda r: (r - df[field].mean()) / df[field].std())
    return df


def encode_content_ratings(df):
    df = df.join(pd.get_dummies(df.contentRating, prefix="content_rating", dtype=str))
    df = df.drop(["contentRating"], axis=1, errors="ignore")
    return df


def encode_directors(df):
    df = df.join(pd.get_dummies(df.director, prefix="directed_by", dtype=str))
    df = df.drop(["director"], axis=1, errors="ignore")
    return df


def create_no_dirs_df(df, filepath):
    df_no_directors = df.copy(deep=True)
    df_no_directors = df_no_directors.drop(["director"], axis=1, errors="ignore")
    df_no_directors = df_no_directors.fillna(0)
    df_no_directors.to_csv(filepath)


if __name__ == "__main__":
    session = limiting_retrying_session()
    sentiment_analysis_model = SentimentAnalysisModel()
    df = combine_raw_data()
    df = df.drop_duplicates(subset=["title"])
    df = df.drop_duplicates(subset=["imdbId"])
    df = scrape_imdb_data(df, session, sentiment_analysis_model)
    df = df.drop(
        [
            "imdbId",
            "availableCountries",
            "type",
        ],
        axis=1,
        errors="ignore",
    )
    df["runtime"] = df["runtime"].apply(lambda x: duration_to_minute(x))
    df = df[df.runtime > 0]
    df = encode_genres(df)
    df = df.drop("Unnamed: 0", axis=1)
    df = normalize_numerical_fields(df)
    df = encode_content_ratings(df)
    Path(DF_DIR).mkdir(exist_ok=True)
    create_no_dirs_df(df, f"{DF_DIR}/{str(uuid4())[:8]}_no_directors.csv")
    df = encode_directors(df)
    df = df.fillna(0)
    df.to_csv(f"{DF_DIR}/{str(uuid4())[:8]}_directors.csv")
