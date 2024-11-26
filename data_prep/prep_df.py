import pandas as pd
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


def duration_to_minute(x):
    try:
        return round(isodate.parse_duration(x).seconds / 60)
    except:
        return 0


def encode_genres(df):
    for genre in GENRES:
        df[f"is_{genre.lower()}"] = 0
    for i, row in df.iterrows():
        for genre in row.genres.split(","):
            df.loc[i, f"is_{genre.strip().lower()}"] = 1
    return df

def prep_df_from_file(filename="df.csv"):
    df = pd.read_csv(filename)
    df = df.drop(
        ["Unnamed: 0.3", "Unnamed: 0.2", "Unnamed: 0.1", "Unnamed: 0", "availableCountries", "type"],
        axis=1,
        errors="ignore",
    )
    df["runtime"] = df["runtime"].apply(lambda x: duration_to_minute(x))
    df = df[df.runtime > 0]
    df = df[df.genres != "0"]
    df = encode_genres(df)
    df = df.drop(["genres"], axis=1, errors='ignore')
    for field in ["releaseYear", "runtime"]:
        df[field] = df[field].apply(lambda r: (r - df[field].mean()) / df[field].std())
    df = df.join(pd.get_dummies(df.contentRating, prefix="content_rating", dtype=str))
    df = df.drop(["contentRating"], axis=1, errors='ignore')
    df = df.join(pd.get_dummies(df.director, prefix="directed_by", dtype=str))
    df = df.drop(["director"], axis=1, errors='ignore')
    return df