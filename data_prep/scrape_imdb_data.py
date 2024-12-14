"""
Grabs data from IMDB site

https://github.com/AGnias47/toolbox/blob/main/imdb/watchlist_selector.py
"""

import json
import pathlib

from bs4 import BeautifulSoup
from rainbow_tqdm import tqdm


def get_movie_data(imdb_id, session):
    filepath = f"cache/{imdb_id}.json"
    if pathlib.Path(filepath).exists():
        with open(filepath) as F:
            data = json.load(F)
    else:
        data = json.loads(
            BeautifulSoup(
                session.get(f"https://www.imdb.com/title/{imdb_id}").text, "html.parser"
            )
            .find("script", type="application/ld+json")
            .text
        )
        with open(f"cache/{imdb_id}.json", "w") as F:
            json.dump(data, F)
    return data


def scrape_imdb_data(df, session, sentiment_analysis_model):
    for i, row in tqdm(df.iloc[0:].iterrows()):
        data = get_movie_data(row.imdbId, session)
        try:
            df.loc[i, "director"] = data.get("director")[0].get("name")
        except Exception as e:
            print(f"Error parsing director for {df.title}: {e}")
            pass
        try:
            df.loc[i, "runtime"] = data.get("duration", 0)
        except Exception as e:
            print(f"Error parsing runtime for {df.title}: {e}")
            pass
        try:
            df.loc[i, "contentRating"] = data.get("contentRating")
        except Exception as e:
            print(f"Error parsing contentRating for {df.title}: {e}")
            pass
        try:
            df.loc[i, "sa_desc"] = sentiment_analysis_model.classify(
                data.get("description")
            )
        except Exception as e:
            print(f"Error parsing contentRating for {df.title}: {e}")
            pass
        if i % 100:
            df.to_csv("backup.csv")
    return df
