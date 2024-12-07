import json
import pathlib
from uuid import uuid4

import pandas as pd
import tqdm
from bs4 import BeautifulSoup

from limiting_retrying_session import limiting_retrying_session
from sentiment_analysis_model import SentimentAnalysisModel

DF_DIR = "df"


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
    for i, row in tqdm.tqdm(df.iloc[0:].iterrows()):
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


if __name__ == "__main__":
    session = limiting_retrying_session()
    sentiment_analysis_model = SentimentAnalysisModel()
    df = pd.read_csv(f"{DF_DIR}/raw.csv")
    df = scrape_imdb_data(df, session, sentiment_analysis_model)
    df.to_csv(f"{DF_DIR}/{str(uuid4())}.csv")
