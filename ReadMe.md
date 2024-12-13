# IMDB Ratings Predictor

Predicts IMDB ratings of movies using available metadata via several regression models.

## Setup

* Install Pytorch via instructions provided at [PyTorch.org](https://pytorch.org/get-started/locally/)
* Install remaining dependencies via `pip install -r requirements.txt`
* Download raw data for streaming services from Kaggle
  * [Amazon Prime](https://www.kaggle.com/datasets/octopusteam/full-amazon-prime-dataset) - save to `raw_data/amazon.csv`
  * [Apple TV](https://www.kaggle.com/datasets/octopusteam/full-apple-tv-dataset) - save to `raw_data/apple.csv`
  * [Hulu](https://www.kaggle.com/datasets/octopusteam/full-hulu-dataset) - save to `raw_data/hulu.csv`
  * [HBO Max](https://www.kaggle.com/datasets/octopusteam/full-hbo-max-dataset) - save to `raw_data/max.csv`
  * [Netflix](https://www.kaggle.com/datasets/octopusteam/full-netflix-dataset) - save to `raw_data/netflix.csv`
* Run `python data_prep/generate_dataframe.py`

## Models

All models are trained with a 70-15-15 split between train, validation, and test datasets. Training involves training the model, running the trained model against the validation dataset, and saving the model to an external file. Testing involves loading the model from an external file and running the trained model against the test dataset.

Two separate datasets are used for testing, one with directors data and one without directors data. The dataset with directors data is only trained on a linear regression model, as the data is a 32k+ one-hot encoding, and thus takes significantly longer and significantly more compute resources.

### Linear Regression

The model without director data runs linear regression, generating separate models for the following methods:
* Closed form
* Closed form Ridge Regression
* Gradient Descent
* Mini-batch Gradient Descent
* Stochastic Gradient Descent

The model with director data only runs using a chunked Mini-batch Gradient Descent model, as this allows the data to be loaded in chunks and not overload a machine's memory.

Models are trained and tested with a script that takes the following arguments:
* `--train` - Train the model. Add `-d` or `--directors` to train the model on the dataset with directors.
* `--test`. Test the saved model. Train must be run first. Add `-d` or `--directors` to test the model trained with directors data.
* `-s` or `--sample`. Generate a list of movies with predicted vs. actual ratings on the model with directors data.

### Ensemble methods

`--train` or `--test` can be run on `models/sk_regressors.py` for a Random Forest (`--rf`), SVC (`--svc`), or XGBoost algorithm (`--xgboost`). Sentiment data can be added to the dataset with `--sentiment`.

### Neural Network

`--train` or `--test` can be run on `models/neural_network[_directors].py`, optionally with `--sentiment` data on the non-directors script.
