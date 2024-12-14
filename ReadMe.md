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
  * This will take about a day to run on first pass, as data is pulled from the IMDB website for each movie

## Datasets

There are three datasets used in this project

* Baseline - Uses features based on Release Year, Content Rating, Runtime, and Genre
* Sentiment - Same as Baseline, but adds Sentiment Analysis score on movie description
* Directors - Same as Baseline, but adds Directors as a one-hot encoding. Memory intensive (1.8 GB at rest)

## Models

All models are trained with a 70-15-15 split between train, validation, and test datasets. Training involves training the model, running the trained model against the validation dataset, and saving the model to an external file. Testing involves loading the model from an external file and running the trained model against the test dataset.

### Linear Regression

Implements the following methods:
* Closed form
* Closed form Ridge Regression
* Gradient Descent
* Mini-batch Gradient Descent
* Stochastic Gradient Descent

The Director dataset only runs using a chunked Mini-batch Gradient Descent model, as this allows the data to be loaded in chunks and not overload a machine's memory. The Sentiment dataset can only be used on the Gradient Descent methods.

Models are trained and tested with the `models/linear_regression.py` script that takes the following arguments:
* `--directors` - Use the Directors dataset.
* `--sentiment` - Use the Sentiment datset.
* `--train` - Train the model. If no dataset specified, use Baseline
* `--test`. Test the saved model. Train must be run first for the dataset. If no dataset specified, use Baseline
* `-s` or `--sample`. Generate a list of movies with predicted vs. actual ratings using the model generated from the Directors dataset

### Ensemble methods / SVM

Uses Scikit and Scikit-adjacent libraries to run regression models. Includes

* Random Forest
* SVR
* XG Boost Regressor

Only runs for Baseline and Sentiment datsets. Models are trained and tested with the `models/sk_regressors.py` script that takes the following arguments:
* `--sentiment` - Use the Sentiment datset.
* `--train` - Train the model. If no dataset specified, use Baseline
* `--test`. Test the saved model. Train must be run first for the dataset. If no dataset specified, use Baseline
* `--rf` - Train or test the Random Forest model
* `--svr` - Train or test the SVR model
* `--xgboost` - Train or test the XG Boost model

### Neural Network

Uses a PyTorch Feedforward Neural Network. Hyperparameters are managed directly in each script.

#### Baseline and Sentiment data

Models are trained and tested with the `models/neural_network.py` script that takes the following arguments:

* `--sentiment` - Use the Sentiment datset.
* `--train` - Train the model. If no dataset specified, use Baseline
* `--test`. Test the saved model. Train must be run first for the dataset. If no dataset specified, use Baseline

#### Directors data

The neural network using directors data has not been refined and this is entirely ineffective. Further hyperparameter tuning needs to be performed for this model to be useful.

The model is trained and tested with the `models/neural_network_directors.py` script that takes the following arguments:

* `--train` - Train the model.
* `--test`. Test the saved model. Train must be run first for the dataset
