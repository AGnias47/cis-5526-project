# Notes

This was done as part of a class project, so taking notes here on my thought processes for a report.

The initial dataset offered limited information. Metadata included title and rating, which would be ID and Y value. Features were limited, but ended up using genre and release year. In searching for new data in an easily accessible way, wanted to see what else I could get from IMDB. API was paid, but scraping offered additional data, including runtime, content rating, and directors.

Adding this data, was initially worried about directors giving more columns than rows, but this was not the case. However, this did cause memory issues which were not easily managed by some of the classifiers. Linear model was configured to handle this larger dataset by loading the DF in chunks. Regression models used from sk-learn could not handle batched approach. Neural Network was tested on non-directors model and saw promising results, though not better than SK-learn models.

With the remaining time left in the project, it seemed more beneficial to see if another dense feature could be added rather than trying to get the NN to work with the sparse one-hot encoding of directors.

Re-analyzing the available data from the IMDB site for each movie, movie descriptions were available. It was determined to grab the sentiment analysis score of these descriptions and add it as a normalized feature.

In testing this dataset with the linear model, a non-invertible matrix was found for the closed form solutions, but was able to be used with GD-approaches. However, saw slight decline in performance here.