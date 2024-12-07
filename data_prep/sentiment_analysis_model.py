"""
https://huggingface.co/blog/sentiment-analysis-python
"""

from transformers import pipeline

class SentimentAnalysisModel:
    def __init__(self):
        self.pipeline = pipeline("sentiment-analysis")

    def classify(self, text):
        if not text:
            return None
        result = self.pipeline(text)[0]
        value = result["score"]
        sign = result["label"]
        if sign == "NEGATIVE":
            value *= -1
        return value