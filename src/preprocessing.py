from sklearn.base import BaseEstimator, TransformerMixin
import html

class SimpleTextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer, stop_words, stemmer):
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.stemmer = stemmer
    
    def stem_and_lemmatize(self, text):
        # Unescape HTML entities
        text = html.unescape(text)
        # Tokenize the text
        tokens = self.tokenizer(text)
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        # Apply stemming
        stemmed = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self.stem_and_lemmatize(text) for text in X]