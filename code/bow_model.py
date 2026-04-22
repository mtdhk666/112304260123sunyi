import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

class BOWModel:
    def __init__(self, vectorizer_type='tfidf', max_features=5000, ngram_range=(1, 2)):
        self.vectorizer_type = vectorizer_type
        if vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=ngram_range)
        else:
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    
    def fit(self, X_text, y):
        X = self.vectorizer.fit_transform(X_text)
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X_text):
        X = self.vectorizer.transform(X_text)
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X_text):
        X = self.vectorizer.transform(X_text)
        return self.model.predict(X)
    
    def evaluate(self, X_text, y):
        y_pred_proba = self.predict_proba(X_text)
        return roc_auc_score(y, y_pred_proba)
    
    def cross_validate(self, X_text, y, cv=5):
        X = self.vectorizer.fit_transform(X_text)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')
        return scores
    
    def save(self, model_path='bow_model.pkl', vectorizer_path='bow_vectorizer.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
    
    def load(self, model_path='bow_model.pkl', vectorizer_path='bow_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
