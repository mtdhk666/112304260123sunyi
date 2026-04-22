import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib

class Word2VecModel:
    def __init__(self, vector_size=300, window=5, min_count=10, workers=4, 
                 sg=1, epochs=10, classifier='lr'):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.word2vec_model = None
        
        if classifier == 'lr':
            self.classifier = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        elif classifier == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    def train_word2vec(self, sentences):
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs
        )
        return self.word2vec_model
    
    def get_sentence_vector(self, words):
        feature_vector = np.zeros((self.vector_size,), dtype="float32")
        nwords = 0.
        index2word_set = set(self.word2vec_model.wv.index_to_key)
        
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, self.word2vec_model.wv[word])
        
        if nwords > 0:
            feature_vector = np.divide(feature_vector, nwords)
        return feature_vector
    
    def transform(self, tokenized_texts):
        features = np.zeros((len(tokenized_texts), self.vector_size), dtype="float32")
        for i, text in enumerate(tokenized_texts):
            features[i] = self.get_sentence_vector(text)
        return features
    
    def fit(self, tokenized_texts, y, train_word2vec=True):
        if train_word2vec:
            self.train_word2vec(tokenized_texts)
        
        X = self.transform(tokenized_texts)
        self.classifier.fit(X, y)
        return self
    
    def predict_proba(self, tokenized_texts):
        X = self.transform(tokenized_texts)
        return self.classifier.predict_proba(X)[:, 1]
    
    def predict(self, tokenized_texts):
        X = self.transform(tokenized_texts)
        return self.classifier.predict(X)
    
    def evaluate(self, tokenized_texts, y):
        y_pred_proba = self.predict_proba(tokenized_texts)
        return roc_auc_score(y, y_pred_proba)
    
    def save(self, word2vec_path='word2vec.model', classifier_path='word2vec_classifier.pkl'):
        if self.word2vec_model:
            self.word2vec_model.save(word2vec_path)
        joblib.dump(self.classifier, classifier_path)
    
    def load(self, word2vec_path='word2vec.model', classifier_path='word2vec_classifier.pkl'):
        self.word2vec_model = Word2Vec.load(word2vec_path)
        self.classifier = joblib.load(classifier_path)
