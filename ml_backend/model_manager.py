from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import joblib

class ModelManager:
    def __init__(self, model_type='logistic', max_depth=3):
        self.model_type = model_type
        self.max_depth = max_depth
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == 'logistic':
            return LogisticRegression()
        elif self.model_type == 'tree':
            return DecisionTreeClassifier(max_depth=self.max_depth)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def set_model(self, model_type, max_depth=None):
        self.model_type = model_type
        if max_depth is not None:
            self.max_depth = max_depth
        self.model = self._init_model()

    def train(self, X, y):
        self.model.fit(X, y)

    def get_weights(self):
        if hasattr(self.model, "coef_"):
            weights = self.model.coef_.tolist()[0]
            bias = self.model.intercept_[0]
            return weights, bias
        elif hasattr(self.model, "tree_"):
            return self.model.get_params(), None
        else:
            return None, None

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)