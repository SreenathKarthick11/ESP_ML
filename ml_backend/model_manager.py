# ml_backend/model_manager.py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class ModelManager:
    def __init__(self, model_type='logistic'):
        if model_type == 'logistic':
            self.model = LogisticRegression()
        elif model_type == 'tree':
            self.model = DecisionTreeClassifier(max_depth=3)  # Example shallow tree
        else:
            raise ValueError("Unsupported model type")

    def train(self, X, y):
        self.model.fit(X, y)

    def get_weights(self):
        if hasattr(self.model, "coef_"):
            weights = self.model.coef_.tolist()[0]
            bias = self.model.intercept_[0]
            return weights, bias
        elif hasattr(self.model, "tree_"):
            # Placeholder: in practice, you'd export the tree structure as rules
            return self.model.get_params(), None
        else:
            return None, None

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        import joblib
        joblib.dump(self.model, path)

    def load(self, path):
        import joblib
        self.model = joblib.load(path)
