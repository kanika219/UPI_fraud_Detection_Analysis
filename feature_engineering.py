from sklearn.base import BaseEstimator, TransformerMixin

class BalanceFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformer that adds balance difference features to detect inconsistencies."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if {'oldbalanceOrg', 'amount', 'newbalanceOrig'}.issubset(X.columns):
            X['balanceOrigDiff'] = X['oldbalanceOrg'] - X['amount'] - X['newbalanceOrig']
        else:
            X['balanceOrigDiff'] = 0.0
        if {'oldbalanceDest', 'amount', 'newbalanceDest'}.issubset(X.columns):
            X['balanceDestDiff'] = X['oldbalanceDest'] + X['amount'] - X['newbalanceDest']
        else:
            X['balanceDestDiff'] = 0.0
        return X
