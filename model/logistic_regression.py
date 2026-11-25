from sklearn.linear_model import LogisticRegression

def build_model(random_state=42):
    """Create the Logistic Regression classifier."""
    return LogisticRegression(max_iter=2000, random_state=random_state)
