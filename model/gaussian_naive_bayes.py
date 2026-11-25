from sklearn.naive_bayes import GaussianNB

def build_model(random_state=42):
    """Create the Gaussian Naive Bayes classifier."""
    return GaussianNB()
