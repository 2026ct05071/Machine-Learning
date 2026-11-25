from sklearn.neighbors import KNeighborsClassifier

def build_model(random_state=42):
    """Create the K-Nearest Neighbors classifier."""
    return KNeighborsClassifier(n_neighbors=5)
