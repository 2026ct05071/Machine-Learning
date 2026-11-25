from sklearn.tree import DecisionTreeClassifier

def build_model(random_state=42):
    """Create the Decision Tree classifier."""
    return DecisionTreeClassifier(random_state=random_state)
