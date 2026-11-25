from sklearn.ensemble import RandomForestClassifier

def build_model(random_state=42):
    """Create the Random Forest ensemble classifier."""
    return RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
