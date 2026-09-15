import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# Enable autologging
mlflow.sklearn.autolog()

# Load and prepare data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

# Train model - MLflow automatically logs everything!
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=10, max_depth=1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation metrics are automatically captured
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")