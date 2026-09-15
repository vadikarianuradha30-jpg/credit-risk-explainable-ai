import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_processing import load_and_preprocess

def train_and_evaluate(model_type="random_forest"):
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model: {model_type}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    # Save model for later use by explain.py and fairness_audit.py
    joblib.dump(model, f"results/{model_type}_model.pkl")
    print(f"Model saved to results/{model_type}_model.pkl")

    return model, X_test, y_test, demo_test


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="random_forest",
                         choices=["logistic", "random_forest"])
    args = parser.parse_args()

    train_and_evaluate(args.model)